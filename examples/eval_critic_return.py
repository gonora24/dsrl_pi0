#!/usr/bin/env python
"""Evaluate a trained DSRL-NA critic against realized returns from pi0.5 rollouts.

For each rollout on the specified LIBERO task:
  1. Run the pi0.5 policy to collect a trajectory.
  2. At every pi0 query boundary record (obs_dict, full_action_chunk).
  3. Collect per-env-step rewards throughout the episode.
  4. Compute the realized discounted return for each chunk: the sum of all
     per-env-step rewards from that chunk's first env step to end of episode,
     discounted per env step. This matches the Bellman target used in
     update_na_critic (rewards.shape[-1] = chunk_size, discount^k per step).
  5. Query the na_critic for Q(obs, action_chunk) and take the ensemble mean.
  6. Report MSE and MAE between critic predictions and realized returns.

The query frequency is auto-derived from the checkpoint config:
  - Chunked training (action_shape[1] == pi0_action_horizon) → query_freq = pi0_action_horizon
  - Non-chunked training (action_shape[1] == 1)             → query_freq = 5 (standard)
Pass --query_freq explicitly to override.
"""

import argparse
import json
import os
import pathlib
import sys
import types
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# XLA GPU performance flag (matches train_sim.py).
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.policies import policy_config
from openpi.training import config as openpi_config

from examples.train_sim import (
    CHECKPOINTS,
    load_pi0_checkpoint,
    load_norm_stats_for_checkpoint,
)
from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos
from jaxrl2.agents.pixel_sac.dsrl_na_learner import DSRLNALearner, get_value

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10

LIBERO_MAX_TIMESTEPS = {
    "libero_spatial": 250,
    "libero_object": 300,
    "libero_goal": 300,
    "libero_10": 550,
    "libero_90": 400,
}

# Standard non-chunked query frequency matching run_eval_pi05_libero.sh.
_DEFAULT_QUERY_FREQ = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_ckpt_config(ckpt_dir: str) -> dict:
    ckpt_path = pathlib.Path(ckpt_dir)
    config_path = ckpt_path.parent / f"{ckpt_path.name}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Checkpoint config not found: {config_path}\n"
            "Only checkpoints saved with DSRLNALearner.save_checkpoint (which writes a "
            "companion *_config.json) are supported."
        )
    with open(config_path) as f:
        return json.load(f)


def _derive_query_freq(cfg: dict) -> int:
    """Infer query_freq from the checkpoint config."""
    action_shape = cfg["action_shape"]   # e.g. [1, 1, 7] or [1, 50, 7]
    pi0_action_horizon = cfg["pi0_action_horizon"]
    if action_shape[1] == pi0_action_horizon:
        return pi0_action_horizon  # chunked: execute every full horizon
    return _DEFAULT_QUERY_FREQ        # non-chunked: standard 5-step replan


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _resolve_openpi_config(pi0_checkpoint: str) -> openpi_config.TrainConfig:
    if pi0_checkpoint in CHECKPOINTS:
        return openpi_config.get_config(CHECKPOINTS[pi0_checkpoint]["config"])
    return openpi_config.get_config("pi0_libero")


def _compute_chunk_realized_returns(
    step_rewards: list,
    chunk_start_indices: list,
    discount: float,
) -> np.ndarray:
    """Compute the realized discounted return for each chunk.

    For chunk t starting at env step s_t, the realized return is:
        G_t = sum_{k=0}^{T-s_t-1} gamma^k * step_rewards[s_t + k]

    This matches the Bellman target used in update_na_critic:
        rewards_for_bootstrap = sum_{k=0}^{chunk_size-1} gamma^k * r_k
        bootstrap_discount    = gamma^chunk_size
        target_q = rewards_for_bootstrap + bootstrap_discount * mask * next_q

    Both the per-env-step discounting and the choice of rewards follow the
    chunk_reward training convention: r_k = -1 if not done at env step k,
    r_k = 0 if done (success) at env step k.

    Args:
        step_rewards:       per-env-step rewards for the full episode
        chunk_start_indices: index of the first env step of each chunk
        discount:           per-env-step discount factor gamma

    Returns:
        Array of shape (num_chunks,) with realized return per chunk.
    """
    # First compute cumulative discounted return from every env step to end.
    T = len(step_rewards)
    G_step = np.zeros(T, dtype=np.float64)
    running = 0.0
    for k in reversed(range(T)):
        running = step_rewards[k] + discount * running
        G_step[k] = running

    # Pick the value at the first env step of each chunk.
    return np.array([G_step[s] for s in chunk_start_indices], dtype=np.float64)


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def _run_rollout(
    env,
    init_state,
    policy,
    agent: DSRLNALearner,
    variant,
    rng: jax.Array,
    max_timesteps: int,
    query_freq: int,
    pi0_action_horizon: int,
    pi0_action_dim: int,
    critic_action_steps: int,
) -> dict:
    """Run one episode and return data needed for critic evaluation.

    The na_critic is trained via update_na_critic which uses per-env-step rewards
    with per-env-step discounting:
        rewards_for_bootstrap = sum_{k=0}^{chunk_size-1} gamma^k * r_k
        bootstrap_discount    = gamma^chunk_size
        target_q = rewards_for_bootstrap + bootstrap_discount * mask * next_q

    So Q(s_t, a_chunk_t) ≈ sum_{k=0}^{∞} gamma^k * r_{env_step(t*query_freq + k)}

    To compare correctly we collect a reward at every env step (following the
    chunk_reward training convention: -1 if not done, 0 if done/success) and
    record the first env-step index of each chunk.  The realized return at chunk t
    is then the sum of all per-env-step rewards from that index onward, discounted
    per env step — the same quantity the critic is trained to predict.

    Args:
        critic_action_steps: Number of action steps expected by the na_critic
            (= pi0_action_horizon for chunked mode, 1 for non-chunked).

    Returns:
        dict with keys:
            obs_dicts           – list of critic-format obs dicts, one per chunk
            actions             – list of critic-format action arrays (1, critic_action_steps, 7)
            step_rewards        – per-env-step rewards for the full episode
            chunk_start_indices – index into step_rewards for the first env step of each chunk
            is_success          – bool
    """
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    obs_dicts = []
    actions = []
    step_rewards = []       # one reward per env step, same convention as chunk_reward training
    chunk_start_indices = []  # step_rewards index where each chunk begins
    pi0_actions = None
    reward = 0.0
    done = False

    for t in range(max_timesteps):
        if t % query_freq == 0:
            # Record obs and action chunk at the start of this chunk.
            curr_image = obs_to_img(obs, variant)           # (H, W, 3)
            qpos = obs_to_qpos(obs, variant)                # (8,)
            obs_dict = {
                "pixels": curr_image[np.newaxis, ..., np.newaxis],  # (1, H, W, 3, 1)
                "state":  qpos[np.newaxis, ..., np.newaxis],         # (1, 8, 1)
            }

            obs_pi0 = obs_to_pi_zero_input(obs, variant)
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (1, pi0_action_horizon, pi0_action_dim))
            pi0_actions = policy.infer(obs_pi0, noise=noise)["actions"]  # (H, 7)

            # Full action chunk for the chunked na_critic.
            critic_action = pi0_actions[None, :critic_action_steps, :]  # (1, K, 7)

            obs_dicts.append(obs_dict)
            actions.append(critic_action)
            chunk_start_indices.append(t)   # env-step index of this chunk's first step

        action_t = pi0_actions[t % query_freq]
        obs, reward, done, _ = env.step(action_t)

        # Per-env-step reward following the chunk_reward training convention:
        #   -1 if the episode is still running, 0 if it just completed (success).
        # Failed episodes accumulate -1 for every step up to max_timesteps.
        step_rewards.append(0.0 if done else -1.0)

        if done:
            break

    is_success = bool(reward == 1.0)
    return {
        "obs_dicts":           obs_dicts,
        "actions":             actions,
        "step_rewards":        step_rewards,
        "chunk_start_indices": chunk_start_indices,
        "is_success":          is_success,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    if args.libero_suite not in LIBERO_MAX_TIMESTEPS:
        valid = ", ".join(sorted(LIBERO_MAX_TIMESTEPS))
        raise ValueError(f"Unknown suite {args.libero_suite!r}. Choose one of: {valid}")

    # --- Read checkpoint config before loading the full agent ---
    print(f"Reading checkpoint config from {args.dsrl_checkpoint} …", flush=True)
    cfg = _read_ckpt_config(args.dsrl_checkpoint)
    resize_image = cfg["obs_shapes"]["pixels"][1]   # [B, H, W, C, 1] → H
    pi0_action_horizon = cfg["pi0_action_horizon"]
    action_shape = cfg["action_shape"]              # [B, n_steps, 7]
    critic_action_steps = action_shape[1]           # 1 (non-chunked) or H (chunked)
    chunked = critic_action_steps == pi0_action_horizon

    # Derive query_freq unless the user overrides.
    if args.query_freq is not None:
        query_freq = args.query_freq
        print(f"query_freq={query_freq} (user-specified)", flush=True)
    else:
        query_freq = _derive_query_freq(cfg)
        mode = "chunked" if chunked else "non-chunked"
        print(f"query_freq={query_freq} (derived from checkpoint: {mode} mode)", flush=True)

    max_timesteps = LIBERO_MAX_TIMESTEPS[args.libero_suite]

    # --- Load LIBERO task ---
    print(f"Loading {args.libero_suite} task {args.task_id} …", flush=True)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_suite]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(task, resolution=256, seed=args.seed)
    print(f"Task: {task_description}", flush=True)

    variant = types.SimpleNamespace(
        env="libero",
        task_description=task_description,
        resize_image=resize_image,
    )

    # --- Load pi0 policy ---
    openpi_train_config = _resolve_openpi_config(args.pi0_checkpoint)
    checkpoint_dir = load_pi0_checkpoint(args.pi0_checkpoint)
    norm_stats = load_norm_stats_for_checkpoint(args.pi0_checkpoint)
    policy = policy_config.create_trained_policy(
        openpi_train_config, checkpoint_dir, norm_stats=norm_stats
    )
    pi0_action_dim = policy.action_dim
    print(
        f"Loaded pi0 policy from {checkpoint_dir} "
        f"(horizon={pi0_action_horizon}, action_dim={pi0_action_dim})",
        flush=True,
    )

    # --- Load DSRL-NA critic ---
    print(f"Loading DSRL-NA critic from {args.dsrl_checkpoint} …", flush=True)
    agent = DSRLNALearner.restore_from_checkpoint_dir(args.dsrl_checkpoint)
    discount = args.discount if args.discount is not None else agent.discount
    print(
        f"Critic loaded. discount={discount}, "
        f"critic_action_steps={critic_action_steps} ({'chunked' if chunked else 'non-chunked'})",
        flush=True,
    )

    # --- Rollout loop ---
    rng = jax.random.PRNGKey(args.seed)
    all_q_preds = []
    all_realized = []
    rollout_results = []

    for rollout_id in tqdm(range(args.num_rollouts), desc="rollouts"):
        init_state = init_states[rollout_id % len(init_states)]
        rng, rollout_key = jax.random.split(rng)

        traj = _run_rollout(
            env,
            init_state,
            policy,
            agent,
            variant,
            rollout_key,
            max_timesteps,
            query_freq,
            pi0_action_horizon,
            pi0_action_dim,
            critic_action_steps,
        )

        step_rewards = traj["step_rewards"]
        chunk_start_indices = traj["chunk_start_indices"]
        if len(chunk_start_indices) == 0:
            print(f"  rollout {rollout_id}: empty trajectory, skipping.", flush=True)
            continue

        # Realized return at chunk t = discounted sum of ALL per-env-step rewards
        # from the first env step of chunk t to the end of the episode.
        realized_returns = _compute_chunk_realized_returns(
            step_rewards, chunk_start_indices, discount
        )

        # Query critic at each chunk boundary.
        q_preds = []
        for obs_dict, action in zip(traj["obs_dicts"], traj["actions"]):
            q_ensemble = get_value(
                jnp.asarray(action, dtype=jnp.float32),
                {k: jnp.asarray(v) for k, v in obs_dict.items()},
                agent._na_critic,
            )  # (num_qs, 1)
            q_preds.append(float(jnp.mean(q_ensemble)))

        q_preds = np.array(q_preds)
        errors = q_preds - realized_returns
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))

        # --- Q-value trend metrics ---
        n = len(q_preds)
        q_initial     = float(q_preds[0])
        q_final       = float(q_preds[-1])
        q_improvement = q_final - q_initial
        q_slope       = float(np.polyfit(np.arange(n), q_preds, 1)[0]) if n > 1 else 0.0

        all_q_preds.extend(q_preds.tolist())
        all_realized.extend(realized_returns.tolist())

        episode_return = float(np.sum(step_rewards))
        result = {
            "rollout_id":           rollout_id,
            "is_success":           traj["is_success"],
            "episode_return":       episode_return,
            "num_chunks":           len(chunk_start_indices),
            "num_env_steps":        len(step_rewards),
            "mse":                  mse,
            "mae":                  mae,
            "q_preds":              q_preds.tolist(),
            "realized_returns":     realized_returns.tolist(),
            "step_rewards":         step_rewards,
            "chunk_start_indices":  chunk_start_indices,
            "q_initial":            q_initial,
            "q_final":              q_final,
            "q_improvement":        q_improvement,
            "q_slope":              q_slope,
        }
        rollout_results.append(result)
        print(
            f"  rollout {rollout_id + 1}/{args.num_rollouts}: "
            f"success={traj['is_success']}, env_steps={len(step_rewards)}, "
            f"MSE={mse:.4f}, MAE={mae:.4f}",
            flush=True,
        )
        print(
            f"    Q trend:  initial={q_initial:+.3f}  final={q_final:+.3f}"
            f"  improvement={q_improvement:+.3f}  slope={q_slope:+.4f}",
            f"  q_preds={q_preds.tolist()}",
            flush=True,
        )

    env.close()

    # --- Aggregate metrics ---
    all_q_preds = np.array(all_q_preds)
    all_realized = np.array(all_realized)
    all_errors = all_q_preds - all_realized
    agg_mse = float(np.mean(all_errors ** 2))
    agg_mae = float(np.mean(np.abs(all_errors)))
    success_rate = float(np.mean([r["is_success"] for r in rollout_results]))

    # --- Q-trend summary split by success / failure ---
    successful = [r for r in rollout_results if r["is_success"]]
    failed     = [r for r in rollout_results if not r["is_success"]]

    def _trend_summary(group: list) -> dict:
        if not group:
            return {}
        return {
            "mean_q_slope":       float(np.mean([r["q_slope"]       for r in group])),
            "mean_q_improvement": float(np.mean([r["q_improvement"] for r in group])),
            "mean_q_initial":     float(np.mean([r["q_initial"]     for r in group])),
            "mean_q_final":       float(np.mean([r["q_final"]       for r in group])),
        }

    trend_success = _trend_summary(successful)
    trend_failed  = _trend_summary(failed)

    def _print_trend(label: str, n: int, stats: dict):
        print(f"--- Q-Value Trend: {label} (N={n}) ---")
        if not stats:
            print("  (no rollouts in this group)")
            return
        print(f"  mean Q slope (per chunk):  {stats['mean_q_slope']:+.4f}")
        print(f"  mean Q improvement:        {stats['mean_q_improvement']:+.4f}")
        print(f"  mean Q initial:            {stats['mean_q_initial']:+.4f}")
        print(f"  mean Q final:              {stats['mean_q_final']:+.4f}")

    print("\n" + "=" * 70)
    print(f"Task {args.task_id} — {task_description}")
    print(f"Suite:          {args.libero_suite}")
    print(f"DSRL ckpt:      {args.dsrl_checkpoint}")
    print(f"pi0 ckpt:       {args.pi0_checkpoint}")
    print(f"Rollouts:       {len(rollout_results)}/{args.num_rollouts}")
    print(f"Success rate:   {success_rate:.1%}")
    print(f"Discount (γ):   {discount}")
    print(f"query_freq:     {query_freq}")
    print(f"chunked critic: {chunked}")
    print(f"--- Critic vs Realized Return ---")
    print(f"Aggregate MSE:  {agg_mse:.6f}")
    print(f"Aggregate MAE:  {agg_mae:.6f}")
    _print_trend("Successful Rollouts", len(successful), trend_success)
    _print_trend("Failed Rollouts",     len(failed),     trend_failed)
    print("=" * 70)

    summary = {
        "task_id":          args.task_id,
        "task_description": task_description,
        "libero_suite":     args.libero_suite,
        "dsrl_checkpoint":  args.dsrl_checkpoint,
        "pi0_checkpoint":   args.pi0_checkpoint,
        "num_rollouts":     args.num_rollouts,
        "completed_rollouts": len(rollout_results),
        "success_rate":     success_rate,
        "discount":         discount,
        "query_freq":       query_freq,
        "chunked":          chunked,
        "critic_action_steps": critic_action_steps,
        "aggregate_mse":    agg_mse,
        "aggregate_mae":    agg_mae,
        "q_trend_successful": trend_success,
        "q_trend_failed":     trend_failed,
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "rollouts":         rollout_results,
    }

    if args.output_json:
        out_path = pathlib.Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Results written to {out_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    suites = ", ".join(sorted(LIBERO_MAX_TIMESTEPS))
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained DSRL-NA critic against realized returns from "
            "pi0.5 rollouts on a LIBERO task."
        )
    )
    parser.add_argument(
        "--dsrl_checkpoint",
        required=True,
        help="Path to the checkpointN directory (e.g. .../run_xyz/checkpoint941).",
    )
    parser.add_argument(
        "--pi0_checkpoint",
        default="pi05_libero",
        help=(
            "pi0 checkpoint preset or local path. "
            "Presets: " + ", ".join(sorted(CHECKPOINTS.keys()))
        ),
    )
    parser.add_argument(
        "--task_id",
        default=59,
        type=int,
        help="LIBERO task index within the suite (default: 59).",
    )
    parser.add_argument(
        "--libero_suite",
        default="libero_90",
        choices=sorted(LIBERO_MAX_TIMESTEPS.keys()),
        help=f"LIBERO task suite ({suites}). Default: libero_90.",
    )
    parser.add_argument(
        "--num_rollouts",
        default=10,
        type=int,
        help="Number of evaluation rollouts (default: 10).",
    )
    parser.add_argument(
        "--discount",
        default=None,
        type=float,
        help="Discount factor γ. If omitted, read from the agent checkpoint.",
    )
    parser.add_argument(
        "--query_freq",
        default=None,
        type=int,
        help=(
            "Env steps per pi0 chunk. Auto-derived from the checkpoint if omitted: "
            "chunked training → pi0_action_horizon; non-chunked → 5."
        ),
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed (default: 0).")
    parser.add_argument(
        "--output_json",
        default="",
        help="Optional path to write JSON results.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.output_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_json = (
            f"logs/eval_critic_return_{args.libero_suite}_task{args.task_id}_{ts}.json"
        )
    evaluate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
