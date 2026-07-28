#!/usr/bin/env python
"""Collect action distributions across multiple checkpoints for coverage analysis.

Supports pi0/pi0.5 (JAX) and XVLA (PyTorch) checkpoints in the same run.
Prefix XVLA checkpoint paths with ``xvla:`` to select the XVLA loader:

    --checkpoints pi05_libero xvla:2toINF/X-VLA-Libero xvla:/local/path/to/xvla

Two collection modes:
  rollout   – full env rollouts; records every executed action.
  openloop  – fix one observation, sample K noise seeds; records predicted chunks.

Outputs (per checkpoint, per mode):
  <output_dir>/data/actions_rollout_<ckpt>.npy   shape (total_steps, action_dim)
  <output_dir>/data/actions_openloop_<ckpt>.npy  shape (K, horizon * action_dim)
  <output_dir>/data/meta.json

Usage:
  python -m examples.collect_action_coverage \\
      --checkpoints pi05_libero xvla:2toINF/X-VLA-Libero \\
      --num_rollouts 20 \\
      --num_noise_seeds 200 \\
      --mode both \\
      --libero_suite libero_90 \\
      --task_id 18
"""

import argparse
import json
import os
import pathlib
import sys
import types
from datetime import datetime

import jax
import numpy as np
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.policies import policy_config
from openpi.training import config as openpi_config

from examples.train_sim import CHECKPOINTS, load_norm_stats_for_checkpoint, load_pi0_checkpoint
from examples.train_utils_sim import obs_to_pi_zero_input, prepare_libero_episode_for_xvla

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10
MAX_TIMESTEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ---------------------------------------------------------------------------
# Rollout (Mode A)
# ---------------------------------------------------------------------------

def _rollout_no_video(env, obs, variant, policy, rng, max_steps, query_freq,
                      action_horizon, action_dim):
    """Run one rollout, collect executed actions; no video recording."""
    all_actions = []
    pi_actions = None

    for t in range(max_steps):
        if t % query_freq == 0:
            obs_pi = obs_to_pi_zero_input(obs, variant)
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (1, action_horizon, action_dim))
            pi_actions = policy.infer(obs_pi, noise=noise)["actions"]

        action = pi_actions[t % query_freq]
        all_actions.append(np.array(action))
        obs, _reward, done, _info = env.step(action)

        if done:
            return True, all_actions, rng

    return False, all_actions, rng


def collect_rollout_actions(env, init_states, variant, policy, rng,
                            num_rollouts, max_steps, query_freq,
                            action_horizon, action_dim):
    """Run *num_rollouts* full rollouts and concatenate all executed actions.

    Returns:
        actions (np.ndarray): shape (total_steps, action_dim)
        n_success (int)
    """
    all_actions = []
    n_success = 0

    for i in tqdm(range(num_rollouts), desc="rollouts", unit="ep"):
        init_state = init_states[i % len(init_states)]
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        success, ep_actions, rng = _rollout_no_video(
            env, obs, variant, policy, rng,
            max_steps, query_freq, action_horizon, action_dim,
        )
        all_actions.extend(ep_actions)
        n_success += int(success)
        print(
            f"  rollout {i+1}/{num_rollouts}: success={success}, "
            f"steps={len(ep_actions)}, total_so_far={len(all_actions)}",
            flush=True,
        )

    return np.array(all_actions, dtype=np.float32), n_success


# ---------------------------------------------------------------------------
# Open-loop noise sweep (Mode B)
# ---------------------------------------------------------------------------

def collect_openloop_actions(env, init_state, variant, policy, rng,
                              num_noise_seeds, action_horizon, action_dim):
    """Fix one env observation, sample *num_noise_seeds* noise vectors.

    Returns:
        actions (np.ndarray): shape (num_noise_seeds * action_horizon, action_dim)
            All predicted action chunks, stacked row-wise.
    """
    # Get a fixed starting observation
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    obs_pi = obs_to_pi_zero_input(obs, variant)

    all_chunks = []
    for _ in tqdm(range(num_noise_seeds), desc="noise seeds", unit="seed"):
        rng, key = jax.random.split(rng)
        noise = jax.random.normal(key, (1, action_horizon, action_dim))
        chunk = policy.infer(obs_pi, noise=noise)["actions"]
        all_chunks.append(np.array(chunk))   # (action_horizon, action_dim)

    chunks = np.stack(all_chunks, axis=0)          # (K, horizon, action_dim)
    per_seed_flat = chunks.reshape(len(all_chunks), -1)  # (K, horizon*action_dim)

    return per_seed_flat.astype(np.float32)


# ---------------------------------------------------------------------------
# XVLA helpers
# ---------------------------------------------------------------------------

_XVLA_PREFIX = "xvla:"


def _is_xvla(ckpt_name: str) -> bool:
    return ckpt_name.startswith(_XVLA_PREFIX)


def _xvla_model_path(ckpt_name: str) -> str:
    return ckpt_name[len(_XVLA_PREFIX):]


def _load_xvla_policy(ckpt_name: str, device: str = "auto"):
    """Load XVLAPolicy from a HuggingFace repo or local path."""
    from examples.xvla_policy import XVLAPolicy
    import jax

    if device == "auto":
        devices = jax.devices()
        device = "cuda" if any(d.platform == "gpu" for d in devices) else "cpu"

    model_path = _xvla_model_path(ckpt_name)
    print(f"  Loading XVLA from {model_path!r} on {device}", flush=True)
    policy = XVLAPolicy.from_pretrained(
        model_path,
        device=device,
        domain_id=3,
        steps=10,
    )
    return policy


# ---------------------------------------------------------------------------
# XVLA rollout (Mode A)
# ---------------------------------------------------------------------------

def _rollout_no_video_xvla(env, obs, task_description, policy, rng,
                            max_steps, query_freq):
    """One XVLA rollout.  Uses numpy noise and stateful proprio.

    Returns:
        success (bool), all_actions (list of 1-D float32 arrays of len 7)
    """
    policy.reset()
    all_actions = []
    xvla_actions = None
    obs_with_prompt = {**obs, "prompt": task_description}

    for t in range(max_steps):
        if t % query_freq == 0:
            noise_np = rng.standard_normal(
                (1, policy.action_horizon, policy.action_dim)
            ).astype(np.float32)
            xvla_actions = policy.infer(
                obs_with_prompt,
                noise=noise_np,
                proprio_from_step=query_freq - 1,
            )["actions"]   # (action_horizon, 7)

        action = xvla_actions[t % query_freq]
        all_actions.append(np.array(action, dtype=np.float32))
        obs, _reward, done, _info = env.step(action)
        obs_with_prompt = {**obs, "prompt": task_description}

        if done:
            return True, all_actions

    return False, all_actions


def collect_rollout_actions_xvla(env, init_states, task_description, policy, rng,
                                  num_rollouts, max_steps, query_freq):
    """Run *num_rollouts* XVLA rollouts; concatenate all executed 7-D actions.

    Returns:
        actions (np.ndarray): shape (total_steps, 7)
        n_success (int)
    """
    all_actions = []
    n_success = 0

    for i in tqdm(range(num_rollouts), desc="rollouts (xvla)", unit="ep"):
        init_state = init_states[i % len(init_states)]
        env.reset()
        env.set_init_state(init_state)
        obs = prepare_libero_episode_for_xvla(env)

        success, ep_actions = _rollout_no_video_xvla(
            env, obs, task_description, policy, rng,
            max_steps, query_freq,
        )
        all_actions.extend(ep_actions)
        n_success += int(success)
        print(
            f"  rollout {i+1}/{num_rollouts}: success={success}, "
            f"steps={len(ep_actions)}, total_so_far={len(all_actions)}",
            flush=True,
        )

    return np.array(all_actions, dtype=np.float32), n_success


# ---------------------------------------------------------------------------
# XVLA open-loop noise sweep (Mode B)
# ---------------------------------------------------------------------------

def collect_openloop_actions_xvla(env, init_state, task_description, policy, rng,
                                   num_noise_seeds):
    """Fix one XVLA observation; sample *num_noise_seeds* independent noise vectors.

    Returns:
        actions (np.ndarray): shape (num_noise_seeds, action_horizon * 7)
            Each row = one noise seed's complete predicted action chunk (flattened).
    """
    env.reset()
    env.set_init_state(init_state)
    obs = prepare_libero_episode_for_xvla(env)
    obs_with_prompt = {**obs, "prompt": task_description}

    all_chunks = []
    for _ in tqdm(range(num_noise_seeds), desc="noise seeds (xvla)", unit="seed"):
        policy.reset()   # clear stateful proprio → independent samples
        noise_np = rng.standard_normal(
            (1, policy.action_horizon, policy.action_dim)
        ).astype(np.float32)
        chunk = policy.infer(obs_with_prompt, noise=noise_np)["actions"]
        all_chunks.append(np.array(chunk, dtype=np.float32))   # (horizon, 7)

    chunks = np.stack(all_chunks, axis=0)               # (K, horizon, 7)
    return chunks.reshape(len(all_chunks), -1).astype(np.float32)   # (K, horizon*7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    args = parse_args(argv)

    if args.libero_suite not in MAX_TIMESTEPS:
        raise ValueError(
            f"Unknown suite {args.libero_suite!r}. "
            f"Choose from: {', '.join(sorted(MAX_TIMESTEPS))}"
        )
    max_steps = MAX_TIMESTEPS[args.libero_suite]
    run_rollout = args.mode in ("rollout", "both")
    run_openloop = args.mode in ("openloop", "both")

    # --- output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.output_dir) / f"{timestamp}_actioncoverage_{args.libero_suite}_task_{args.task_id}"
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}", flush=True)

    # --- load task ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_suite]()
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    init_states = task_suite.get_task_init_states(args.task_id)
    print(f"Task {args.task_id}: {task_description}", flush=True)
    print(f"Init states available: {len(init_states)}", flush=True)

    # --- build env ---
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)

    meta = {
        "timestamp": timestamp,
        "libero_suite": args.libero_suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_description": task_description,
        "seed": args.seed,
        "mode": args.mode,
        "num_rollouts": args.num_rollouts,
        "num_noise_seeds": args.num_noise_seeds,
        "query_freq": args.query_freq,
        "checkpoints": [],
    }

    # Shared numpy RNG for XVLA (seeded once; advanced per checkpoint).
    np_rng = np.random.default_rng(args.seed)

    for ckpt_name in args.checkpoints:
        print(f"\n{'='*60}", flush=True)
        print(f"Checkpoint: {ckpt_name}", flush=True)
        print(f"{'='*60}", flush=True)

        safe_name = ckpt_name.replace("/", "_").replace(":", "_")
        is_xvla = _is_xvla(ckpt_name)

        # ------------------------------------------------------------------
        # Load policy
        # ------------------------------------------------------------------
        if is_xvla:
            policy = _load_xvla_policy(ckpt_name)
            action_horizon = policy.action_horizon
            action_dim = policy.env_action_dim   # 7 – env-facing executed dim
            print(f"action_horizon={action_horizon}  action_dim={action_dim}  "
                  f"(internal noise_dim={policy.action_dim})", flush=True)
        else:
            cfg = openpi_config.get_config(
                CHECKPOINTS[ckpt_name]["config"] if ckpt_name in CHECKPOINTS else "pi0_libero"
            )
            checkpoint_dir = load_pi0_checkpoint(ckpt_name)
            norm_stats = load_norm_stats_for_checkpoint(ckpt_name)
            policy = policy_config.create_trained_policy(
                cfg, checkpoint_dir, norm_stats=norm_stats
            )
            print(f"Loaded policy from {checkpoint_dir}", flush=True)
            action_horizon = cfg.model.action_horizon
            action_dim = policy.action_dim
            print(f"action_horizon={action_horizon}  action_dim={action_dim}", flush=True)

        variant = types.SimpleNamespace(
            env="libero",
            task_description=task_description,
            resize_image=-1,
        )
        jax_rng = jax.random.PRNGKey(args.seed)

        ckpt_meta = {
            "name": ckpt_name,
            "policy_type": "xvla" if is_xvla else "pi0",
            "action_dim": action_dim,
            "action_horizon": action_horizon,
        }

        # ------------------------------------------------------------------
        # Mode A – full rollouts
        # ------------------------------------------------------------------
        if run_rollout:
            print(f"\n[{ckpt_name}] Mode A: full rollouts (N={args.num_rollouts})", flush=True)
            if is_xvla:
                actions_rollout, n_success = collect_rollout_actions_xvla(
                    env, init_states, task_description, policy, np_rng,
                    args.num_rollouts, max_steps, args.query_freq,
                )
            else:
                jax_rng, sub_rng = jax.random.split(jax_rng)
                actions_rollout, n_success = collect_rollout_actions(
                    env, init_states, variant, policy, sub_rng,
                    args.num_rollouts, max_steps, args.query_freq,
                    action_horizon, action_dim,
                )

            save_path = data_dir / f"actions_rollout_{safe_name}.npy"
            np.save(str(save_path), actions_rollout)
            print(
                f"  Saved {actions_rollout.shape} → {save_path}  "
                f"(success rate {n_success}/{args.num_rollouts})",
                flush=True,
            )
            ckpt_meta["rollout"] = {
                "file": str(save_path.relative_to(out_dir)),
                "shape": list(actions_rollout.shape),
                "n_success": n_success,
                "success_rate": n_success / args.num_rollouts,
            }

        # ------------------------------------------------------------------
        # Mode B – open-loop noise sweep
        # ------------------------------------------------------------------
        if run_openloop:
            print(
                f"\n[{ckpt_name}] Mode B: open-loop sweep (K={args.num_noise_seeds})",
                flush=True,
            )
            if is_xvla:
                actions_openloop = collect_openloop_actions_xvla(
                    env, init_states[0], task_description, policy, np_rng,
                    args.num_noise_seeds,
                )
            else:
                jax_rng, sub_rng = jax.random.split(jax_rng)
                actions_openloop = collect_openloop_actions(
                    env, init_states[0], variant, policy, sub_rng,
                    args.num_noise_seeds, action_horizon, action_dim,
                )

            save_path = data_dir / f"actions_openloop_{safe_name}.npy"
            np.save(str(save_path), actions_openloop)
            print(f"  Saved {actions_openloop.shape} → {save_path}", flush=True)
            ckpt_meta["openloop"] = {
                "file": str(save_path.relative_to(out_dir)),
                "shape": list(actions_openloop.shape),
            }

        meta["checkpoints"].append(ckpt_meta)

    env.close()

    # Write metadata
    meta_path = data_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nMeta written to {meta_path}", flush=True)
    print(f"\nDone. All data in: {out_dir}", flush=True)

    return str(out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    valid_ckpts = sorted(CHECKPOINTS.keys())
    parser = argparse.ArgumentParser(
        description="Collect action distributions for coverage analysis."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["pi05_libero"],
        help=(
            "One or more checkpoint identifiers. "
            "Pi0 checkpoints: names from CHECKPOINTS dict or local paths. "
            "XVLA checkpoints: prefix with 'xvla:', e.g. 'xvla:2toINF/X-VLA-Libero' "
            "or 'xvla:/local/path'. "
            f"Known pi0 names: {', '.join(valid_ckpts)}"
        ),
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["rollout", "openloop", "both"],
        help="Collection mode (default: both)",
    )
    parser.add_argument(
        "--num_rollouts",
        default=20,
        type=int,
        help="Number of full rollouts per checkpoint (Mode A)",
    )
    parser.add_argument(
        "--num_noise_seeds",
        default=200,
        type=int,
        help="Number of noise seeds for open-loop sweep (Mode B)",
    )
    parser.add_argument(
        "--query_freq",
        default=5,
        type=int,
        help="Re-plan every N env steps (Mode A)",
    )
    parser.add_argument(
        "--libero_suite",
        default="libero_90",
        choices=sorted(MAX_TIMESTEPS.keys()),
    )
    parser.add_argument(
        "--task_id",
        default=18,
        type=int,
        help="LIBERO task index (default 18 = frying-pan task)",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.environ.get("OUTPUT_DIR", "./logs"), "action_coverage"
        ),
        help="Root directory for output data (default: ./logs/action_coverage)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
