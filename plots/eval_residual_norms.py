"""Infer residual-prediction norms from saved DSRL checkpoints.

The residual actor (`AutoregressiveActorTransformer` in
jaxrl2/networks/actor_transformer.py) is trained with one of two residual
modes:

  - ``use_actor_diff``:      each AR step after the first predicts a bounded
                              action increment that is accumulated into the
                              sampled action chunk.
  - ``use_actor_diff_mean``: each AR step after the first predicts a
                              (mean, log_std) increment that is accumulated
                              into the running TanhNormal parameters.

Neither raw per-step residual was logged to W&B during training, so this
script reconstructs them from checkpoints by running on-policy rollouts
through pi0 and differencing consecutive AR-chunk steps of the outputs that
``sample_and_log_prob_diff`` / ``sample_and_log_prob_diff_mean`` already
return (no changes to actor_transformer.py needed):

  - ``use_actor_diff``:      residual[t]      = actions[:, t]   - actions[:, t-1]
  - ``use_actor_diff_mean``: residual_mean[t] = mu[:, t]        - mu[:, t-1]
                              residual_log_std[t] = log_std[:, t] - log_std[:, t-1]

for AR steps t = 1..T-1 (there is no residual at t=0, the first token is
sampled directly from the TanhNormal head; consecutive steps are always
compared to their immediate predecessor, never the first step to the last).

Each residual vector (per AR step, per batch element) is normed over the
action dimension, then averaged over both the chunk (the T-1 AR steps) and
the batch, so every query yields a single scalar. For every checkpoint found
under ``--run_dir`` this script runs ``--num_rollouts`` on-policy episodes
(querying the residual actor every ``query_frequency`` env steps, exactly
like examples/eval_noise_7dims.py) and writes one row per (checkpoint,
rollout, query) to a long-format CSV with columns ``step, metric, value`` so
that downstream plotting can compute means/CI bands per checkpoint step
(see plots/plot_residual_norms_multi_tasks.py).
"""

import argparse
import json
import pathlib
import re
import sys

import jax
import numpy as np
import pandas as pd

from LIBERO.libero.libero import benchmark
from examples.train_sim import CHECKPOINTS, load_pi0_checkpoint, load_norm_stats_for_checkpoint
from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos, _prepare_pi0_noise
from jaxrl2.tests.finite_differences import create_libero_env
from jaxrl2.utils.general_utils import AttrDict
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from openpi.policies import policy_config
from openpi.training import config as openpi_config

CHECKPOINT_DIR_RE = re.compile(r"^checkpoint(\d+)$")


def _infer_use_actor_diff_mean(run_name: str) -> bool:
    """Decide use_actor_diff_mean from the run directory name.

    Older checkpoints' saved configs predate the ``use_actor_diff_mean``
    field, so it must be inferred: runs named like ``..._aractor_diff_mean``
    use the (mean, log_std) residual head, while diff-actor runs without
    "mean" in the name (e.g. ``..._diffaractor_...``) use the plain action
    residual head instead.
    """
    return "mean" in run_name.lower()


def discover_checkpoints(run_dir: pathlib.Path, steps=None):
    """Return a list of (step, ckpt_dir, config) sorted ascending.

    Only directories named ``checkpoint<N>`` are considered. Each one is
    paired with hyperparameters read from its own companion
    ``checkpoint<N>_config.json`` (written by PixelSACLearner.save_checkpoint)
    if present, or -- since training runs only started saving that file
    partway through, so typically only the *last* checkpoint of a run has
    one -- falls back to whichever other companion config exists in
    ``run_dir``. All checkpoints within one run share the same architecture,
    so any one config is a valid template for the whole run. Checkpoints in
    runs with no companion config at all are skipped, since there's nothing
    to reconstruct hyperparameters from.

    Every resulting config additionally gets ``use_actor_diff_mean`` filled
    in via ``_infer_use_actor_diff_mean`` if the field is missing (see that
    function's docstring).
    """
    ckpt_dirs = []
    for p in sorted(run_dir.iterdir()):
        if not p.is_dir():
            continue
        m = CHECKPOINT_DIR_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if steps is not None and step not in steps:
            continue
        ckpt_dirs.append((step, p))
    ckpt_dirs.sort(key=lambda x: x[0])

    configs_by_step = {}
    for step, p in ckpt_dirs:
        config_path = run_dir / f"{p.name}_config.json"
        if config_path.exists():
            with open(config_path) as f:
                configs_by_step[step] = json.load(f)

    template_step = max(configs_by_step) if configs_by_step else None

    found = []
    for step, p in ckpt_dirs:
        if step in configs_by_step:
            cfg = dict(configs_by_step[step])
        elif template_step is not None:
            print(f"[info] {p} has no companion config json; reusing config "
                  f"from checkpoint{template_step} (same run, same architecture)",
                  file=sys.stderr)
            cfg = dict(configs_by_step[template_step])
        else:
            print(f"[warn] skipping {p} (no companion config json found anywhere "
                  f"in {run_dir})", file=sys.stderr)
            continue

        if "use_actor_diff_mean" not in cfg:
            inferred = _infer_use_actor_diff_mean(run_dir.name)
            print(f"[info] {p} config missing use_actor_diff_mean; inferring "
                  f"{inferred} from run name '{run_dir.name}'", file=sys.stderr)
            cfg["use_actor_diff_mean"] = inferred

        found.append((step, p, cfg))
    return found


def _load_pi0_policy(pi0_ckpt: str):
    ckpt_dir = load_pi0_checkpoint(pi0_ckpt)
    cfg = openpi_config.get_config(CHECKPOINTS[pi0_ckpt]["config"])
    norm_stats = load_norm_stats_for_checkpoint(pi0_ckpt)
    print(f"Loading pi0 policy '{pi0_ckpt}' from {ckpt_dir}", flush=True)
    return policy_config.create_trained_policy(cfg, ckpt_dir, norm_stats=norm_stats)


def _residual_norm_diff(actions):
    """actions: [B, T, D] cumulative actions from sample_and_log_prob_diff.

    Computes the residual between each pair of consecutive AR steps
    (actions[:, t] - actions[:, t-1], for t=1..T-1 -- never the first step
    compared to the last), norms each residual vector over the action
    dimension, then averages over the chunk (the T-1 AR steps) and the
    batch. Returns a single scalar.
    """
    actions = np.asarray(actions)
    residual = actions[:, 1:, :] - actions[:, :-1, :]   # [B, T-1, D]
    norms = np.linalg.norm(residual, axis=-1)           # [B, T-1]
    return float(norms.mean())


def _residual_norm_diff_mean(mu, log_std):
    """mu, log_std: [B, T, D] accumulated TanhNormal params.

    ``ar_sample_diff_mean`` returns mu/log_std as ``[T, B, D]`` (scan axis
    first) while ``actions`` is transposed to ``[B, T, D]``.  Reorder here
    so consecutive-step residuals match the ``use_actor_diff`` path.

    Same as _residual_norm_diff, applied independently to the (mean,
    log_std) residual pair. Returns (residual_mean_norm, residual_log_std_norm)
    scalars, each averaged over the chunk and the batch.
    """
    mu = np.asarray(mu)
    log_std = np.asarray(log_std)
    if mu.ndim == 3:
        mu = np.transpose(mu, (1, 0, 2))
        log_std = np.transpose(log_std, (1, 0, 2))
    residual_mean = mu[:, 1:, :] - mu[:, :-1, :]                 # [B, T-1, D]
    residual_log_std = log_std[:, 1:, :] - log_std[:, :-1, :]    # [B, T-1, D]
    mean_norm = float(np.linalg.norm(residual_mean, axis=-1).mean())
    log_std_norm = float(np.linalg.norm(residual_log_std, axis=-1).mean())
    return mean_norm, log_std_norm


def eval_checkpoint(agent, agent_dp, env, variant, num_rollouts, max_timesteps, seed):
    """Run on-policy rollouts through one restored checkpoint.

    Returns a list of row-dicts with keys: metric, value, rollout, query_idx.
    Each row's value is already averaged over the chunk (AR steps) and the
    batch (see _residual_norm_diff / _residual_norm_diff_mean), so there is
    exactly one row per (rollout, query). Ready to be concatenated into the
    long-format output CSV.
    """
    if not (agent.use_actor_diff or agent.use_actor_diff_mean):
        raise ValueError(
            "Checkpoint uses neither use_actor_diff nor use_actor_diff_mean; "
            "nothing to evaluate."
        )

    rows = []
    rng = jax.random.PRNGKey(seed)
    query_frequency = 10 if agent.use_chunky_actor_critic > 0 else 5

    for r in range(num_rollouts):
        obs = env.reset()
        actions_pi0 = None
        for t in range(max_timesteps):
            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)
                curr_image = obs_to_img(obs, variant)
                qpos = obs_to_qpos(obs, variant)
                obs_dict = {
                    "pixels": curr_image[np.newaxis, ..., np.newaxis],
                    "state": qpos[np.newaxis, ..., np.newaxis],
                }
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)

                dist, _, _ = agent._actor.apply_fn(
                    {"params": agent._actor.params}, obs_dict, training=False
                )

                if agent.use_actor_diff:
                    actions, _ = dist.sample_and_log_prob_diff(seed=key)
                    norm = _residual_norm_diff(actions)
                    rows.append({
                        "metric": "residual_norm", "value": norm,
                        "rollout": r, "query_idx": t,
                    })
                    action_chunk = actions
                else:  # agent.use_actor_diff_mean
                    actions, _, mu, log_std = dist.sample_and_log_prob_diff_mean(seed=key)
                    mean_norm, log_std_norm = _residual_norm_diff_mean(mu, log_std)
                    rows.append({
                        "metric": "residual_mean_norm", "value": mean_norm,
                        "rollout": r, "query_idx": t,
                    })
                    rows.append({
                        "metric": "residual_log_std_norm", "value": log_std_norm,
                        "rollout": r, "query_idx": t,
                    })
                    action_chunk = actions

                noise_pi0 = _prepare_pi0_noise(np.asarray(action_chunk), agent, agent_dp.action_horizon)
                actions_pi0 = agent_dp.infer(obs_pi_zero, noise=noise_pi0)["actions"]

            action_t = actions_pi0[t % query_frequency]
            obs, reward, done, info = env.step(action_t)
            if done:
                break
        print(f"  rollout {r + 1}/{num_rollouts} done", flush=True)
    return rows


def eval_residual_norms(
    run_dir,
    task_id,
    libero_suite="libero_90",
    pi0_checkpoint="pi05_libero",
    num_rollouts=10,
    max_timesteps=100,
    seed=0,
    steps=None,
    output=None,
):
    run_dir = pathlib.Path(run_dir)
    run_name = run_dir.name
    if output is None:
        output = pathlib.Path("plots/data/residual_norms") / f"{run_name}.csv"
    else:
        output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(run_dir, steps=steps)
    if not checkpoints:
        raise ValueError(f"No restorable checkpoints found under {run_dir}")
    print(f"Found {len(checkpoints)} checkpoints in {run_dir}: "
          f"{[s for s, _, _ in checkpoints]}", flush=True)

    print(f"Loading pi0 policy '{pi0_checkpoint}'...", flush=True)
    agent_dp = _load_pi0_policy(pi0_checkpoint)

    task_suite = benchmark.get_benchmark_dict()[libero_suite]()
    task = task_suite.get_task(task_id)
    variant = AttrDict({
        "env": "libero",
        "resize_image": 64,
        "task_description": task.language,
    })
    env = create_libero_env(task, 256, seed)

    all_rows = []
    for step, ckpt_dir, config in checkpoints:
        print(f"Evaluating checkpoint{step} ({ckpt_dir})...", flush=True)
        agent = PixelSACLearner.restore_from_checkpoint_dir(str(ckpt_dir), config=config)
        rows = eval_checkpoint(agent, agent_dp, env, variant, num_rollouts, max_timesteps, seed)
        for row in rows:
            row["step"] = step
        all_rows.extend(rows)
        print(f"  collected {len(rows)} residual samples at step {step}", flush=True)

    df = pd.DataFrame(all_rows, columns=["step", "metric", "value", "rollout", "query_idx"])
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}", flush=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer residual-prediction norms from checkpoints via on-policy rollouts."
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the training run directory containing checkpoint<N> subdirs.")
    parser.add_argument("--task_id", type=int, required=True,
                        help="LIBERO task index used to build the eval environment.")
    parser.add_argument("--libero_suite", type=str, default="libero_90")
    parser.add_argument("--pi0_checkpoint", type=str, default="pi05_libero",
                        help="Key into examples.train_sim.CHECKPOINTS for the base VLA policy.")
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--max_timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Optional subset of checkpoint steps to evaluate (default: all found).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: plots/data/residual_norms/<run_name>.csv).")
    args = parser.parse_args()

    eval_residual_norms(
        run_dir=args.run_dir,
        task_id=args.task_id,
        libero_suite=args.libero_suite,
        pi0_checkpoint=args.pi0_checkpoint,
        num_rollouts=args.num_rollouts,
        max_timesteps=args.max_timesteps,
        seed=args.seed,
        steps=args.steps,
        output=args.output,
    )
