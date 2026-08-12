"""Plot within-chunk KL divergence for multiple noise-actor checkpoints.

Runs rollouts for each supplied checkpoint, computes the KL divergence between
consecutive 7-dim chunk distributions predicted by the noise actor, and outputs
a single grouped bar chart.  A random-Gaussian baseline (expected KL between
two independent draws from N(0,I)) is included in every group for reference.

Usage
-----
python examples/plot_kl_within_chunk_checkpoints.py \
    --checkpoint pi05_libero \
    --noise_actor_dirs /path/to/ckpt1 /path/to/ckpt2 \
    --labels "baseline" "5vecs_residual" \
    --filename kl_multi_ckpt_task38
"""

import argparse
import pathlib
import re
from collections import defaultdict

import jax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LIBERO.libero.libero import benchmark
from examples.train_sim import load_pi0_checkpoint, load_norm_stats_for_checkpoint
from examples.train_utils_sim import obs_to_img, obs_to_qpos, obs_to_pi_zero_input, _prepare_pi0_noise
from jaxrl2.tests.finite_differences import create_libero_env
from jaxrl2.utils.general_utils import AttrDict
from openpi.policies import policy_config
from openpi.training import config as openpi_config
from examples.train_sim import CHECKPOINTS
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner

CHUNK_DIM = 7


plt.rcParams.update({
    "mathtext.fontset": "cm",
})


# ---------------------------------------------------------------------------
# Math helpers (shared with eval_noise_7dims.py)
# ---------------------------------------------------------------------------

def _kl_diag_gaussian(mean_p, log_std_p, mean_q, log_std_q):
    """KL(p || q) for diagonal Gaussians, mean-reduced over the last dim."""
    mean_p = np.asarray(mean_p)
    log_std_p = np.asarray(log_std_p)
    mean_q = np.asarray(mean_q)
    log_std_q = np.asarray(log_std_q)
    var_p = np.exp(2.0 * log_std_p)
    var_q = np.exp(2.0 * log_std_q)
    kl = (log_std_q - log_std_p) + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q) - 0.5
    return np.mean(kl, axis=-1)


def _to_chunks(mean, log_std, chunk_dim=CHUNK_DIM):
    mean = np.asarray(mean).reshape(-1)
    log_std = np.asarray(log_std).reshape(-1)
    d = mean.shape[0]
    if d % chunk_dim != 0:
        raise ValueError(f"Distribution dim {d} is not a multiple of chunk_dim={chunk_dim}")
    num_chunks = d // chunk_dim
    return mean.reshape(num_chunks, chunk_dim), log_std.reshape(num_chunks, chunk_dim)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extra_args_for_dir(noise_actor_dir: str) -> dict:
    if "residualmlp" in noise_actor_dir:
        return {
            "use_residual_mlp": True,
            "only_predict_dims_until": 7,
            "num_noise_vectors": 5,
            "noise_repeats_per_vector": 2,
        }
    if "chunkrewardcriticactor_mlp_7dims_5vecs_2reps" in noise_actor_dir:
        return {
            "only_predict_dims_until": 7,
            "num_noise_vectors": 5,
            "noise_repeats_per_vector": 2,
        }
    if "chunkrewardcriticactor_mlp_7dims" in noise_actor_dir:
        return {"only_predict_dims_until": 7}
    return {}


def _extract_task_id(paths: list[str]) -> int | None:
    """Return the first task ID found via regex ``task(\\d+)`` in any of paths."""
    for p in paths:
        m = re.search(r"task(\d+)", p)
        if m:
            return int(m.group(1))
    return None


def _auto_label(noise_actor_dir: str) -> str:
    """Derive a short human-readable label from a checkpoint path."""
    path = pathlib.Path(noise_actor_dir)
    step_raw = path.name  # e.g. checkpoint2000000
    step_num = re.search(r"(\d+)$", step_raw)
    step_str = f"{int(step_num.group(1)) // 1000}k" if step_num else step_raw

    run_name = path.parent.name  # e.g. dsrl_pi05_..._chunkrewardcriticactor_mlp_7dims_5vecs_2reps_residualmlp
    # Try to extract the actor type after the seed suffix "--s-N_"
    m = re.search(r"--s-\d+_(.*)", run_name)
    if m:
        actor_type = m.group(1)
    else:
        # Fall back to last two underscore-joined tokens of the run name
        actor_type = "_".join(run_name.split("_")[-3:])

    return f"{actor_type}\n{step_str}"


def _build_labels(noise_actor_dirs: list[str], user_labels: list[str] | None) -> list[str]:
    labels = []
    for i, d in enumerate(noise_actor_dirs):
        if user_labels and i < len(user_labels) and user_labels[i]:
            labels.append(user_labels[i])
        else:
            labels.append(_auto_label(d))
    return labels


def _load_policy(ckpt_key: str):
    ckpt_dir = load_pi0_checkpoint(ckpt_key)
    cfg = openpi_config.get_config(CHECKPOINTS[ckpt_key]["config"])
    norm_stats = load_norm_stats_for_checkpoint(ckpt_key)
    print(f"Loading openpi policy '{ckpt_key}' from {ckpt_dir}", flush=True)
    return policy_config.create_trained_policy(cfg, ckpt_dir, norm_stats=norm_stats)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _collect_records(agent_dp, noise_actor_dirs: list[str], task_id: int,
                     num_rollouts: int, max_timesteps: int):
    """Run rollouts for every noise actor and return per-step (mean, log_std) records.

    Returns
    -------
    records : list[list[dict]]
        records[actor_idx][rollout_idx][t] = (mean_flat, log_std_flat)
    """
    task_suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = task_suite.get_task(task_id)
    variant = AttrDict({
        "env": "libero",
        "resize_image": 64,
        "task_description": task.language,
        "dsrl_action_dim": 32,
        "pi0_action_horizon": 10,
    })
    env = create_libero_env(task, 256, 0)

    records = [
        [dict() for _ in range(num_rollouts)]
        for _ in range(len(noise_actor_dirs))
    ]

    for i, noise_actor_dir in enumerate(noise_actor_dirs):
        print(f"\n[{i+1}/{len(noise_actor_dirs)}] Restoring noise actor from {noise_actor_dir}", flush=True)
        extra_args = _extra_args_for_dir(noise_actor_dir)
        if extra_args:
            print(f"  extra_args: {extra_args}", flush=True)
        agent_noise = PixelSACLearner.restore_from_checkpoint_dir(noise_actor_dir, extra_args=extra_args)

        agent_noise._rng, rng = jax.random.split(agent_noise._rng)
        query_frequency = 10 if agent_noise.use_chunky_actor_critic > 0 else 5
        print(f"  query_frequency: {query_frequency}", flush=True)

        for r in range(num_rollouts):
            print(f"  rollout {r+1}/{num_rollouts}", flush=True)
            obs = env.reset()
            actions = None
            for t in range(max_timesteps):
                curr_image = obs_to_img(obs, variant)
                qpos = obs_to_qpos(obs, variant)
                obs_dict = {
                    "pixels": curr_image[np.newaxis, ..., np.newaxis],
                    "state": qpos[np.newaxis, ..., np.newaxis],
                }
                if t % query_frequency == 0:
                    rng, key = jax.random.split(rng)
                    obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                    _dist, means, log_stds = agent_noise._actor.apply_fn(
                        {"params": agent_noise._actor.params}, obs_dict, training=False
                    )
                    noise_cd = agent_noise.sample_actions(
                        obs_dict,
                        marginalize_logprobs=agent_noise.marginalize_logprobs,
                        use_actor_diff=agent_noise.use_actor_diff,
                    )
                    records[i][r][t] = (
                        np.asarray(means).reshape(-1),
                        np.asarray(log_stds).reshape(-1),
                    )
                    # Build full noise tensor for the base policy
                    if noise_cd.shape[0] == 1 and noise_cd.shape[1] == 7:
                        noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.dsrl_action_dim))
                        noise_pi0 = noise.at[0, :, :noise_cd.shape[1]].set(noise_cd[0])
                    elif noise_cd.shape[1] == 5 and noise_cd.shape[2] == 7:
                        noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.dsrl_action_dim))
                        repeated = jax.numpy.repeat(noise_cd, repeats=2, axis=1)
                        noise_pi0 = noise.at[0, :, :repeated.shape[2]].set(repeated[0])
                    elif noise_cd.shape[1] == 10 and noise_cd.shape[2] == 7:
                        noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.dsrl_action_dim))
                        noise_pi0 = noise.at[0, :, :noise_cd.shape[2]].set(noise_cd[0])
                    else:
                        noise_pi0 = _prepare_pi0_noise(noise_cd, agent_noise, agent_dp.action_horizon)[0]
                    actions = agent_dp.infer(obs_pi_zero, noise=noise_pi0)["actions"]

                if actions is not None:
                    action_t = actions[t % query_frequency]
                    obs, _reward, done, _info = env.step(action_t)
                    if done:
                        break

    return records


# ---------------------------------------------------------------------------
# KL aggregation
# ---------------------------------------------------------------------------

def _aggregate_within_chunk_kl(records, noise_actor_dirs):
    """KL between consecutive chunks, with per-sample values for std computation.

    Returns
    -------
    kl_means : dict[str, list[float]]
        Mean KL per consecutive chunk pair, keyed by actor dir.
    kl_stds : dict[str, list[float]]
        Std of KL per consecutive chunk pair, keyed by actor dir.
    num_pairs : int
        Number of chunk pairs (e.g. 4 for a 5-chunk actor).
    """
    kl_means = {}
    kl_stds = {}
    num_pairs = 0

    for i, name in enumerate(noise_actor_dirs):
        sample = None
        for r_dict in records[i]:
            if r_dict:
                sample = next(iter(r_dict.values()))
                break
        if sample is None:
            print(f"Warning: no data recorded for {name}", flush=True)
            continue

        pair_vals: list[list[float]] | None = None
        for r_dict in records[i]:
            for mean, log_std in r_dict.values():
                m_chunks, ls_chunks = _to_chunks(mean, log_std)
                n = m_chunks.shape[0]
                if pair_vals is None:
                    pair_vals = [[] for _ in range(n - 1)]
                for c in range(n - 1):
                    kl = float(_kl_diag_gaussian(
                        m_chunks[c], ls_chunks[c], m_chunks[c + 1], ls_chunks[c + 1]
                    ))
                    pair_vals[c].append(kl)

        if pair_vals is not None:
            kl_means[name] = [float(np.mean(v)) if v else float("nan") for v in pair_vals]
            kl_stds[name] = [float(np.std(v)) if v else float("nan") for v in pair_vals]
            num_pairs = max(num_pairs, len(pair_vals))

    return kl_means, kl_stds, num_pairs


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def _random_kl_baseline(num_pairs: int, chunk_dim: int = CHUNK_DIM, n_samples: int = 2000,
                         seed: int = 42) -> tuple[list[float], list[float]]:
    """Expected KL between two independent random diagonal Gaussians.

    Means drawn from N(0,1), log-stds drawn from N(0,1).  The result is the
    same for every chunk-pair slot, but we return one value per pair so the
    bar chart stays symmetric.

    Returns
    -------
    means : list[float]  length num_pairs
    stds  : list[float]  length num_pairs
    """
    rng = np.random.RandomState(seed)
    kls = []
    for _ in range(n_samples):
        m1 = rng.randn(chunk_dim)
        ls1 = rng.randn(chunk_dim)
        m2 = rng.randn(chunk_dim)
        ls2 = rng.randn(chunk_dim)
        kls.append(float(_kl_diag_gaussian(m1, ls1, m2, ls2)))
    mean_val = float(np.mean(kls))
    std_val = float(np.std(kls))
    return [mean_val] * num_pairs, [std_val] * num_pairs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_bar_chart(kl_means: dict, kl_stds: dict, labels: list[str],
                    rand_means: list[float], rand_stds: list[float],
                    num_pairs: int, out_path: pathlib.Path, errorbar: bool = True):
    n_runs = len(labels)
    n_bars = n_runs + 1  # +1 for random baseline
    group_width = 0.8
    bar_width = group_width / n_bars
    xs = np.arange(num_pairs)

    fig, ax = plt.subplots(figsize=(max(6, num_pairs * 2.2), 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, label) in enumerate(zip(kl_means.keys(), labels)):
        means = kl_means[name]
        stds = kl_stds[name]
        offsets = xs - group_width / 2 + bar_width * (idx + 0.5)
        if errorbar:
            ax.bar(offsets, means, width=bar_width, yerr=stds,
                   color=colors[idx % len(colors)], label=label,
                   capsize=4, error_kw={"elinewidth": 1.2})
        else:
            ax.bar(offsets, means, width=bar_width,
                   color=colors[idx % len(colors)], label=label,
                   capsize=4)

    # Random baseline — hatched, grey
    rand_offsets = xs - group_width / 2 + bar_width * (n_runs + 0.5)
    if errorbar:
        ax.bar(rand_offsets, rand_means, width=bar_width, yerr=rand_stds,
               color="lightgrey", edgecolor="grey", hatch="//", label="random",
               capsize=4, error_kw={"elinewidth": 1.2})
    else:
        ax.bar(rand_offsets, rand_means, width=bar_width,
               color="lightgrey", edgecolor="grey", hatch="//", label="random",
               capsize=4)

    pair_labels = [f"{c}→{c+1}" for c in range(num_pairs)]
    ax.set_xticks(xs)
    ax.set_xticklabels(pair_labels)
    ax.set_xlabel("Consecutive chunk pair")
    ax.set_ylabel("KL divergence (nats)")
    ax.set_title("Within-chunk KL divergence for \$\pi_{0.5}\$ LIBERO-90")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_kl_within_chunk_checkpoints(
    checkpoint: str,
    noise_actor_dirs: list[str],
    task_id: int | None,
    num_rollouts: int,
    max_timesteps: int,
    filename: str,
    out_dir: str,
    labels: list[str] | None,
    random_baseline_samples: int,
    errorbar: bool = True,
):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resolve task ID
    if task_id is None:
        task_id = _extract_task_id(noise_actor_dirs)
        if task_id is None:
            raise ValueError(
                "Could not extract task_id from any checkpoint path. "
                "Please pass --task_id explicitly."
            )
        print(f"Auto-detected task_id={task_id} from checkpoint path.", flush=True)

    run_labels = _build_labels(noise_actor_dirs, labels)
    print(f"Run labels: {run_labels}", flush=True)

    # Load base policy
    print("Loading base policy...", flush=True)
    agent_dp = _load_policy(checkpoint)

    # Collect rollout records
    records = _collect_records(agent_dp, noise_actor_dirs, task_id, num_rollouts, max_timesteps)

    # Aggregate within-chunk KL
    kl_means, kl_stds, num_pairs = _aggregate_within_chunk_kl(records, noise_actor_dirs)

    if num_pairs == 0:
        print("No within-chunk pairs found (all actors may be single-chunk). Nothing to plot.", flush=True)
        return

    # Random baseline
    rand_means, rand_stds = _random_kl_baseline(
        num_pairs=num_pairs,
        chunk_dim=CHUNK_DIM,
        n_samples=random_baseline_samples,
    )

    # Plot
    svg_path = out_path / f"{filename}_within_chunk_kl_multi_ckpt.svg"
    _plot_bar_chart(
        kl_means=kl_means,
        kl_stds=kl_stds,
        labels=run_labels,
        rand_means=rand_means,
        rand_stds=rand_stds,
        num_pairs=num_pairs,
        out_path=svg_path,
        errorbar=errorbar,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grouped bar chart of within-chunk KL for multiple noise-actor checkpoints."
    )
    parser.add_argument("--checkpoint", type=str, default="pi05_libero",
                        help="Key in CHECKPOINTS dict for the base pi0 policy.")
    parser.add_argument("--noise_actor_dirs", type=str, nargs="+", required=True,
                        help="Checkpoint dirs for each noise actor to compare. "
                             "Can also be passed as a single comma-separated string.")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Libero-90 task index. If omitted, extracted from the "
                             "checkpoint path via regex task(\\d+).")
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--max_timesteps", type=int, default=400)
    parser.add_argument("--filename", type=str, default="kl_multi_ckpt",
                        help="Output SVG filename prefix.")
    parser.add_argument("--out_dir", type=str, default="plots/plots/noise_7dims/",
                        help="Directory to save the output SVG.")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Human-readable labels for each checkpoint dir (positional). "
                             "Falls back to auto-extraction for any missing entry.")
    parser.add_argument("--random_baseline_samples", type=int, default=2000,
                        help="Number of random Gaussian pairs used to estimate the "
                             "random KL baseline.")
    parser.add_argument("--errorbar", type=int, default=1,
                        help="Whether to plot error bars. 0 for no error bars, 1 for error bars.")
    args = parser.parse_args()

    # Support comma-separated dirs passed as a single string
    noise_actor_dirs = args.noise_actor_dirs[0].split(",") if len(args.noise_actor_dirs) == 1 else args.noise_actor_dirs

    plot_kl_within_chunk_checkpoints(
        checkpoint=args.checkpoint,
        noise_actor_dirs=noise_actor_dirs,
        task_id=args.task_id,
        num_rollouts=args.num_rollouts,
        max_timesteps=args.max_timesteps,
        filename=args.filename,
        out_dir=args.out_dir,
        labels=args.labels,
        random_baseline_samples=args.random_baseline_samples,
        errorbar=args.errorbar,
    )
