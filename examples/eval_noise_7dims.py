import argparse
import pathlib
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

CHUNK_DIM = 7  # dimensionality of a single action's noise distribution


def _load_policy(ckpt_key):
    ckpt_dir = load_pi0_checkpoint(ckpt_key)
    cfg = openpi_config.get_config(CHECKPOINTS[ckpt_key]["config"])
    norm_stats = load_norm_stats_for_checkpoint(ckpt_key)
    print(f"Loading openpi policy '{ckpt_key}' from {ckpt_dir}", flush=True)
    return policy_config.create_trained_policy(cfg, ckpt_dir, norm_stats=norm_stats)


def _kl_diag_gaussian(mean_p, log_std_p, mean_q, log_std_q):
    """KL(p || q) between independent diagonal Gaussians, summed over the last dim.

    All inputs are arrays with matching shape (..., D). Returns shape (...).
    """
    mean_p = np.asarray(mean_p)
    log_std_p = np.asarray(log_std_p)
    mean_q = np.asarray(mean_q)
    log_std_q = np.asarray(log_std_q)

    var_p = np.exp(2.0 * log_std_p)
    var_q = np.exp(2.0 * log_std_q)
    kl = (log_std_q - log_std_p) + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q) - 0.5
    return np.mean(kl, axis=-1)

def _jenson_shannon_divergence_fixed(mean_p, log_std_p, mean_q, log_std_q):
    """Corrected Jensen-Shannon Divergence upper bound for independent diagonal Gaussians.
    
    Correctly includes mean differences and variances, mean over the last dimension.
    """
    mean_p = np.asarray(mean_p)
    log_std_p = np.asarray(log_std_p)
    mean_q = np.asarray(mean_q)
    log_std_q = np.asarray(log_std_q)

    var_p = np.exp(2.0 * log_std_p)
    var_q = np.exp(2.0 * log_std_q)
    
    # 1. Calculate Closed-Form KL(P || Q) 
    kl_pq = 0.5 * np.mean(
        2.0 * (log_std_q - log_std_p) - 1.0 + (var_p + (mean_p - mean_q)**2) / var_q, 
        axis=-1
    )
    
    # 2. Calculate Closed-Form KL(Q || P)
    kl_qp = 0.5 * np.mean(
        2.0 * (log_std_p - log_std_q) - 1.0 + (var_q + (mean_q - mean_p)**2) / var_p, 
        axis=-1
    )
    
    # 3. Average the KL values and convert from nats to bits (log base 2)
    jsd_upper_bound = 0.5 * (kl_pq + kl_qp)
    return jsd_upper_bound / np.log(2.0)


def _to_chunks(mean, log_std, chunk_dim=CHUNK_DIM):
    """Reshape a flat (D,) distribution into (num_chunks, chunk_dim).

    D == chunk_dim      -> a single "chunk"      (num_chunks == 1)
    D == 5 * chunk_dim  -> five 7-dim chunks     (num_chunks == 5)
    """
    mean = np.asarray(mean).reshape(-1)
    log_std = np.asarray(log_std).reshape(-1)
    d = mean.shape[0]
    if d % chunk_dim != 0:
        raise ValueError(f"Distribution dim {d} is not a multiple of chunk_dim={chunk_dim}")
    num_chunks = d // chunk_dim
    return mean.reshape(num_chunks, chunk_dim), log_std.reshape(num_chunks, chunk_dim)


def _first_chunk(mean, log_std, chunk_dim=CHUNK_DIM):
    """The first 7-dim chunk. Used so a 7-dim actor and a 35-dim actor can be
    compared on equal footing (35-dim -> take chunk 0)."""
    m_chunks, ls_chunks = _to_chunks(mean, log_std, chunk_dim)
    return m_chunks[0], ls_chunks[0]


def eval_noise_7dims(checkpoint, noise_actor_dirs, task_id, num_rollouts, max_timesteps, filename, cross_actor=True):
    out_dir = pathlib.Path("plots/plots/noise_7dims/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    print("Loading policy...", flush=True)
    agent_dp = _load_policy(checkpoint)

    # Load noise actors
    agent_noise_list = []
    print(f"noise_actor_dirs: {noise_actor_dirs}", flush=True)
    print(f"type(noise_actor_dirs): {type(noise_actor_dirs)}", flush=True)
    print(f"first actor dir: {noise_actor_dirs[0]}", flush=True)
    for noise_actor_dir in noise_actor_dirs:
        print(f"Restoring DSRL noise actor from {noise_actor_dir}...", flush=True)
        if "residualmlp" in noise_actor_dir:
            extra_args = {
                "use_residual_mlp": True,
                "only_predict_dims_until": 7,
                "num_noise_vectors": 5,
                "noise_repeats_per_vector": 2,
            }
            print(f"Using extra args for chunkrewardcriticactor_mlp_7dims_5vecs_2reps_residualmlp: {extra_args}", flush=True)
        elif "chunkrewardcriticactor_mlp_7dims_5vecs_2reps" in noise_actor_dir:
            extra_args = {
                "only_predict_dims_until": 7,
                "num_noise_vectors": 5,
                "noise_repeats_per_vector": 2,
            }
            print(f"Using extra args for chunkrewardcriticactor_mlp_7dims_5vecs_2reps: {extra_args}", flush=True)
        elif "chunkrewardcriticactor_mlp_7dims" in noise_actor_dir:
            extra_args = {
                "only_predict_dims_until": 7,
            }
            print(f"Using extra args for chunkrewardcriticactor_mlp_7dims: {extra_args}", flush=True)
        else:
            extra_args = {}
        agent_noise = PixelSACLearner.restore_from_checkpoint_dir(noise_actor_dir, extra_args=extra_args)
        agent_noise_list.append(agent_noise)

    # Load environment
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

    # records[actor_idx][rollout_idx][t] = (mean, log_std) as flat np arrays
    records = [
        [dict() for _ in range(num_rollouts)]
        for _ in range(len(agent_noise_list))
    ]
    

    for i, agent_noise in enumerate(agent_noise_list):
        print("New noise actor", flush=True)
        print(agent_noise)
        agent_noise._rng, rng = jax.random.split(agent_noise._rng)
        query_frequency = 10 if agent_noise.use_chunky_actor_critic > 0 else 5
        print(f"query_frequency: {query_frequency}", flush=True)
        for r in range(num_rollouts):
            obs = env.reset()
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
                    distribution, means, log_stds = agent_noise._actor.apply_fn({'params': agent_noise._actor.params}, obs_dict, training=False)
                    noise_cd = agent_noise.sample_actions(
                        obs_dict,
                        marginalize_logprobs=agent_noise.marginalize_logprobs,
                        use_actor_diff=agent_noise.use_actor_diff,
                    )
                    records[i][r][t] = (
                        np.asarray(means).reshape(-1),
                        np.asarray(log_stds).reshape(-1),
                    )
                    if noise_cd.shape[0] == 1 and noise_cd.shape[1] == 7:
                        # Build full (1, H, 32) noise and embed the actor's 7-dim prediction
                        # into dims 0:only_predict_dims_until across all timesteps of the horizon.
                        noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.dsrl_action_dim))
                        noise_pi0 = noise.at[0, :, :noise_cd.shape[1]].set(noise_cd[0])
                    elif noise_cd.shape[1] == 5 and noise_cd.shape[2] == 7:
                        noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.dsrl_action_dim))
                        repeated_actions_noise = jax.numpy.repeat(noise_cd, repeats=2, axis=1)
                        noise_pi0 = noise.at[0, :, :repeated_actions_noise.shape[2]].set(repeated_actions_noise[0, :, :])
                    elif noise_cd.shape[1] == 10 and noise_cd.shape[2] == 7:
                        noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.dsrl_action_dim))
                        noise_pi0 = noise.at[0, :, :noise_cd.shape[2]].set(noise_cd[0])
                    else:
                        noise_pi0 = _prepare_pi0_noise(noise_cd, agent_noise, agent_dp.action_horizon)[0]
                    actions = agent_dp.infer(obs_pi_zero, noise=noise_pi0)["actions"]

                action_t = actions[t % query_frequency]
                obs, reward, done, info = env.step(action_t)
                if done:
                    break

    # ---- Analysis ----
    within_chunk_kl, within_chunk_jsd = _aggregate_within_chunk_kl(records, noise_actor_dirs)

    cross_actor_kl = None
    if cross_actor:
        cross_actor_kl = _aggregate_cross_actor_kl(records, noise_actor_dirs)
        _plot_cross_actor_kl(cross_actor_kl, out_dir, filename)

    _plot_within_chunk_kl(within_chunk_kl, out_dir, filename, task_id)
    # _plot_within_chunk_jsd(within_chunk_jsd, out_dir, filename, task_id)

    return cross_actor_kl, within_chunk_kl, within_chunk_jsd


def _aggregate_cross_actor_kl(records, noise_actor_dirs):
    """For every ordered pair of actors (i, j), compute
    KL(first_chunk_i || first_chunk_j) at every timestep both actors queried
    within the same rollout index, averaged over rollouts.

    Returns {(name_i, name_j): {t: mean_kl_over_rollouts}}
    """
    n_actors = len(records)
    num_rollouts = len(records[0])
    result = {}

    for i in range(n_actors):
        for j in range(n_actors):
            if i == j:
                continue
            per_t_vals = defaultdict(list)
            for r in range(num_rollouts):
                rec_i = records[i][r]
                rec_j = records[j][r]
                common_ts = sorted(set(rec_i.keys()) & set(rec_j.keys()))
                for t in common_ts:
                    mean_i, log_std_i = _first_chunk(*rec_i[t])
                    mean_j, log_std_j = _first_chunk(*rec_j[t])
                    kl = _kl_diag_gaussian(mean_i, log_std_i, mean_j, log_std_j)
                    per_t_vals[t].append(float(kl))
            result[(noise_actor_dirs[i], noise_actor_dirs[j])] = {
                t: float(np.mean(vals)) for t, vals in sorted(per_t_vals.items())
            }
    return result


def _aggregate_within_chunk_kl(records, noise_actor_dirs):
    """For actors whose distribution is 35-dim (5 chunks of 7): KL and JSD between
    consecutive 7-dim chunks within the same prediction (chunk0 vs chunk1,
    chunk1 vs chunk2, ...), averaged over all rollouts and queried timesteps.

    Actors that only predict a single 7-dim distribution are skipped (there
    is no internal chunk variation to measure).

    Returns:
        kl_dict  -- {actor_name: [kl_01, kl_12, kl_23, kl_34]}
        jsd_dict -- {actor_name: [jsd_01, jsd_12, jsd_23, jsd_34]}
    """
    kl_result = {}
    jsd_result = {}
    for i, name in enumerate(noise_actor_dirs):
        # Peek at dimensionality from any recorded step
        sample = None
        for r_dict in records[i]:
            if r_dict:
                sample = next(iter(r_dict.values()))
                break
        if sample is None:
            continue
        d = sample[0].shape[0]
        # if d == CHUNK_DIM:
        #     continue  # single 7-dim distribution, nothing to compare internally

        pair_vals = None
        jsd_vals = None
        for r_dict in records[i]:
            for mean, log_std in r_dict.values():
                m_chunks, ls_chunks = _to_chunks(mean, log_std)
                num_chunks = m_chunks.shape[0]
                if pair_vals is None:
                    pair_vals = [[] for _ in range(num_chunks - 1)]
                if jsd_vals is None:
                    jsd_vals = [[] for _ in range(num_chunks - 1)]
                for c in range(num_chunks - 1):
                    kl = _kl_diag_gaussian(
                        m_chunks[c], ls_chunks[c], m_chunks[c + 1], ls_chunks[c + 1]
                    )
                    print(f"m_chunks[c]: {m_chunks[c]}", flush=True)
                    print(f"ls_chunks[c]: {ls_chunks[c]}", flush=True)
                    print(f"m_chunks[c + 1]: {m_chunks[c + 1]}", flush=True)
                    print(f"ls_chunks[c + 1]: {ls_chunks[c + 1]}", flush=True)
                    jsd = _jenson_shannon_divergence_fixed(
                        m_chunks[c], ls_chunks[c], m_chunks[c + 1], ls_chunks[c + 1]
                    )
                    pair_vals[c].append(float(kl))
                    jsd_vals[c].append(float(jsd))

        if pair_vals is not None:
            kl_result[name] = [float(np.mean(v)) if v else float("nan") for v in pair_vals]
        if jsd_vals is not None:
            jsd_result[name] = [float(np.mean(v)) if v else float("nan") for v in jsd_vals]
    return kl_result, jsd_result


def _plot_cross_actor_kl(cross_actor_kl, out_dir, filename):
    if not cross_actor_kl:
        return
    plt.figure(figsize=(8, 5))
    for (name_i, name_j), per_t in cross_actor_kl.items():
        if not per_t:
            continue
        parent_name1 = pathlib.Path(name_i).parent.name
        parent_name2 = pathlib.Path(name_j).parent.name
        name1 = "_".join(parent_name1.split("_")[-2:])
        name2 = "_".join(parent_name2.split("_")[-2:])
        ts = sorted(per_t.keys())
        vals = [per_t[t] for t in ts]
        label = f"{name1} vs {name2}"
        plt.plot(ts, vals, marker="o", label=label)
    plt.xlabel("Timestep")
    plt.ylabel("KL divergence (first 7-dim chunk)")
    plt.title("Cross-actor KL divergence over time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / f"{filename}_cross_actor_kl.svg"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved cross-actor KL plot to {out_path}", flush=True)


def _plot_within_chunk_kl(within_chunk_kl, out_dir, filename, task_id):
    if not within_chunk_kl:
        return
    plt.figure(figsize=(8, 5))
    width = 0.8 / max(len(within_chunk_kl), 1)
    for idx, (name, vals) in enumerate(within_chunk_kl.items()):
        parent_name = pathlib.Path(name).parent.name
        label = "_".join(parent_name.split("_")[-2:])
        xs = np.arange(len(vals)) + idx * width
        plt.bar(xs, vals, width=width, label=label)
    plt.xlabel("Consecutive chunk pair (chunk c vs chunk c+1)")
    plt.ylabel("KL divergence")
    plt.title(f"Within-chunk consecutive KL divergence for Task {task_id}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / f"{filename}_within_chunk_kl.svg"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved within-chunk KL plot to {out_path}", flush=True)


def _plot_within_chunk_jsd(within_chunk_jsd, out_dir, filename, task_id):
    if not within_chunk_jsd:
        return
    plt.figure(figsize=(8, 5))
    width = 0.8 / max(len(within_chunk_jsd), 1)
    for idx, (name, vals) in enumerate(within_chunk_jsd.items()):
        parent_name = pathlib.Path(name).parent.name
        label = "_".join(parent_name.split("_")[-2:])
        xs = np.arange(len(vals)) + idx * width
        plt.bar(xs, vals, width=width, label=label)
    plt.xlabel("Consecutive chunk pair (chunk c vs chunk c+1)")
    plt.ylabel("JSD upper bound (bits, sym-KL/2)")
    plt.title(f"Within-chunk consecutive JSD for Task {task_id}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / f"{filename}_within_chunk_jsd.svg"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved within-chunk JSD plot to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pi05_base")
    parser.add_argument("--noise_actor_dirs", type=str, nargs="+", default=["logs/logs/noise_actors/noise_actor_1","logs/logs/noise_actors/noise_actor_2"])
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--max_timesteps", type=int, default=100)
    parser.add_argument("--filename", type=str, default="noise_eval_7dims")
    parser.add_argument("--cross_actor", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute and plot cross-actor KL divergence (use --no_cross_actor to disable).")
    args = parser.parse_args()
    noise_actor_dirs = args.noise_actor_dirs[0].split(",") if args.noise_actor_dirs else []
    eval_noise_7dims(
        checkpoint=args.checkpoint,
        noise_actor_dirs=noise_actor_dirs,
        task_id=args.task_id,
        num_rollouts=args.num_rollouts,
        max_timesteps=args.max_timesteps,
        filename=args.filename,
        cross_actor=args.cross_actor,
    )