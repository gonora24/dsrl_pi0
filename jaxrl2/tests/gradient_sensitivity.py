import argparse
import pathlib
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from openpi.models import model as _model
from openpi.policies import policy_config

from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos
from examples.train_sim import CHECKPOINTS, load_pi0_checkpoint, load_norm_stats_for_checkpoint
from openpi.training import config as _openpi_config

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import AttrDict
from libero.libero import benchmark

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def create_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def _prepare_pi0_noise(actions_noise, pi0_action_horizon, action_dim=None):
    """Reshape SAC noise and pad to the VLA inference horizon (and dim) if needed.

    Args:
        actions_noise      : (C, D) or (1, C, D)-like array
        pi0_action_horizon : target T (pad by repeating last row)
        action_dim         : if set, zero-pad / truncate the last axis to this D
    """
    noise = actions_noise[None] if actions_noise.ndim == 2 else actions_noise
    if noise.shape[1] < pi0_action_horizon:
        noise_repeat = np.repeat(
            noise[:, -1:, :], pi0_action_horizon - noise.shape[1], axis=1
        )
        noise = np.concatenate([noise, noise_repeat], axis=1)
    elif noise.shape[1] > pi0_action_horizon:
        noise = noise[:, :pi0_action_horizon, :]

    if action_dim is not None:
        D_cur = noise.shape[2]
        if D_cur < action_dim:
            pad = np.zeros(
                (noise.shape[0], noise.shape[1], action_dim - D_cur),
                dtype=np.asarray(noise).dtype,
            )
            noise = np.concatenate([noise, pad], axis=-1)
        elif D_cur > action_dim:
            noise = noise[:, :, :action_dim]
    return noise


def _preprocess_obs(agent_dp, obs_pi_zero):
    """Preprocess a raw obs_pi_zero dict into a model Observation.

    Runs the policy's input transform and wraps into _model.Observation so the
    result can be passed directly to agent_dp._sample_actions.  This is done
    once per state, entirely outside the jax.jacfwd-traced path.
    """
    inputs = agent_dp._input_transform(jax.tree.map(lambda x: x, obs_pi_zero))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[jnp.newaxis, ...], inputs)
    return _model.Observation.from_dict(inputs)


def gradient_sensitivity_matrix(agent_dp, obs_proc, noise_pi0):
    """Exact (T_action, T_noise) sensitivity matrix via jax.jacfwd.

    Entry [i, j] = ||∂a_i / ∂z_j||_F  (Frobenius norm over action dim A and
    latent dim D of the (A, D) Jacobian block).
    """
    def fn(z):
        # z: (T, D) — add batch dim for _sample_actions, remove it from output
        return agent_dp._sample_actions(obs_proc, noise=z[None])[0]  # (T_action, A)

    J = jax.jacfwd(fn)(noise_pi0)   # (T_action, A, T_noise, D)
    return jnp.sqrt(jnp.sum(J ** 2, axis=(1, 3)))  # (T_action, T_noise)


def gradient_sensitivity_matrix_over_states(agent_dp, obs_procs, noises_pi0):
    """Compute the sensitivity matrix averaged over N (state, noise) pairs."""
    mats = [
        gradient_sensitivity_matrix(agent_dp, obs_proc, noise_pi0)
        for obs_proc, noise_pi0 in zip(obs_procs, noises_pi0)
    ]
    return jnp.mean(jnp.stack(mats, axis=0), axis=0)  # (T_action, T_noise)


def plot_sensitivity_matrix(
    mat,
    title: str = "Sensitivity matrix  ||∂aᵢ / ∂zⱼ||",
    action_label: str = "Action timestep  i",
    latent_label: str = "Latent timestep  j  (perturbed)",
    ax=None,
    cmap: str = "viridis",
):
    """Heat-map of a (T_action, T_noise) sensitivity matrix.

    Args:
        mat          : (T_action, T_noise) numpy / jnp array
        title        : plot title
        action_label : y-axis label
        latent_label : x-axis label
        ax           : existing Axes to draw into (creates figure if None)
        cmap         : matplotlib colormap

    Returns:
        fig, ax
    """
    mat = jnp.array(mat)
    C_a, C_l = mat.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, C_l * 0.6 + 1),
                                        max(3, C_a * 0.6 + 1)))
    else:
        fig = ax.get_figure()

    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper",
                   interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("sensitivity", fontsize=9)

    ax.set_xlabel(latent_label)
    ax.set_ylabel(action_label)
    ax.set_title(title)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if C_a * C_l <= 100:
        vmax = float(mat.max())
        for i in range(C_a):
            for j in range(C_l):
                v = float(mat[i, j])
                color = "white" if v < 0.6 * vmax else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.tight_layout()
    return fig, ax


def sensitivity_matrix_test(noise_actor_dir, libero_suite, task_id, N=100, checkpoint="pi05_libero",
                             seed=0, filename="sensitivity_matrix_gradient"):
    """Compute OpenPI action sensitivity to DSRL latent noise via jax.jacfwd,
    averaged over N sampled environment states.

    For each sampled state:
      1. Run the restored DSRL actor on the observation to obtain a (C, D)
         noise sample (the operating point for differentiation).
      2. Preprocess the observation once (input transforms + Observation build).
      3. Prepare the pi0-shaped noise once (numpy, outside the traced path).
      4. Call jax.jacfwd through _sample_actions to get the exact Jacobian and
         compute the (T, C) sensitivity matrix.
      5. Average the resulting matrices over all N states.
    """

    def load_noise_actor(ckpt_dir):
        return PixelSACLearner.restore_from_checkpoint_dir(ckpt_dir)

    def policy_load(pi0_ckpt):
        ckpt_dir = load_pi0_checkpoint(pi0_ckpt)
        cfg = _openpi_config.get_config(CHECKPOINTS[pi0_ckpt]["config"])
        norm_stats = load_norm_stats_for_checkpoint(pi0_ckpt)
        print(f"Loading openpi policy from {ckpt_dir}", flush=True)
        return policy_config.create_trained_policy(cfg, ckpt_dir, norm_stats=norm_stats)

    print("Loading policy...", flush=True)
    agent_dp = policy_load(checkpoint)
    action_horizon = agent_dp.action_horizon

    if noise_actor_dir is not None:
        print("Restoring DSRL noise actor...", flush=True)
        agent = load_noise_actor(noise_actor_dir)
        C, D = agent.action_chunk_shape
    else:
        C, D = (10, 32)
    print(f"C, D: {C, D}")
    print(f"action_horizon: {action_horizon}", flush=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_suite]()

    if task_id is None:
        state_pool = []
        for tid in range(task_suite.n_tasks):
            suite_task = task_suite.get_task(tid)
            suite_init_states = task_suite.get_task_init_states(tid)
            for init_state in suite_init_states:
                state_pool.append((suite_task, init_state))
        if len(state_pool) == 0:
            raise ValueError(f"No init states found for suite {libero_suite}")
        print(f"Sampling from full suite: {task_suite.n_tasks} tasks, {len(state_pool)} init states")
        task_for_env = state_pool[0][0]
    else:
        suite_task = task_suite.get_task(task_id)
        suite_init_states = task_suite.get_task_init_states(task_id)
        if len(suite_init_states) == 0:
            raise ValueError(f"No init states found for suite {libero_suite}, task_id {task_id}")
        state_pool = [(suite_task, init_state) for init_state in suite_init_states]
        print(f"Sampling from task {task_id}: {len(state_pool)} init states")
        task_for_env = suite_task

    variant = AttrDict({
        'env': 'libero',
        'resize_image': 64,
        'task_description': task_for_env.language,
    })

    env = create_libero_env(task_for_env, 256, seed)

    def _collect_obs(sample_idx):
        """Reset env to one sampled state and return (obs_dict, obs_pi_zero)."""
        task_i, init_state_i = state_pool[sample_idx % len(state_pool)]
        variant.task_description = task_i.language
        env.reset()
        obs = env.set_init_state(init_state_i)
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)
        obs_dict = {
            'pixels': curr_image[np.newaxis, ..., np.newaxis],
            'state':  qpos[np.newaxis, ..., np.newaxis],
        }
        obs_pi_zero = obs_to_pi_zero_input(obs, variant)
        return obs_dict, obs_pi_zero

    def _get_noise_cd(obs_dict, i):
        """Sample or generate (C, D) noise for a given observation."""
        if noise_actor_dir is not None:
            noise_cd = agent.sample_actions(obs_dict, marginalize_logprobs=False, use_actor_diff=False)
            if agent.action_chunk_shape[0] == 1:
                noise_repeat = np.repeat(
                    noise_cd[-1:, :], action_horizon - noise_cd.shape[0], axis=0
                )
                noise_cd = np.concatenate([noise_cd, noise_repeat], axis=0)
            else:
                noise_cd = noise_cd[0]
        else:
            noise_cd = np.asarray(
                jax.random.normal(jax.random.PRNGKey(seed + i), (C, D)),
                dtype=np.float32,
            )
        return noise_cd  # (T, D)

    # Single-state sanity check
    obs_dict_0, obs_pi_zero_0 = _collect_obs(0)
    noise_cd_0 = _get_noise_cd(obs_dict_0, 0)
    print(f"noise_cd_0 shape: {noise_cd_0.shape}")

    obs_proc_0 = _preprocess_obs(agent_dp, obs_pi_zero_0)
    noise_pi0_0 = _prepare_pi0_noise(noise_cd_0, action_horizon)[0]

    print("Computing single-state gradient sensitivity matrix...", flush=True)
    mat_0 = gradient_sensitivity_matrix(agent_dp, obs_proc_0, noise_pi0_0)
    print(f"Single-state sensitivity matrix shape: {mat_0.shape}")
    print(mat_0)

    # Average over N sampled states
    print(f"\nComputing gradient sensitivity matrix over {N} states...", flush=True)
    obs_procs_list = []
    noises_pi0_list = []

    for i in range(N):
        obs_dict_i, obs_pi_zero_i = _collect_obs(i)
        noise_cd_i = _get_noise_cd(obs_dict_i, i)
        obs_procs_list.append(_preprocess_obs(agent_dp, obs_pi_zero_i))
        noises_pi0_list.append(_prepare_pi0_noise(noise_cd_i, action_horizon)[0])
        if (i + 1) % 10 == 0:
            print(f"  collected {i + 1}/{N} states", flush=True)

    mean_mat = gradient_sensitivity_matrix_over_states(
        agent_dp, obs_procs_list, noises_pi0_list,
    )
    print("Mean sensitivity matrix:")
    print(mean_mat)

    fig, ax = plot_sensitivity_matrix(
        mean_mat,
        title=(
            f"Avg. gradient sensitivity for openpi {checkpoint} ({N} states, {libero_suite} all_tasks)"
            if task_id is None
            else f"Avg. gradient sensitivity for openpi {checkpoint} ({N} states, {libero_suite} task {task_id})"
        ),
        action_label=f"action timestep  i  (of {action_horizon})",
        latent_label=f"DSRL latent timestep  j  (of {C})",
    )
    out_path = f"plots/plots/sensitivity_matrices/{filename}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return mean_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_actor_dir", type=str, default=None,
                        help="Path to checkpointXXX directory from PixelSACLearner")
    parser.add_argument("--libero_suite", type=str, default="libero_90",
                        help="LIBERO benchmark suite (default: libero_90)")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Task index within the suite. If omitted, sample states across the whole suite.")
    parser.add_argument("--N", type=int, default=100,
                        help="Number of states to average over (default: 100)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filename", type=str, default="sensitivity_matrix_gradient",
                        help="Filename to save the sensitivity matrix (default: sensitivity_matrix_gradient)")
    parser.add_argument("--checkpoint", type=str, default="pi05_base",
                        help="Pi0 checkpoint to use (default: pi05_base)")
    args = parser.parse_args()
    sensitivity_matrix_test(
        noise_actor_dir=args.noise_actor_dir,
        libero_suite=args.libero_suite,
        task_id=args.task_id,
        N=args.N,
        seed=args.seed,
        filename=args.filename,
        checkpoint=args.checkpoint,
    )
