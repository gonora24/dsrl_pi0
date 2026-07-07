import argparse
import pathlib
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from openpi.policies import policy_config

from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def create_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env

def _prepare_pi0_noise(actions_noise, agent, pi0_action_horizon):
    """Reshape SAC noise and pad to pi0's inference horizon if needed."""
    # actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
    noise = actions_noise[None] if actions_noise.shape[0] > 1 else actions_noise
    if noise.shape[1] < pi0_action_horizon:
        noise_repeat = np.repeat(
            noise[:, -1:, :], pi0_action_horizon - noise.shape[1], axis=1
        )
        noise = np.concatenate([noise, noise_repeat], axis=1)
    return noise

def finite_difference_column(
    policy_apply,
    state,
    noise,
    latent_idx,
    eps: float = 1e-2,
    key=None,
):
    """
    Symmetric finite-difference estimate of how much each action timestep
    changes when latent timestep `latent_idx` is perturbed by a random unit
    direction scaled to magnitude `eps`.

    Args:
        policy_apply : callable (params, state, noise[C,D]) -> actions[C,A]
        state        : observation / context
        noise        : (C, D)  latent noise array
        latent_idx   : int in [0, C)
        eps          : perturbation magnitude
        key          : JAX PRNGKey; created from PRNGKey(0) if None

    Returns:
        change : (C,) L2 sensitivity per action timestep
    """
    C, D = noise.shape

    if key is None:
        key = jax.random.PRNGKey(0)

    # Unit-norm random direction scaled to eps  →  correct regardless of D
    raw = jax.random.normal(key, (D,))
    delta = eps * raw #/ jnp.linalg.norm(raw)

    perturb = jnp.zeros_like(noise).at[latent_idx].set(delta)

    a_plus  = policy_apply(state, noise + perturb)
    a_minus = policy_apply(state, noise - perturb)

    # Directional-derivative magnitude at each action timestep
    change = jnp.linalg.norm(
        (a_plus - a_minus) / (2.0 * eps),
        axis=-1,
    )
    return change  # (C,)


def _average_column_with_keys(
    policy_apply,
    state,
    noise,
    latent_idx,
    eps: float,
    direction_keys,         # (num_directions, 2)  – PRNG keys
):
    """
    Average `finite_difference_column` over `num_directions` random directions.
    Uses a Python loop so that non-vmappable policy_apply functions (e.g.
    pi0 inference through Policy.infer) work correctly.
    """
    results = [
        finite_difference_column(policy_apply, state, noise, latent_idx, eps, key)
        for key in direction_keys
    ]
    return jnp.mean(jnp.stack(results), axis=0)   # (C_action,)


def sensitivity_matrix(
    policy_apply,
    state,
    noise,
    eps: float = 1e-2,
    num_directions: int = 20,
    seed: int = 0,
):
    """
    Compute the (C_action, C_latent) sensitivity matrix for one (state, noise) pair.

    Entry [i, j]  = ||∂a_i / ∂z_j||  (L2 over action dim, averaged over
    `num_directions` random perturbation directions of z_j).

    Args:
        policy_apply   : callable (state, noise[C,D]) -> actions[T,A]
                         state  : observation / context forwarded unchanged
                         noise  : (C, D) DSRL latent noise
                         actions: (T, A) pi0 output actions
        state          : observation / context
        noise          : (C, D) — DSRL actor output, NOT the pi0-expanded form
        eps            : perturbation magnitude
        num_directions : number of random unit directions per latent column
        seed           : base PRNG seed

    Returns:
        mat : (T, C) sensitivity matrix, where T = action timesteps and
              C = number of latent noise timesteps
    """
    C = noise.shape[0]
    base_key = jax.random.PRNGKey(seed)

    # One key block per latent column, subdivided into per-direction keys.
    col_keys = jax.random.split(base_key, C)                       # (C, 2)
    direction_keys_per_col = [
        jax.random.split(col_keys[i], num_directions)
        for i in range(C)
    ]                                                               # list of (num_directions, 2)

    cols = [
        _average_column_with_keys(
            policy_apply, state, noise, i, eps, direction_keys_per_col[i]
        )
        for i in range(C)
    ]                                                               # list of (T,)

    return jnp.stack(cols).T   # (T, C)


def sensitivity_matrix_over_states(
    policy_apply,
    states,         # list / array of N states
    noises,         # list / array of N noise tensors, each (C, D)
    eps: float = 1e-2,
    num_directions: int = 20,
    seed: int = 0,
):
    """
    Compute the sensitivity matrix averaged over N (state, noise) pairs.

    Args:
        policy_apply   : callable (params, state, noise[C,D]) -> actions[C,A]
        params         : policy parameters
        states         : (N, *state_shape)
        noises         : (N, C, D)
        eps            : perturbation magnitude
        num_directions : random directions per column
        seed           : base PRNG seed

    Returns:
        mean_mat : (C_action, C_latent) sensitivity matrix averaged over N
    """
    # Use independent seeds per state so state_i does not share directions
    # with state_j.
    state_seeds = jnp.arange(len(noises)) * 1000 + seed

    mats = []
    for i, (s, z, st) in enumerate(zip(states, noises, state_seeds)):
        mat = sensitivity_matrix(
            policy_apply, s, z,
            eps=eps, num_directions=num_directions, seed=int(st),
        )
        mats.append(mat)

    return jnp.mean(jnp.stack(mats, axis=0), axis=0)   # (C_action, C_latent)


def plot_sensitivity_matrix(
    mat,
    title: str = "Sensitivity matrix  ||∂aᵢ / ∂zⱼ||",
    action_label: str = "Action timestep  i", 
    latent_label: str = "Latent timestep  j  (perturbed)",
    ax=None,
    cmap: str = "viridis",
):
    """
    Heat-map of a (C_action, C_latent) sensitivity matrix.

    Args:
        mat          : (C_action, C_latent) numpy / jnp array
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

    # Annotate cells if matrix is small enough
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


def sensitivity_matrix_test(noise_actor_dir, libero_suite, task_id, N=100,
                            num_directions=20, eps=1e-3, seed=0, filename="sensitivity_matrix"):
    """
    Compute the pi0 action sensitivity to DSRL latent noise, averaged over N
    sampled environment states.

    For each sampled state:
      1. Run the restored DSRL actor on the observation to obtain a (C, D)
         noise sample (the operating point for the finite differences).
      2. Perturb each latent timestep independently and measure how much each
         pi0 action timestep changes  →  one column of the sensitivity matrix.
      3. Average the resulting (T, C) matrix over all N states.

    Args:
        noise_actor_dir : path to a ``checkpointXXX`` directory produced by
                          PixelSACLearner.save_checkpoint
        libero_suite    : LIBERO benchmark suite name, e.g. ``libero_90``
        task_id         : integer task index within the suite
        N               : number of states to sample
        num_directions  : random perturbation directions per latent column
        eps             : perturbation magnitude
        seed            : base PRNG seed
    """
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
    from jaxrl2.utils.general_utils import AttrDict
    from libero.libero import benchmark

    # Load models
    def load_noise_actor(ckpt_dir):
        """Restore a PixelSACLearner using the companion config JSON."""
        return PixelSACLearner.restore_from_checkpoint_dir(ckpt_dir)

    def policy_load(pi0_ckpt="pi05_libero"):
        from examples.train_sim import CHECKPOINTS, _load_pi0_checkpoint
        from openpi.training import config as _openpi_config
        ckpt_dir = _load_pi0_checkpoint(pi0_ckpt)
        cfg = _openpi_config.get_config(CHECKPOINTS[pi0_ckpt]["config"])
        return policy_config.create_trained_policy(cfg, ckpt_dir)

    print("Loading pi05 policy...", flush=True)
    agent_dp = policy_load()
    print("Restoring DSRL noise actor...", flush=True)
    agent = load_noise_actor(noise_actor_dir)

    C, D = agent.action_chunk_shape       # e.g. (1, 32) baseline, (10, 32) chunky
    print(f"agent.action_chunk_shape: {agent.action_chunk_shape}")
    pi05_horizon = agent_dp.action_horizon    # 10

    # Wrap pi0 inference as a (obs_pi_zero, noise_cd) → (T, A) function.
    # noise_cd : (C, D) DSRL output — prepared and padded internally.
    def policy_apply(obs_pi_zero, noise_cd):
        """Evaluate pi0 given a (C, D) DSRL noise sample.

        Converts the 2-D DSRL latent to the 3-D format expected by pi0,
        runs inference, and returns the (T, A) action chunk.
        """
        noise_pi0 = _prepare_pi0_noise(noise_cd, agent, pi05_horizon)  # (1, T, D)
        return agent_dp.infer(obs_pi_zero, noise=noise_pi0)["actions"]  # (T, A)

    # Set up LIBERO environment and variant descriptor
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_suite]()
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    num_init_states = len(init_states)

    variant = AttrDict({
        'env': 'libero',
        'resize_image': 64,
        'task_description': task.language,
    })

    env = create_libero_env(task, 256, seed)

    def _collect_obs(init_state_idx):
        """Reset env to a preset initial state and return obs dicts."""
        env.reset()
        obs = env.set_init_state(init_states[init_state_idx % num_init_states])
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)
        obs_dict = {
            'pixels': curr_image[np.newaxis, ..., np.newaxis],
            'state':  qpos[np.newaxis, ..., np.newaxis],
        }
        obs_pi_zero = obs_to_pi_zero_input(obs, variant)
        return obs_dict, obs_pi_zero

    # Single-state sanity check (first init state)
    obs_dict_0, obs_pi_zero_0 = _collect_obs(0)
    noise_cd_0 = agent.sample_actions(obs_dict_0, marginalize_logprobs=False, use_actor_diff=False) 

    if agent.action_chunk_shape[0] == 1:
        noise_repeat = np.repeat(
            noise_cd_0[-1:, :], pi05_horizon - noise_cd_0.shape[0], axis=0
        )
        noise_cd_0 = np.concatenate([noise_cd_0, noise_repeat], axis=0)
    else:
        noise_cd_0 = noise_cd_0[0]
    print(f"noise_cd_0 shape: {noise_cd_0.shape}")

    mat_0 = sensitivity_matrix(
        policy_apply, obs_pi_zero_0, noise_cd_0,
        eps=eps, num_directions=num_directions, seed=seed,
    )
    print(f"Single-state sensitivity matrix shape: {mat_0.shape}")
    print(mat_0)

    # Average over N sampled states
    print(f"\nComputing sensitivity matrix over {N} states...", flush=True)
    states_list  = []
    noises_list  = []

    for i in range(N):
        obs_dict_i, obs_pi_zero_i = _collect_obs(i)
        noise_cd_i = agent.sample_actions(obs_dict_i, marginalize_logprobs=False, use_actor_diff=False)
        if agent.action_chunk_shape[0] == 1:
            noise_repeat = np.repeat(
                noise_cd_i[-1:, :], pi05_horizon - noise_cd_i.shape[0], axis=0
            )
            noise_cd_i = np.concatenate([noise_cd_i, noise_repeat], axis=0)
        else:
            noise_cd_i = noise_cd_i[0]

        states_list.append(obs_pi_zero_i)
        noises_list.append(noise_cd_i)
        if (i + 1) % 10 == 0:
            print(f"  collected {i + 1}/{N} states", flush=True)

    mean_mat = sensitivity_matrix_over_states(
        policy_apply, states_list, noises_list,
        eps=eps, num_directions=num_directions, seed=seed,
    )
    print("Mean sensitivity matrix:")
    print(mean_mat)

    fig, ax = plot_sensitivity_matrix(
        mean_mat,
        title=f"Avg. sensitivity  ({N} states,  {libero_suite} task {task_id})",
        action_label=f"pi05 action timestep  i  (of {pi05_horizon})",
        latent_label=f"DSRL latent timestep  j  (of {C}, perturbed)",
    )
    out_path = f"plots/plots/sensitivity_matrices/{filename}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return mean_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_actor_dir", type=str, required=True,
                        help="Path to checkpointXXX directory from PixelSACLearner")
    parser.add_argument("--libero_suite", type=str, default="libero_90",
                        help="LIBERO benchmark suite (default: libero_90)")
    parser.add_argument("--task_id", type=int, default=28,
                        help="Task index within the suite (default: 28)")
    parser.add_argument("--N", type=int, default=100,
                        help="Number of states to average over (default: 100)")
    parser.add_argument("--num_directions", type=int, default=20,
                        help="Random perturbation directions per column (default: 20)")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Perturbation magnitude (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filename", type=str, default="sensitivity_matrix",
                        help="Filename to save the sensitivity matrix (default: sensitivity_matrix)")
    args = parser.parse_args()
    sensitivity_matrix_test(
        noise_actor_dir=args.noise_actor_dir,
        libero_suite=args.libero_suite,
        task_id=args.task_id,
        N=args.N,
        num_directions=args.num_directions,
        eps=args.eps,
        seed=args.seed,
        filename=args.filename,
    )
