"""Jacobian block analysis for the pi0 diffusion policy noise interface.

Computes and visualises the raw (A, D) Jacobian block

    J[action_idx, :, noise_idx, :]  =  ∂a_{action_idx} / ∂z_{noise_idx}

where the full Jacobian J has shape (T_action, A, T_noise, D) and is obtained
via jax.jacfwd.  Unlike gradient_sensitivity.py (which collapses to Frobenius
norms) this module keeps the signed partial-derivative values so the directional
structure of the noise-to-action mapping is visible.

Typical usage
-------------
python -m jaxrl2.tests.jacobian \\
    --checkpoint pi05_base \\
    --action_idx 0 --noise_idx 0 \\
    --N 10 --libero_suite libero_90 --task_id 1
"""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.training import config as _openpi_config

from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos
from examples.train_sim import CHECKPOINTS, load_pi0_checkpoint, load_norm_stats_for_checkpoint

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import AttrDict
from jaxrl2.tests.gradient_sensitivity import (
    _prepare_pi0_noise,
    _preprocess_obs,
    create_libero_env,
)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ---------------------------------------------------------------------------
# Core Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian_block(agent_dp, obs_proc, noise_pi0, action_idx=0, noise_idx=0):
    """Compute the (A, D) Jacobian block for one (action, noise) timestep pair.

    The full Jacobian J = d(actions) / d(noise) has shape (T_action, A, T_noise, D).
    This function returns J[action_idx, :, noise_idx, :] of shape (A, D).

    Args:
        agent_dp   : trained pi0 policy (has _sample_actions and _input_transform)
        obs_proc   : preprocessed Observation (output of _preprocess_obs)
        noise_pi0  : (T_noise, D) noise array at which to differentiate
        action_idx : action timestep to select (default 0)
        noise_idx  : noise timestep to select (default 0)

    Returns:
        J_block : (A, D) array of signed partial derivatives
    """
    def fn(z):
        # z: (T_noise, D) — add batch dim, remove it from output
        return agent_dp._sample_actions(obs_proc, noise=z[None])[0]  # (T_action, A)

    J = jax.jacfwd(fn)(noise_pi0)          # (T_action, A, T_noise, D)
    return J[action_idx, :, noise_idx, :]  # (A, D)


def compute_jacobian_block_over_states(
    agent_dp, obs_procs, noises_pi0, action_idx=0, noise_idx=0
):
    """Compute the (A, D) Jacobian block averaged over N (state, noise) pairs.

    Args:
        agent_dp   : trained pi0 policy
        obs_procs  : list of N preprocessed Observations
        noises_pi0 : list of N (T_noise, D) noise arrays
        action_idx : action timestep to select (default 0)
        noise_idx  : noise timestep to select (default 0)

    Returns:
        mean_block : (A, D) mean Jacobian block
    """
    blocks = [
        compute_jacobian_block(agent_dp, obs_proc, noise_pi0, action_idx, noise_idx)
        for obs_proc, noise_pi0 in zip(obs_procs, noises_pi0)
    ]
    return jnp.mean(jnp.stack(blocks, axis=0), axis=0)  # (A, D)

    
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_jacobian_block(
    J_block,
    action_idx=0,
    noise_idx=0,
    title=None,
    ax=None,
    cmap="RdBu_r",
    num_actions=None,
):
    """Heatmap of a signed (A, D) Jacobian block.

    Uses a diverging colormap centred at zero so positive and negative partial
    derivatives are immediately distinguishable.  Cell values are annotated
    when A * D ≤ 200.

    pi0 zero-pads robot actions to its internal action_dim (32).  Pass
    ``num_actions`` to crop the plot to only the first N rows, discarding the
    zero-gradient padding rows.

    Args:
        J_block     : (A, D) array of partial derivatives
        action_idx  : action timestep index (used in auto-title and labels)
        noise_idx   : noise timestep index (used in auto-title and labels)
        title       : plot title; auto-generated from indices if None
        ax          : existing Axes to draw into (new figure created if None)
        cmap        : matplotlib colormap (diverging recommended)
        num_actions : if set, crop J_block to its first num_actions rows before
                      plotting (use to hide zero-padding rows, e.g. 7 for LIBERO)

    Returns:
        fig, ax
    """
    J_block = np.array(J_block)
    if num_actions is not None:
        J_block = J_block[:num_actions, :]
    A, D = J_block.shape

    if title is None:
        title = (
            f"Jacobian  ∂a_{action_idx} / ∂z_{noise_idx}"
            f"  (A={A}, D={D})"
        )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(4, D * 0.4 + 1), max(3, A * 0.6 + 1))
        )
    else:
        fig = ax.get_figure()

    vabs = float(max(abs(J_block.min()), abs(J_block.max())))
    vabs_min=float(min(abs(J_block.min()), abs(J_block.max())))
    if vabs == 0.0:
        vabs = 1.0  # avoid degenerate colour scale for zero matrices

    im = ax.imshow(
        J_block,
        aspect="auto",
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        vmin=-vabs_min,
        vmax=vabs,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("∂a / ∂z", fontsize=9)

    ax.set_xlabel(f"Noise dim  (of {D})", fontsize=9)
    ax.set_ylabel(f"Action dim  (of {A})", fontsize=9)
    ax.set_title(title, fontsize=10)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if A * D <= 200:
        for i in range(A):
            for j in range(D):
                v = float(J_block[i, j])
                color = "white" if abs(v) > 0.6 * vabs else "black"
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=6, color=color,
                )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def jacobian_test(
    noise_actor_dir=None,
    libero_suite="libero_90",
    task_id=None,
    N=1,
    seed=0,
    checkpoint="pi05_base",
    action_idx=0,
    noise_idx=0,
    filename="jacobian",
    num_actions=None,
    run_over_trajectory=False,
    query_frequency=5,
    max_timesteps=400,
):
    """Compute and plot the Jacobian block ∂a_{action_idx} / ∂z_{noise_idx}.

    Steps:
      1. Load the pi0 checkpoint and (optionally) the DSRL noise actor.
      2. Sample N (state, noise) pairs from the LIBERO environment.
      3. Compute the (A, D) Jacobian block, averaged over all N states.
      4. Save the figure to plots/plots/jacobians/ and the raw array as .npy.

    Args:
        noise_actor_dir : path to PixelSACLearner checkpoint dir (random if None)
        libero_suite    : LIBERO benchmark suite name
        task_id         : task index within the suite (None → all tasks)
        N               : number of (state, noise) samples to average over
        seed            : RNG seed
        checkpoint      : pi0 checkpoint key in CHECKPOINTS
        action_idx      : which action timestep to differentiate (default 0)
        noise_idx       : which noise timestep to perturb (default 0)
        filename        : output filename stem
        num_actions     : if set, crop the plot to only the first num_actions
                          action-dim rows (use 7 for LIBERO to hide zero-padding)
        run_over_trajectory : if True, run over a trajectory instead of N states
        query_frequency   : how often to query the policy (default 5)
        max_timesteps     : maximum number of timesteps to run over (default 400)
    Returns:
        mean_J_block : (A, D) averaged Jacobian block (full, before any crop)
    """
    out_dir = pathlib.Path("plots/plots/jacobians")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    def _load_policy(ckpt_key):
        ckpt_dir   = load_pi0_checkpoint(ckpt_key)
        cfg        = _openpi_config.get_config(CHECKPOINTS[ckpt_key]["config"])
        norm_stats = load_norm_stats_for_checkpoint(ckpt_key)
        print(f"Loading openpi policy '{ckpt_key}' from {ckpt_dir}", flush=True)
        return policy_config.create_trained_policy(cfg, ckpt_dir, norm_stats=norm_stats)

    print("Loading policy...", flush=True)
    agent_dp       = _load_policy(checkpoint)
    action_horizon = agent_dp.action_horizon

    # ------------------------------------------------------------------
    # Load (optional) noise actor
    # ------------------------------------------------------------------
    if noise_actor_dir is not None:
        print("Restoring DSRL noise actor...", flush=True)
        agent_noise = PixelSACLearner.restore_from_checkpoint_dir(noise_actor_dir)
        C, D = agent_noise.action_chunk_shape
    else:
        agent_noise = None
        C, D = (10, 32)
    print(f"C={C}, D={D}, action_horizon={action_horizon}", flush=True)

    # ------------------------------------------------------------------
    # Build environment / state pool
    # ------------------------------------------------------------------
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[libero_suite]()

    if task_id is None:
        state_pool = []
        for tid in range(task_suite.n_tasks):
            suite_task = task_suite.get_task(tid)
            for init_state in task_suite.get_task_init_states(tid):
                state_pool.append((suite_task, init_state))
        if len(state_pool) == 0:  
            raise ValueError(f"No init states found for suite {libero_suite}")
        print(
            f"Pool: {task_suite.n_tasks} tasks, {len(state_pool)} init states",
            flush=True,
        )
        task_for_env = state_pool[0][0]
    else:
        suite_task  = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        if init_states is None:
            raise ValueError(
                f"No init states found for suite {libero_suite}, task_id {task_id}"
            )
        state_pool   = [(suite_task, s) for s in init_states]
        task_for_env = suite_task
        print(f"Task {task_id}: {len(state_pool)} init states", flush=True)

    variant = AttrDict({
        "env":              "libero",
        "resize_image":     64,
        "task_description": task_for_env.language,
    })
    env = create_libero_env(task_for_env, 256, seed)

    def _collect_obs(idx):
        task_i, init_state_i     = state_pool[idx % len(state_pool)]
        variant.task_description = task_i.language
        env.reset()
        obs        = env.set_init_state(init_state_i)
        curr_image = obs_to_img(obs, variant)
        qpos       = obs_to_qpos(obs, variant)
        obs_dict   = {
            "pixels": curr_image[np.newaxis, ..., np.newaxis],
            "state":  qpos[np.newaxis, ..., np.newaxis],
        }
        return obs_dict, obs_to_pi_zero_input(obs, variant)

    def _get_noise(obs_dict, i):
        if agent_noise is not None:
            noise_cd = agent_noise.sample_actions(
                obs_dict, marginalize_logprobs=False, use_actor_diff=False
            )
            if agent_noise.action_chunk_shape[0] == 1:
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
        return noise_cd  # (C, D)

    # ------------------------------------------------------------------
    # Collect N (state, noise) pairs
    # ------------------------------------------------------------------
    print(f"Collecting {N} state(s)...", flush=True)
    obs_procs_list = []
    noises_pi0_list = []
    if not run_over_trajectory:
        for i in range(N):
            obs_dict_i, obs_pi_zero_i = _collect_obs(i)
            noise_cd_i                = _get_noise(obs_dict_i, i)
            obs_procs_list.append(_preprocess_obs(agent_dp, obs_pi_zero_i))
            noises_pi0_list.append(_prepare_pi0_noise(noise_cd_i, action_horizon)[0])
            if (i + 1) % 10 == 0:
                print(f"  collected {i + 1}/{N}", flush=True)
    else:
        obs = env.reset()
        for t in range(max_timesteps):
            curr_image = obs_to_img(obs, variant)
            qpos = obs_to_qpos(obs, variant)
            obs_dict = {
                "pixels": curr_image[np.newaxis, ..., np.newaxis],
                "state":  qpos[np.newaxis, ..., np.newaxis],
            }
            if t % query_frequency == 0:
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                obs_proc = _preprocess_obs(agent_dp, obs_pi_zero)
                noise_cd = _get_noise(obs_dict, t)
                noise_pi0 = _prepare_pi0_noise(noise_cd, action_horizon)[0]
                actions = agent_dp.infer(obs_pi_zero, noise=noise_pi0[None])["actions"]
                obs_procs_list.append(obs_proc)
                noises_pi0_list.append(noise_pi0)

            action_t = actions[t % query_frequency]
            obs, reward, done, info = env.step(action_t)
            if done:
                break

    # Std of noise
    noise = np.array(noises_pi0_list)   # (N, 10, 32)

    # Std for each action dimension across all samples and horizons
    dim_std = np.std(noise, axis=(0, 1))   # (32,)

    print("Per-dimension std:")
    print(dim_std)

    print("First 7 dims mean std:", dim_std[:7].mean())
    print("Remaining dims mean std:", dim_std[7:].mean())

    # ------------------------------------------------------------------
    # Compute Jacobian block
    # ------------------------------------------------------------------
    print(
        f"Computing Jacobian block (action_idx={action_idx}, noise_idx={noise_idx})"
        f" averaged over {N} state(s)...",
        flush=True,
    )
    mean_J_block = compute_jacobian_block_over_states(
        agent_dp, obs_procs_list, noises_pi0_list, action_idx, noise_idx
    )
    print(f"Jacobian block shape: {mean_J_block.shape}")
    if num_actions is not None:
        cropped = np.array(mean_J_block)[:num_actions, :]
        print(f"Cropped to num_actions={num_actions}: shape {cropped.shape}")
        print(f"Value range (cropped): [{float(cropped.min()):.4f}, {float(cropped.max()):.4f}]")
    else:
        print(f"Value range: [{float(mean_J_block.min()):.4f}, {float(mean_J_block.max()):.4f}]")

    # ------------------------------------------------------------------
    # Save raw array
    # ------------------------------------------------------------------
    stem     = f"{filename}_a{action_idx}_n{noise_idx}"
    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, np.array(mean_J_block))
    print(f"Saved array → {npy_path}")

    # ------------------------------------------------------------------
    # Plot and save figure
    # ------------------------------------------------------------------
    task_desc = (
        f"{libero_suite} all_tasks" if task_id is None
        else f"{libero_suite} task {task_id}"
    )
    fig, ax = plot_jacobian_block(
        mean_J_block,
        action_idx=action_idx,
        noise_idx=noise_idx,
        title=(
            f"∂a_{action_idx} / ∂z_{noise_idx}  —  {checkpoint}"
            f"  ({N} state{'s' if N != 1 else ''}, {task_desc})"
        ),
        num_actions=num_actions,
    )

    fig_path = out_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure → {fig_path}")
    plt.close(fig)

    return mean_J_block


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot a single Jacobian block ∂a_i / ∂z_j for pi0"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="pi05_base",
        help="Pi0 checkpoint key in CHECKPOINTS (default: pi05_base)",
    )
    parser.add_argument(
        "--noise_actor_dir", type=str, default=None,
        help="Path to PixelSACLearner checkpoint dir (random noise if omitted)",
    )
    parser.add_argument(
        "--libero_suite", type=str, default="libero_90",
        help="LIBERO benchmark suite (default: libero_90)",
    )
    parser.add_argument(
        "--task_id", type=int, default=None,
        help="Task index within the suite (all tasks if omitted)",
    )
    parser.add_argument(
        "--N", type=int, default=1,
        help="Number of states to average over (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed (default: 0)",
    )
    parser.add_argument(
        "--action_idx", type=int, default=0,
        help="Action timestep index i for ∂a_i / ∂z_j (default: 0)",
    )
    parser.add_argument(
        "--noise_idx", type=int, default=0,
        help="Noise timestep index j for ∂a_i / ∂z_j (default: 0)",
    )
    parser.add_argument(
        "--filename", type=str, default="jacobian",
        help="Output filename stem (default: jacobian)",
    )
    parser.add_argument(
        "--num_actions", type=int, default=None,
        help=(
            "Crop the plot to only the first NUM_ACTIONS action-dim rows, "
            "hiding zero-padding rows.  Use 7 for LIBERO. "
            "The full (uncropped) array is still saved as .npy. (default: no crop)"
        ),
    )
    parser.add_argument(
        "--run_over_trajectory", type=int, default=0,
        help="If 1, run over a trajectory instead of N states (default: 0)",
    )
    parser.add_argument(
        "--query_frequency", type=int, default=5,
        help="How often to query the policy (default: 5)",
    )
    parser.add_argument(
        "--max_timesteps", type=int, default=400,
        help="Maximum number of timesteps to run over (default: 400)",
    )
    args = parser.parse_args()

    jacobian_test(
        noise_actor_dir=args.noise_actor_dir,
        libero_suite=args.libero_suite,
        task_id=args.task_id,
        N=args.N,
        seed=args.seed,
        checkpoint=args.checkpoint,
        action_idx=args.action_idx,
        noise_idx=args.noise_idx,
        filename=args.filename,
        num_actions=args.num_actions,
        run_over_trajectory=args.run_over_trajectory,
        query_frequency=args.query_frequency,
        max_timesteps=args.max_timesteps,
    )
