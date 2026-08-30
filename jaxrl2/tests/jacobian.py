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
import json
import pathlib
import sys

# Allow importing shared plot helpers from the sibling plots/ package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "plots"))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

from libero.libero import benchmark

from plot_fonts import PLOT_FONT_FAMILY

plt.rcParams.update({
    "font.family": PLOT_FONT_FAMILY,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Core Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian_block(
    agent_dp, obs_proc, noise_pi0, action_idx=0, noise_idx=0, average_over_chunk=False
):
    """Compute the (A, D) Jacobian block for one (action, noise) timestep pair.

    The full Jacobian J = d(actions) / d(noise) has shape (T_action, A, T_noise, D).
    This function returns J[action_idx, :, noise_idx, :] of shape (A, D), unless
    ``average_over_chunk`` is set, in which case it instead averages J over the
    T_action and T_noise axes.

    Args:
        agent_dp   : trained pi0 policy (has _sample_actions and _input_transform)
        obs_proc   : preprocessed Observation (output of _preprocess_obs)
        noise_pi0  : (T_noise, D) noise array at which to differentiate
        action_idx : action timestep to select (default 0), ignored if
                     average_over_chunk=True
        noise_idx  : noise timestep to select (default 0), ignored if
                     average_over_chunk=True
        average_over_chunk : if True, average J over the T_action and T_noise
                     axes instead of indexing a single (action_idx, noise_idx)
                     pair, so no indices need to be specified

    Returns:
        J_block : (A, D) array of signed partial derivatives
    """
    def fn(z):
        # z: (T_noise, D) — add batch dim, remove it from output
        return agent_dp._sample_actions(obs_proc, noise=z[None])[0]  # (T_action, A)

    J = jax.jacfwd(fn)(noise_pi0)  # (T_action, A, T_noise, D)
    if average_over_chunk:
        return jnp.mean(J, axis=(0, 2))    # (A, D), averaged over T_action & T_noise
    return J[action_idx, :, noise_idx, :]  # (A, D)


def compute_jacobian_block_over_states(
    agent_dp, obs_procs, noises_pi0, action_idx=0, noise_idx=0, average_over_chunk=False
):
    """Compute the (A, D) Jacobian block averaged over N (state, noise) pairs.

    Args:
        agent_dp   : trained pi0 policy
        obs_procs  : list of N preprocessed Observations
        noises_pi0 : list of N (T_noise, D) noise arrays
        action_idx : action timestep to select (default 0), ignored if
                     average_over_chunk=True
        noise_idx  : noise timestep to select (default 0), ignored if
                     average_over_chunk=True
        average_over_chunk : if True, also average each state's Jacobian block
                     over the T_action and T_noise axes (see
                     compute_jacobian_block)

    Returns:
        mean_block : (A, D) mean Jacobian block
    """
    blocks = [
        compute_jacobian_block(
            agent_dp, obs_proc, noise_pi0, action_idx, noise_idx,
            average_over_chunk=average_over_chunk,
        )
        for obs_proc, noise_pi0 in zip(obs_procs, noises_pi0)
    ]
    return jnp.mean(jnp.stack(blocks, axis=0), axis=0)  # (A, D)


# ---------------------------------------------------------------------------
# Influence metric
# ---------------------------------------------------------------------------

def compute_influence(J_block, num_actions=None):
    """Per-latent-dimension influence metric.

    For a Jacobian block J_block = ∂a / ∂w with shape (A, D), the influence of
    latent dimension d is the squared L2 norm of the corresponding column:

        Infl_d(w) = || ∂a / ∂w_d ||_2^2 = sum_a J_block[a, d]^2

    i.e. aggregating (summing squares) one column of the Jacobian over the
    action dimension.

    Args:
        J_block     : (A, D) array of partial derivatives
        num_actions : if set, crop J_block to its first num_actions rows
                      before aggregating (matches the heatmap crop; padding
                      rows are already zero so this normally doesn't change
                      the result)

    Returns:
        influence : (D,) array, one value per latent dimension
    """
    J_block = np.array(J_block)
    if num_actions is not None:
        J_block = J_block[:num_actions, :]
    return np.sum(J_block ** 2, axis=0)  # (D,)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_jacobian_block(
    J_block,
    action_idx=0,
    noise_idx=0,
    title=None,
    ax=None,
    cmap="RdBu",
    num_actions=None,
    average_over_chunk=False,
    vmin=None,
    vmax=None,
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
        action_idx  : action timestep index (used in auto-title and labels),
                      ignored if average_over_chunk=True
        noise_idx   : noise timestep index (used in auto-title and labels),
                      ignored if average_over_chunk=True
        title       : plot title; auto-generated from indices if None
        ax          : existing Axes to draw into (new figure created if None)
        cmap        : matplotlib colormap (diverging recommended)
        num_actions : if set, crop J_block to its first num_actions rows before
                      plotting (use to hide zero-padding rows, e.g. 7 for LIBERO)
        average_over_chunk : if True, use an auto-title reflecting that
                      J_block was averaged over the whole action/noise chunk
                      instead of citing action_idx/noise_idx
        vmin        : fixed heatmap color lower limit; omit with vmax for
                      symmetric auto-scale from data
        vmax        : fixed heatmap color upper limit; omit with vmin for
                      symmetric auto-scale from data

    Returns:
        fig, ax
    """
    J_block = np.array(J_block)
    if num_actions is not None:
        J_block = J_block[:num_actions, :]
    A, D = J_block.shape

    if title is None:
        if average_over_chunk:
            title = (
                f"Jacobian  ∂a / ∂w  (averaged over action & noise chunk)"
                f"  (A={A}, D={D})"
            )
        else:
            title = (
                f"Jacobian  ∂a_{action_idx} / ∂w_{noise_idx}"
                f"  (A={A}, D={D})"
            )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8, 4)
        )
    else:
        fig = ax.get_figure()

    if vmin is None and vmax is None:
        vabs = float(max(abs(J_block.min()), abs(J_block.max())))
        if vabs == 0.0:
            vabs = 1.0  # avoid degenerate colour scale for zero matrices
        vmin, vmax = -vabs, vabs
    elif vmin is None or vmax is None:
        raise ValueError("Pass both vmin and vmax, or neither for auto-scale")

    scale_ref = max(abs(vmin), abs(vmax))

    im = ax.imshow(
        J_block,
        aspect="auto",
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\partial a / \partial w$", fontsize=11)

    ax.set_xlabel(f"Noise dimensions", fontsize=11)
    ax.set_ylabel(f"Action dimensions", fontsize=11)
    ax.set_title(title, fontsize=15, pad=10)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if A * D <= 200:
        for i in range(A):
            for j in range(D):
                v = float(J_block[i, j])
                color = "white" if abs(v) > 0.6 * scale_ref else "black"
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=6, color=color,
                )

    fig.tight_layout()
    return fig, ax


def plot_influence(
    influence,
    action_idx=0,
    noise_idx=0,
    title=None,
    ax=None,
    average_over_chunk=False,
    task_id=None,
):
    """Bar chart of the per-latent-dimension influence metric.

    Args:
        influence  : (D,) array, Infl_d(w) for each latent dimension d
        action_idx : action timestep index (used in auto-title), ignored if
                     average_over_chunk=True
        noise_idx  : noise timestep index (used in auto-title), ignored if
                     average_over_chunk=True
        title      : plot title; auto-generated from indices if None
        ax         : existing Axes to draw into (new figure created if None)
        average_over_chunk : if True, use an auto-title reflecting that the
                     underlying Jacobian was averaged over the whole
                     action/noise chunk instead of citing action_idx/noise_idx
        task_id    : task id (used in auto-title)
    Returns:
        fig, ax
    """
    influence = np.array(influence)
    D = influence.shape[0]

    if title is None:
        if average_over_chunk:
            title = (
                f"Influence Metric $\pi_{{0.5}}$ Task {task_id} LIBERO-90 "
            )
        else:
            title = (
                f"Influence  Infl_d(w) = ||∂a_{action_idx} / ∂w_d||₂²"
                f"  (D={D}, noise_idx={noise_idx})"
            )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.bar(np.arange(D), influence, color="tab:blue")

    ax.set_xlabel("Latent dimension  d", fontsize=11)
    ax.set_ylabel(r"$\mathrm{Infl}_d(w)$", fontsize=11)
    ax.set_title(title, fontsize=15, pad=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Shared save/plot helper
# ---------------------------------------------------------------------------

def _save_jacobian_outputs(
    J_block,
    out_dir,
    stem,
    metrics_base,
    action_idx=0,
    noise_idx=0,
    num_actions=None,
    average_over_chunk=False,
    compute_influence_metric=True,
    plot=True,
    title_str=None,
    task_id=None,
    vmin=None,
    vmax=None,
):
    """Save the raw array, metrics JSON, and (optionally) plots for one Jacobian block.

    Args:
        J_block       : (A, D) array of partial derivatives
        out_dir       : directory to save into (created if missing)
        stem          : output filename stem (without extension)
        metrics_base  : dict of run-level metadata to include verbatim in the
                         saved metrics JSON (e.g. checkpoint, libero_suite,
                         task_id, rollout/timestep info, ...)
        action_idx    : action timestep index (used in auto-titles), ignored
                         if average_over_chunk=True
        noise_idx     : noise timestep index (used in auto-titles), ignored
                         if average_over_chunk=True
        num_actions   : if set, crop J_block to its first num_actions rows
                         before computing influence / plotting
        average_over_chunk : whether J_block was averaged over the whole
                         action/noise chunk (affects auto-titles/metrics)
        compute_influence_metric : if True, compute and save the per-latent
                         influence metric
        plot          : if True, save the heatmap (and influence bar chart)
        title_str     : title string for the heatmap plot
        vmin          : fixed heatmap color lower limit (requires vmax)
        vmax          : fixed heatmap color upper limit (requires vmin)

    Returns:
        influence : (D,) array, or None if compute_influence_metric=False
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    influence = None
    if compute_influence_metric:
        influence = compute_influence(J_block, num_actions=num_actions)
        print(f"Influence Infl_d(w) (D={influence.shape[0]}):")
        print(influence)
        top_dims = np.argsort(influence)[::-1][:5]
        print(f"Top-5 influential latent dims: {top_dims.tolist()}")

    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, np.array(J_block))
    print(f"Saved array → {npy_path}")

    metrics = dict(metrics_base)
    metrics.update({
        "action_idx": None if average_over_chunk else action_idx,
        "noise_idx": None if average_over_chunk else noise_idx,
        "average_over_chunk": average_over_chunk,
        "num_actions": num_actions,
        "jacobian_shape": list(np.array(J_block).shape),
        "jacobian_value_range": [
            float(np.array(J_block).min()),
            float(np.array(J_block).max()),
        ],
        "influence": influence.tolist() if influence is not None else None,
    })
    metrics_path = out_dir / f"{stem}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    if plot:
        fig, ax = plot_jacobian_block(
            J_block,
            action_idx=action_idx,
            noise_idx=noise_idx,
            title=title_str,
            num_actions=num_actions,
            average_over_chunk=average_over_chunk,
            vmin=vmin,
            vmax=vmax,
        )
        fig_path = out_dir / f"{stem}.svg"
        fig.savefig(fig_path)
        print(f"Saved figure → {fig_path}")
        plt.close(fig)

        if influence is not None:
            infl_fig, infl_ax = plot_influence(
                influence,
                action_idx=action_idx,
                noise_idx=noise_idx,
                average_over_chunk=average_over_chunk,
                task_id=task_id,
            )
            infl_fig_path = out_dir / f"{stem}_influence.svg"
            infl_fig.savefig(infl_fig_path)
            print(f"Saved influence figure → {infl_fig_path}")
            plt.close(infl_fig)

    return influence


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
    average_over_chunk=False,
    filename="jacobian",
    num_actions=None,
    run_over_trajectory=False,
    query_frequency=5,
    max_timesteps=400,
    title_str=None,
    num_rollouts=1,
    compute_influence_metric=True,
    plot=True,
    gripper_close_mode=False,
    gripper_close_threshold=0.5,
    vmin=None,
    vmax=None,
):
    """Compute and plot the Jacobian block ∂a_{action_idx} / ∂z_{noise_idx}.

    Steps:
      1. Load the pi0 checkpoint and (optionally) the DSRL noise actor.
      2. Sample N (state, noise) pairs from the LIBERO environment.
      3. Compute the (A, D) Jacobian block, averaged over all N states.
      4. Optionally compute the per-latent-dimension influence metric.
      5. Save the raw array as .npy, metrics as .json, and (optionally) the
         figures to plots/plots/jacobians/.

    Args:
        noise_actor_dir : path to PixelSACLearner checkpoint dir (random if None)
        libero_suite    : LIBERO benchmark suite name
        task_id         : task index within the suite (None → all tasks)
        N               : number of (state, noise) samples to average over
        seed            : RNG seed
        checkpoint      : pi0 checkpoint key in CHECKPOINTS
        action_idx      : which action timestep to differentiate (default 0),
                          ignored if average_over_chunk=True
        noise_idx       : which noise timestep to perturb (default 0),
                          ignored if average_over_chunk=True
        average_over_chunk : if True, average the Jacobian over the whole
                          action and noise chunk (T_action and T_noise axes)
                          instead of indexing a single action_idx/noise_idx
                          pair, so no indices need to be specified
        filename        : output filename stem
        num_actions     : if set, crop the plot to only the first num_actions
                          action-dim rows (use 7 for LIBERO to hide zero-padding)
        run_over_trajectory : if True, run over a trajectory instead of N states
        query_frequency   : how often to query the policy (default 5)
        max_timesteps     : maximum number of timesteps to run over (default 400)
        title_str       : title string for the plot
        num_rollouts    : number of rollouts to run over (default 1)
        compute_influence_metric : if True, compute the per-latent-dimension
                          influence metric Infl_d(w) and include it in the
                          saved metrics JSON (and, if plot=True, plot it)
        plot            : if True, generate and save the heatmap (and, if
                          compute_influence_metric=True, the influence bar
                          chart). If False, only the raw .npy and metrics
                          .json are saved.
        gripper_close_mode : if True, switch to a different mode entirely:
                          instead of the usual whole-trajectory-averaged
                          output, roll out a trajectory (requires
                          run_over_trajectory=True) and, whenever the
                          gripper action (index num_actions - 1) transitions
                          from open to closed (crosses
                          gripper_close_threshold), compute the chunk-
                          averaged Jacobian/influence for that event and
                          save it to plots/plots/jacobian/<filename>/.
                          Requires num_actions to be set.
        gripper_close_threshold : threshold on the gripper action dimension
                          above which it is considered "closing" (default
                          0.5), only used if gripper_close_mode=True
        vmin            : fixed heatmap color lower limit (requires vmax)
        vmax            : fixed heatmap color upper limit (requires vmin)
    Returns:
        mean_J_block : (A, D) averaged Jacobian block (full, before any crop),
                       or None if gripper_close_mode=True (outputs are saved
                       per-event instead)
    """
    if gripper_close_mode and not run_over_trajectory:
        raise ValueError(
            "gripper_close_mode requires run_over_trajectory=True"
        )
    if gripper_close_mode and num_actions is None:
        raise ValueError(
            "gripper_close_mode requires num_actions to be set (e.g. 7 for "
            "LIBERO) to identify the gripper action dimension"
        )

    out_dir = pathlib.Path("plots/plots/jacobians")
    out_dir.mkdir(parents=True, exist_ok=True)
    events_out_dir = pathlib.Path("plots/plots/jacobian") / filename
    event_idx = 0

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
        gripper_events = []
        for r in range(num_rollouts):
            print(f"Running rollout {r + 1}/{num_rollouts}...", flush=True)
            obs = env.reset()
            prev_closed = False
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

                if gripper_close_mode:
                    # obs_proc/noise_pi0 here are those from the most recent
                    # policy query, i.e. the operating point that produced
                    # the currently-executing action chunk (including action_t).
                    gripper_idx = num_actions - 1
                    gripper_val = float(action_t[gripper_idx])
                    is_closed = gripper_val > gripper_close_threshold
                    if is_closed and not prev_closed:
                        print(
                            f"Gripper-close event at rollout {r}, timestep {t} "
                            f"(action[{gripper_idx}]={gripper_val:.3f})",
                            flush=True,
                        )
                        J_event = compute_jacobian_block(
                            agent_dp, obs_proc, noise_pi0, average_over_chunk=True,
                        )
                        stem = f"event{event_idx:03d}_rollout{r}_t{t}"
                        _save_jacobian_outputs(
                            J_event, events_out_dir, stem,
                            metrics_base={
                                "checkpoint": checkpoint,
                                "libero_suite": libero_suite,
                                "task_id": task_id,
                                "rollout": r,
                                "timestep": t,
                                "gripper_idx": gripper_idx,
                                "gripper_close_threshold": gripper_close_threshold,
                                "gripper_value": gripper_val,
                            },
                            num_actions=num_actions,
                            average_over_chunk=True,
                            compute_influence_metric=compute_influence_metric,
                            plot=plot,
                            title_str=title_str,
                            task_id=task_id,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        gripper_events.append({
                            "event_idx": event_idx,
                            "rollout": r,
                            "timestep": t,
                            "gripper_value": gripper_val,
                            "stem": stem,
                        })
                        event_idx += 1
                    prev_closed = is_closed

                obs, reward, done, info = env.step(action_t)
                if done:
                    break

    if gripper_close_mode:
        events_out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = events_out_dir / "events_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "checkpoint": checkpoint,
                "libero_suite": libero_suite,
                "task_id": task_id,
                "num_rollouts": num_rollouts,
                "max_timesteps": max_timesteps,
                "query_frequency": query_frequency,
                "num_actions": num_actions,
                "gripper_close_threshold": gripper_close_threshold,
                "num_events": event_idx,
                "events": gripper_events,
            }, f, indent=2)
        print(
            f"Gripper-close mode: saved {event_idx} event(s) → {events_out_dir} "
            f"(summary: {summary_path})",
            flush=True,
        )
        return None

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
    if average_over_chunk:
        print(
            f"Computing Jacobian block (averaged over the whole action/noise "
            f"chunk) averaged over {N} state(s)...",
            flush=True,
        )
    else:
        print(
            f"Computing Jacobian block (action_idx={action_idx}, noise_idx={noise_idx})"
            f" averaged over {N} state(s)...",
            flush=True,
        )
    mean_J_block = compute_jacobian_block_over_states(
        agent_dp, obs_procs_list, noises_pi0_list, action_idx, noise_idx,
        average_over_chunk=average_over_chunk,
    )
    print(f"Jacobian block shape: {mean_J_block.shape}")
    if num_actions is not None:
        cropped = np.array(mean_J_block)[:num_actions, :]
        print(f"Cropped to num_actions={num_actions}: shape {cropped.shape}")
        print(f"Value range (cropped): [{float(cropped.min()):.4f}, {float(cropped.max()):.4f}]")
    else:
        print(f"Value range: [{float(mean_J_block.min()):.4f}, {float(mean_J_block.max()):.4f}]")

    # ------------------------------------------------------------------
    # Influence metric, save raw array/metrics JSON, and plot(s)
    # ------------------------------------------------------------------
    if average_over_chunk:
        stem = f"{filename}_avgchunk"
    else:
        stem = f"{filename}_a{action_idx}_n{noise_idx}"

    _save_jacobian_outputs(
        mean_J_block, out_dir, stem,
        metrics_base={
            "checkpoint": checkpoint,
            "libero_suite": libero_suite,
            "task_id": task_id,
            "N": N,
            "run_over_trajectory": run_over_trajectory,
        },
        action_idx=action_idx,
        noise_idx=noise_idx,
        num_actions=num_actions,
        average_over_chunk=average_over_chunk,
        compute_influence_metric=compute_influence_metric,
        plot=plot,
        title_str=title_str,
        task_id=task_id,
        vmin=vmin,
        vmax=vmax,
    )

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
        "--average_over_chunk", type=int, default=0,
        help=(
            "If 1, average the Jacobian over the whole action/noise chunk "
            "(T_action and T_noise axes) instead of indexing a single "
            "action_idx/noise_idx pair, so --action_idx/--noise_idx don't "
            "need to be specified (default: 0)"
        ),
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
    parser.add_argument(
        "--title_str", type=str, default=None,
        help="Title string for the plot (default: None)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=1,
        help="Number of rollouts to run over (default: 1)",
    )
    parser.add_argument(
        "--compute_influence", type=int, default=1,
        help=(
            "If 1, compute the per-latent-dimension influence metric "
            "Infl_d(w) = ||da/dw_d||_2^2 and include it in the metrics JSON "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--plot", type=int, default=1,
        help=(
            "If 1, generate and save plots (heatmap and, if "
            "--compute_influence 1, the influence bar chart). If 0, only "
            "the raw .npy and metrics .json are saved (default: 1)"
        ),
    )
    parser.add_argument(
        "--vmin", type=float, default=None,
        help=(
            "Fixed heatmap color lower limit (requires --vmax; omit both "
            "for auto-scale)"
        ),
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help=(
            "Fixed heatmap color upper limit (requires --vmin; omit both "
            "for auto-scale)"
        ),
    )
    parser.add_argument(
        "--gripper_close_mode", type=int, default=0,
        help=(
            "If 1, switch to a different mode: instead of the usual "
            "whole-trajectory-averaged output, roll out a trajectory and, "
            "whenever the gripper action (index num_actions - 1) transitions "
            "from open to closed (crosses --gripper_close_threshold), save a "
            "chunk-averaged Jacobian/influence for that event to "
            "plots/plots/jacobian/<filename>/. Requires --run_over_trajectory 1 "
            "and --num_actions to be set (default: 0)"
        ),
    )
    parser.add_argument(
        "--gripper_close_threshold", type=float, default=0.5,
        help=(
            "Threshold on the gripper action dimension above which it is "
            "considered closing, only used if --gripper_close_mode 1 "
            "(default: 0.5)"
        ),
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
        average_over_chunk=bool(args.average_over_chunk),
        filename=args.filename,
        num_actions=args.num_actions,
        run_over_trajectory=args.run_over_trajectory,
        query_frequency=args.query_frequency,
        max_timesteps=args.max_timesteps,
        title_str=args.title_str,
        num_rollouts=args.num_rollouts,
        compute_influence_metric=bool(args.compute_influence),
        plot=bool(args.plot),
        gripper_close_mode=bool(args.gripper_close_mode),
        gripper_close_threshold=args.gripper_close_threshold,
        vmin=args.vmin,
        vmax=args.vmax,
    )
