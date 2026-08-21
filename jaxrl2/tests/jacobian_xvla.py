"""Jacobian block analysis for the X-VLA denoising policy noise interface.

X-VLA analogue of ``jaxrl2/tests/jacobian.py``. Computes and visualises the raw
(A, D) Jacobian block

    J[action_idx, :, noise_idx, :]  =  d a_{action_idx} / d z_{noise_idx}

where the full Jacobian J has shape (T_action, A, T_noise, D). X-VLA is a
PyTorch model, so ``jax.jacfwd`` (used by ``jacobian.py`` for pi0) cannot be
used. Instead, following the sequential-VJP strategy from
``gradient_sensitivity_xvla.py`` (checkpoint each denoise step, then a single
forward pass followed by per-output-scalar ``torch.autograd.grad`` calls so
the full Jacobian is never materialized), but since a single Jacobian *block*
only needs one ``action_idx`` row (or a chunk-averaged row) instead of the
full (T_action, T_noise) sensitivity matrix, this module only loops over the
``A`` action dimensions rather than all ``T_action * A`` output scalars.

Typical usage
-------------
python -m jaxrl2.tests.jacobian_xvla \\
    --checkpoint xvla_libero \\
    --action_idx 0 --noise_idx 0 \\
    --N 10 --libero_suite libero_90 --task_id 1
"""

import argparse
import json
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import jax

from examples.train_utils_sim import (
    obs_to_img,
    obs_to_qpos,
    obs_to_xvla_input,
    prepare_libero_episode_for_xvla,
)
from examples.xvla_policy import XVLAPolicy

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import AttrDict
from jaxrl2.tests.gradient_sensitivity import (
    _prepare_pi0_noise,
    create_libero_env,
)
from jaxrl2.tests.gradient_sensitivity_xvla import (
    _load_xvla_policy,
    _preprocess_obs_xvla,
)

from libero.libero import benchmark

plt.rcParams.update({
    "mathtext.fontset": "cm",
})


# ---------------------------------------------------------------------------
# Core Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian_block_xvla(
    agent_dp, obs_proc, noise_td, action_idx=0, noise_idx=0, average_over_chunk=False
):
    """Compute the (A, D) Jacobian block for one (action, noise) timestep pair (X-VLA).

    Differentiates the checkpointed ``model.generate_actions_from_enc`` output
    (raw model actions, shape (T_action, A), with sigmoid gripper postprocess)
    w.r.t. the input noise (T_noise, D), via sequential ``torch.autograd.grad``
    calls — one per action dimension ``A`` — instead of a full Jacobian
    (avoids materializing the (T_action, A, T_noise, D) tensor, and avoids the
    ``T_action * A`` backward passes that a full sensitivity matrix would
    require, since only one action_idx row (or a chunk-average) is needed
    here).

    Args:
        agent_dp   : XVLAPolicy (has .model, .device, .steps)
        obs_proc   : dict from ``_preprocess_obs_xvla`` (enc, proprio, domain_id)
        noise_td   : (T_noise, D) noise array at which to differentiate
        action_idx : action timestep to select (default 0), ignored if
                     average_over_chunk=True
        noise_idx  : noise timestep to select (default 0), ignored if
                     average_over_chunk=True
        average_over_chunk : if True, average J over the T_action and T_noise
                     axes instead of indexing a single (action_idx, noise_idx)
                     pair, so no indices need to be specified

    Returns:
        J_block : (A, D) numpy array of signed partial derivatives
    """
    enc = obs_proc["enc"]
    proprio = obs_proc["proprio"]
    domain_id = obs_proc["domain_id"]
    steps = agent_dp.steps
    model = agent_dp.model

    noise_t = torch.tensor(
        np.array(noise_td, dtype=np.float32, copy=True),
        device=agent_dp.device,
        dtype=proprio.dtype,
        requires_grad=True,
    )

    actions = model.generate_actions_from_enc(
        enc,
        domain_id,
        proprio,
        noise_t[None],
        steps=steps,
        use_checkpoint=True,
    )[0]  # (T_action, A)

    T_a, A = actions.shape

    if average_over_chunk:
        targets = actions.mean(dim=0)  # (A,), averaged over T_action
    else:
        targets = actions[action_idx]  # (A,)

    rows = []
    for a_idx in range(A):
        grad = torch.autograd.grad(
            targets[a_idx], noise_t, retain_graph=(a_idx < A - 1), create_graph=False,
        )[0]  # (T_noise, D)
        if average_over_chunk:
            row = grad.mean(dim=0)  # (D,), averaged over T_noise
        else:
            row = grad[noise_idx]  # (D,)
        rows.append(row.detach().float().cpu().numpy())

    del actions, noise_t
    torch.cuda.empty_cache()
    return np.stack(rows, axis=0)  # (A, D)


def compute_jacobian_block_over_states_xvla(
    agent_dp, obs_procs, noises_td, action_idx=0, noise_idx=0, average_over_chunk=False
):
    """Compute the (A, D) Jacobian block averaged over N (state, noise) pairs.

    Args:
        agent_dp   : XVLAPolicy
        obs_procs  : list of N preprocessed obs dicts (see ``_preprocess_obs_xvla``)
        noises_td  : list of N (T_noise, D) noise arrays
        action_idx : action timestep to select (default 0), ignored if
                     average_over_chunk=True
        noise_idx  : noise timestep to select (default 0), ignored if
                     average_over_chunk=True
        average_over_chunk : if True, also average each state's Jacobian block
                     over the T_action and T_noise axes (see
                     compute_jacobian_block_xvla)

    Returns:
        mean_block : (A, D) mean Jacobian block
    """
    blocks = []
    for i, (obs_proc, noise_td) in enumerate(zip(obs_procs, noises_td)):
        blocks.append(
            compute_jacobian_block_xvla(
                agent_dp, obs_proc, noise_td, action_idx, noise_idx,
                average_over_chunk=average_over_chunk,
            )
        )
        torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            print(f"  jacobian {i + 1}/{len(obs_procs)} states", flush=True)
    return np.mean(np.stack(blocks, axis=0), axis=0)  # (A, D)


# ---------------------------------------------------------------------------
# Influence metric
# ---------------------------------------------------------------------------

def compute_influence(J_block, num_actions=None):
    """Per-latent-dimension influence metric.

    For a Jacobian block J_block = da / dw with shape (A, D), the influence of
    latent dimension d is the squared L2 norm of the corresponding column:

        Infl_d(w) = || da / dw_d ||_2^2 = sum_a J_block[a, d]^2

    i.e. aggregating (summing squares) one column of the Jacobian over the
    action dimension.

    Args:
        J_block     : (A, D) array of partial derivatives
        num_actions : if set, crop J_block to its first num_actions rows
                      before aggregating (matches the heatmap crop; padding /
                      unused rows are already ~zero so this normally doesn't
                      change the result much)

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
    cmap="coolwarm",
    num_actions=None,
    average_over_chunk=False,
):
    """Heatmap of a signed (A, D) Jacobian block.

    Uses a diverging colormap centred at zero so positive and negative partial
    derivatives are immediately distinguishable. Cell values are annotated
    when A * D <= 200.

    X-VLA's raw action space includes an unused second-arm block (dims 10:20
    for the single-arm LIBERO domain). Pass ``num_actions`` to crop the plot
    to only the first N rows, discarding those unused rows.

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
                      plotting (use to hide unused second-arm rows, e.g. 10)
        average_over_chunk : if True, use an auto-title reflecting that
                      J_block was averaged over the whole action/noise chunk
                      instead of citing action_idx/noise_idx

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
                f"Jacobian  da / dz  (averaged over action & noise chunk)"
                f"  (A={A}, D={D})"
            )
        else:
            title = (
                f"Jacobian  da_{action_idx} / dz_{noise_idx}"
                f"  (A={A}, D={D})"
            )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8, 4)
        )
    else:
        fig = ax.get_figure()

    vabs = float(max(abs(J_block.min()), abs(J_block.max())))

    im = ax.imshow(
        J_block,
        aspect="auto",
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        vmin=-vabs,
        vmax=vabs,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\partial a / \partial z$", fontsize=11)

    ax.set_xlabel(f"Noise dimensions", fontsize=11)
    ax.set_ylabel(f"Action dimensions", fontsize=11)
    ax.set_title(title, fontsize=15, pad=10)

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


def plot_influence(
    influence,
    action_idx=0,
    noise_idx=0,
    title=None,
    ax=None,
    average_over_chunk=False,
    task_id=None,
    model_label="XVLA",
):
    """Bar chart of the per-latent-dimension influence metric.

    Args:
        influence   : (D,) array, Infl_d(w) for each latent dimension d
        action_idx  : action timestep index (used in auto-title), ignored if
                      average_over_chunk=True
        noise_idx   : noise timestep index (used in auto-title), ignored if
                      average_over_chunk=True
        title       : plot title; auto-generated if None (honors an explicit
                      override, unlike jacobian.py's version)
        ax          : existing Axes to draw into (new figure created if None)
        average_over_chunk : if True, use an auto-title reflecting that the
                      underlying Jacobian was averaged over the whole
                      action/noise chunk instead of citing action_idx/noise_idx
        task_id     : task id (used in auto-title)
        model_label : model name used in the auto-title (default "XVLA")
    Returns:
        fig, ax
    """
    influence = np.array(influence)
    D = influence.shape[0]

    if title is None:
        if average_over_chunk:
            if task_id is not None:
                title = f"Influence Metric {model_label} Task {task_id} LIBERO-90 "
            else:
                title = f"Influence Metric {model_label} (averaged over chunk) "
        else:
            title = (
                f"Influence  Infl_d(w) = ||da_{action_idx} / dw_d||\u2082\u00b2"
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
        title_str     : title string for both the heatmap and influence plots

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
    print(f"Saved array \u2192 {npy_path}")

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
    print(f"Saved metrics \u2192 {metrics_path}")

    if plot:
        fig, ax = plot_jacobian_block(
            J_block,
            action_idx=action_idx,
            noise_idx=noise_idx,
            title=title_str,
            num_actions=num_actions,
            average_over_chunk=average_over_chunk,
        )
        fig_path = out_dir / f"{stem}.svg"
        fig.savefig(fig_path)
        print(f"Saved figure \u2192 {fig_path}")
        plt.close(fig)

        if influence is not None:
            infl_fig, infl_ax = plot_influence(
                influence,
                action_idx=action_idx,
                noise_idx=noise_idx,
                title=title_str,
                average_over_chunk=average_over_chunk,
                task_id=task_id,
            )
            infl_fig_path = out_dir / f"{stem}_influence.svg"
            infl_fig.savefig(infl_fig_path)
            print(f"Saved influence figure \u2192 {infl_fig_path}")
            plt.close(infl_fig)

    return influence


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def jacobian_test_xvla(
    noise_actor_dir=None,
    libero_suite="libero_90",
    task_id=None,
    N=1,
    seed=0,
    checkpoint="xvla_libero",
    action_idx=0,
    noise_idx=0,
    average_over_chunk=False,
    filename="jacobian_xvla",
    num_actions=None,
    run_over_trajectory=False,
    query_frequency=None,
    max_timesteps=800,
    title_str=None,
    num_rollouts=1,
    compute_influence_metric=True,
    plot=True,
    gripper_close_mode=False,
    gripper_close_threshold=0.5,
):
    """Compute and plot the Jacobian block da_{action_idx} / dz_{noise_idx} for X-VLA.

    Steps:
      1. Load the X-VLA checkpoint and (optionally) the DSRL noise actor.
      2. Sample N (state, noise) pairs from the LIBERO environment.
      3. Compute the (A, D) Jacobian block, averaged over all N states.
      4. Optionally compute the per-latent-dimension influence metric.
      5. Save the raw array as .npy, metrics as .json, and (optionally) the
         figures to plots/plots/jacobians/.

    Args:
        noise_actor_dir : path to PixelSACLearner checkpoint dir (random if None)
        libero_suite    : LIBERO benchmark suite name
        task_id         : task index within the suite (None -> all tasks)
        N               : number of (state, noise) samples to average over
        seed            : RNG seed
        checkpoint      : X-VLA checkpoint key ("xvla_libero" or "xvla_base")
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
                          action-dim rows in X-VLA's raw 20-dim EE6D action
                          space (use 10 to hide the unused second-arm dims)
        run_over_trajectory : if True, run over a trajectory instead of N states
        query_frequency   : how often to query the policy. X-VLA's absolute
                          ee6d chunks must be executed in full before replan,
                          so this is forced to action_horizon (with a warning)
                          if left as None or set to anything else
        max_timesteps     : maximum number of timesteps to run over (default 800)
        title_str       : title string for the plot (heatmap and influence chart)
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
                          *executed* (7-dim, post-processed) Libero gripper
                          action transitions from open to closed (crosses
                          gripper_close_threshold), compute the chunk-
                          averaged Jacobian/influence for that event and
                          save it to plots/plots/jacobian/<filename>/. Unlike
                          the pi0 script, this does NOT require num_actions to
                          be set: the gripper index is always
                          agent_dp.env_action_dim - 1, since X-VLA's raw
                          differentiable action space (used for the Jacobian)
                          is a different representation than its executed env
                          action space (used for gripper-close detection).
        gripper_close_threshold : threshold on the executed gripper action
                          dimension above which it is considered "closing"
                          (default 0.5), only used if gripper_close_mode=True
    Returns:
        mean_J_block : (A, D) averaged Jacobian block (full, before any crop),
                       or None if gripper_close_mode=True (outputs are saved
                       per-event instead)
    """
    if gripper_close_mode and not run_over_trajectory:
        raise ValueError(
            "gripper_close_mode requires run_over_trajectory=True"
        )

    out_dir = pathlib.Path("plots/plots/jacobians")
    out_dir.mkdir(parents=True, exist_ok=True)
    events_out_dir = pathlib.Path("plots/plots/jacobian") / filename
    event_idx = 0

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    print("Loading policy...", flush=True)
    agent_dp = _load_xvla_policy(checkpoint)
    action_horizon = agent_dp.action_horizon
    action_dim = agent_dp.action_dim
    env_action_dim = agent_dp.env_action_dim

    if run_over_trajectory:
        if query_frequency is None or query_frequency != action_horizon:
            print(
                f"Warning: X-VLA absolute ee6d chunks must be executed in "
                f"full before replan; overriding query_frequency="
                f"{query_frequency} to action_horizon={action_horizon}",
                flush=True,
            )
            query_frequency = action_horizon

    # ------------------------------------------------------------------
    # Load (optional) noise actor
    # ------------------------------------------------------------------
    if noise_actor_dir is not None:
        print("Restoring DSRL noise actor...", flush=True)
        agent_noise = PixelSACLearner.restore_from_checkpoint_dir(noise_actor_dir)
        C, D = agent_noise.action_chunk_shape
    else:
        agent_noise = None
        C, D = (action_horizon, action_dim)
    print(f"C={C}, D={D}, action_horizon={action_horizon}, action_dim={action_dim}", flush=True)

    # ------------------------------------------------------------------
    # Build environment / state pool
    # ------------------------------------------------------------------
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_suite]()

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
        suite_task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        if init_states is None:
            raise ValueError(
                f"No init states found for suite {libero_suite}, task_id {task_id}"
            )
        state_pool = [(suite_task, s) for s in init_states]
        task_for_env = suite_task
        print(f"Task {task_id}: {len(state_pool)} init states", flush=True)

    variant = AttrDict({
        "env": "libero",
        "resize_image": 64,
        "task_description": task_for_env.language,
        "vla": "xvla",
    })
    env = create_libero_env(task_for_env, 256, seed)

    def _collect_obs(idx):
        task_i, init_state_i = state_pool[idx % len(state_pool)]
        variant.task_description = task_i.language
        env.reset()
        env.set_init_state(init_state_i)
        agent_dp.reset()
        obs = prepare_libero_episode_for_xvla(env)
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)
        obs_dict = {
            "pixels": curr_image[np.newaxis, ..., np.newaxis],
            "state": qpos[np.newaxis, ..., np.newaxis],
        }
        obs_policy = obs_to_xvla_input(obs, variant, env=env)
        return obs_dict, obs_policy

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
    print(f"Collecting {N if not run_over_trajectory else num_rollouts} sample(s)...", flush=True)
    obs_procs_list = []
    noises_list = []
    if not run_over_trajectory:
        for i in range(N):
            obs_dict_i, obs_policy_i = _collect_obs(i)
            noise_cd_i = _get_noise(obs_dict_i, i)
            obs_procs_list.append(_preprocess_obs_xvla(agent_dp, obs_policy_i))
            noises_list.append(_prepare_pi0_noise(noise_cd_i, action_horizon, action_dim=action_dim)[0])
            if (i + 1) % 10 == 0:
                print(f"  collected {i + 1}/{N}", flush=True)
    else:
        gripper_events = []
        for r in range(num_rollouts):
            print(f"Running rollout {r + 1}/{num_rollouts}...", flush=True)
            task_r, init_state_r = state_pool[r % len(state_pool)]
            variant.task_description = task_r.language
            env.reset()
            env.set_init_state(init_state_r)
            agent_dp.reset()
            obs = prepare_libero_episode_for_xvla(env)
            prev_closed = False
            obs_proc = None
            noise_td = None
            for t in range(max_timesteps):
                if t % query_frequency == 0:
                    curr_image = obs_to_img(obs, variant)
                    qpos = obs_to_qpos(obs, variant)
                    obs_dict = {
                        "pixels": curr_image[np.newaxis, ..., np.newaxis],
                        "state": qpos[np.newaxis, ..., np.newaxis],
                    }
                    obs_policy = obs_to_xvla_input(obs, variant, env=env)
                    obs_proc = _preprocess_obs_xvla(agent_dp, obs_policy)
                    noise_cd = _get_noise(obs_dict, t)
                    noise_td = _prepare_pi0_noise(noise_cd, action_horizon, action_dim=action_dim)[0]
                    actions = agent_dp.infer(
                        obs_policy, noise=noise_td[None], proprio_from_step=query_frequency - 1,
                    )["actions"]
                    obs_procs_list.append(obs_proc)
                    noises_list.append(noise_td)

                action_t = actions[t % query_frequency]

                if gripper_close_mode:
                    # obs_proc/noise_td here are those from the most recent
                    # policy query, i.e. the operating point that produced
                    # the currently-executing action chunk (including action_t).
                    gripper_idx = env_action_dim - 1
                    gripper_val = float(action_t[gripper_idx])
                    is_closed = gripper_val > gripper_close_threshold
                    if is_closed and not prev_closed:
                        print(
                            f"Gripper-close event at rollout {r}, timestep {t} "
                            f"(action[{gripper_idx}]={gripper_val:.3f})",
                            flush=True,
                        )
                        J_event = compute_jacobian_block_xvla(
                            agent_dp, obs_proc, noise_td, average_over_chunk=True,
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
            f"Gripper-close mode: saved {event_idx} event(s) \u2192 {events_out_dir} "
            f"(summary: {summary_path})",
            flush=True,
        )
        return None

    # Std of noise
    noise = np.array(noises_list)  # (N, T, D)

    # X-VLA's raw action/noise space is EE6D: dims 0:10 are arm 1
    # (pos3+rot6d+grip1), dims 10:20 are an unused second-arm block for the
    # single-arm LIBERO domain.
    dim_std = np.std(noise, axis=(0, 1))  # (D,)

    print("Per-dimension std:")
    print(dim_std)

    print("Arm-1 dims (0:10) mean std:", dim_std[:10].mean())
    if dim_std.shape[0] > 10:
        print("Arm-2 dims (10:20) mean std:", dim_std[10:].mean())

    # ------------------------------------------------------------------
    # Compute Jacobian block
    # ------------------------------------------------------------------
    if average_over_chunk:
        print(
            f"Computing Jacobian block (averaged over the whole action/noise "
            f"chunk) averaged over {len(obs_procs_list)} state(s)...",
            flush=True,
        )
    else:
        print(
            f"Computing Jacobian block (action_idx={action_idx}, noise_idx={noise_idx})"
            f" averaged over {len(obs_procs_list)} state(s)...",
            flush=True,
        )
    mean_J_block = compute_jacobian_block_over_states_xvla(
        agent_dp, obs_procs_list, noises_list, action_idx, noise_idx,
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
    )

    return mean_J_block


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot a single Jacobian block da_i / dz_j for X-VLA"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="xvla_libero",
        help="X-VLA checkpoint key: xvla_libero or xvla_base (default: xvla_libero)",
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
        help="Action timestep index i for da_i / dz_j (default: 0)",
    )
    parser.add_argument(
        "--noise_idx", type=int, default=0,
        help="Noise timestep index j for da_i / dz_j (default: 0)",
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
        "--filename", type=str, default="jacobian_xvla",
        help="Output filename stem (default: jacobian_xvla)",
    )
    parser.add_argument(
        "--num_actions", type=int, default=None,
        help=(
            "Crop the plot to only the first NUM_ACTIONS action-dim rows in "
            "X-VLA's raw 20-dim EE6D action space, hiding the unused "
            "second-arm rows. Use 10 for single-arm LIBERO. "
            "The full (uncropped) array is still saved as .npy. (default: no crop)"
        ),
    )
    parser.add_argument(
        "--run_over_trajectory", type=int, default=0,
        help="If 1, run over a trajectory instead of N states (default: 0)",
    )
    parser.add_argument(
        "--query_frequency", type=int, default=None,
        help=(
            "How often to query the policy. X-VLA's absolute ee6d chunks "
            "must be executed in full before replan, so this is always "
            "overridden to action_horizon (with a warning) if left unset or "
            "set to something else (default: None -> action_horizon)"
        ),
    )
    parser.add_argument(
        "--max_timesteps", type=int, default=800,
        help="Maximum number of timesteps to run over (default: 800)",
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
        "--gripper_close_mode", type=int, default=0,
        help=(
            "If 1, switch to a different mode: instead of the usual "
            "whole-trajectory-averaged output, roll out a trajectory and, "
            "whenever the executed (7-dim) gripper action transitions from "
            "open to closed (crosses --gripper_close_threshold), save a "
            "chunk-averaged Jacobian/influence for that event to "
            "plots/plots/jacobian/<filename>/. Requires --run_over_trajectory 1 "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--gripper_close_threshold", type=float, default=0.5,
        help=(
            "Threshold on the executed gripper action dimension above which "
            "it is considered closing, only used if --gripper_close_mode 1 "
            "(default: 0.5)"
        ),
    )
    args = parser.parse_args()

    jacobian_test_xvla(
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
        run_over_trajectory=bool(args.run_over_trajectory),
        query_frequency=args.query_frequency,
        max_timesteps=args.max_timesteps,
        title_str=args.title_str,
        num_rollouts=args.num_rollouts,
        compute_influence_metric=bool(args.compute_influence),
        plot=bool(args.plot),
        gripper_close_mode=bool(args.gripper_close_mode),
        gripper_close_threshold=args.gripper_close_threshold,
    )
