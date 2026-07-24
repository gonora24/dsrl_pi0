"""Jacobian SVD analysis for the pi0 diffusion policy noise interface.

Complements gradient_sensitivity.py: instead of collapsing the full Jacobian to a
scalar Frobenius norm per (action-step, noise-step) pair, this module preserves the
(A, D) block structure and performs SVD to expose scale-invariant geometry:

  - Stable rank:          sr  = (Σᵢ σᵢ²) / σ_max²
  - Normalized spectrum:  p   = σ / Σ σ
  - Spectrum entropy:     H   = -Σᵢ pᵢ log(pᵢ)
  - Subspace alignment:   principal angles between top-k right singular subspaces
                          of two checkpoints (base vs. finetuned)

Shape convention (matches gradient_sensitivity.py):
  J          : (T_action, A, T_noise, D)  raw Jacobian from jax.jacfwd
  J_blocks   : (T_action, T_noise, A, D)  transposed for vmapped SVD
  s          : (T_action, T_noise, k)     singular values, k = min(A, D)
  Vt         : (T_action, T_noise, k, D)  right singular vectors (rows)
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
# Core SVD computation
# ---------------------------------------------------------------------------

def compute_jacobian_svd(agent_dp, obs_proc, noise_pi0):
    """Compute per-block SVD of the action/noise Jacobian.

    Args:
        agent_dp  : trained pi0 policy (has _sample_actions and _input_transform)
        obs_proc  : preprocessed Observation (output of _preprocess_obs)
        noise_pi0 : (T_noise, D) noise array at which to differentiate

    Returns:
        s  : (T_action, T_noise, k)    singular values, k = min(A, D)
        Vt : (T_action, T_noise, k, D) right singular vectors (rows = directions
                                        in the D-dim noise space)
    """
    def fn(z):
        return agent_dp._sample_actions(obs_proc, noise=z[None])[0]  # (T_action, A)

    J = jax.jacfwd(fn)(noise_pi0)          # (T_action, A, T_noise, D)
    J_blocks = jnp.transpose(J, (0, 2, 1, 3))  # (T_action, T_noise, A, D)

    def _svd_block(block):
        # block: (A, D)
        _, s_b, Vt_b = jnp.linalg.svd(block, full_matrices=False)
        return s_b, Vt_b  # (k,), (k, D)

    s, Vt = jax.vmap(jax.vmap(_svd_block))(J_blocks)
    # s : (T_action, T_noise, k)
    # Vt: (T_action, T_noise, k, D)
    return s, Vt


# ---------------------------------------------------------------------------
# Scalar summary functions
# ---------------------------------------------------------------------------

def stable_rank(s):
    """Stable / effective rank of each (A, D) Jacobian block.

    stable_rank = (Σᵢ σᵢ²) / σ_max²

    Ranges from 1 (single dominant direction) up to k = min(A, D)
    (perfectly isotropic).  Scale-invariant.

    Args:
        s: (T_action, T_noise, k) singular values (non-negative, non-increasing)

    Returns:
        (T_action, T_noise) stable rank per block
    """
    sum_sq   = jnp.sum(s ** 2, axis=-1)          # (T_action, T_noise)
    max_sq   = jnp.max(s, axis=-1) ** 2          # (T_action, T_noise)
    return sum_sq / (max_sq + 1e-30)


def normalized_spectrum(s):
    """Normalize singular values to a probability simplex.

    p = σ / Σ σ

    Args:
        s: (..., k) singular values

    Returns:
        same shape as s, rows sum to 1
    """
    total = jnp.sum(s, axis=-1, keepdims=True)
    return s / (total + 1e-30)


def spectrum_entropy(s):
    """Shannon entropy of the normalized singular value spectrum.

    H = -Σᵢ pᵢ log(pᵢ),  p = σ / Σ σ

    H = 0 means a single direction dominates; H = log(k) means all k directions
    are equally weighted.

    Args:
        s: (T_action, T_noise, k) singular values

    Returns:
        (T_action, T_noise) entropy per block
    """
    p = normalized_spectrum(s)               # (..., k)
    return -jnp.sum(p * jnp.log(p + 1e-30), axis=-1)


# ---------------------------------------------------------------------------
# Subspace alignment
# ---------------------------------------------------------------------------

def subspace_alignment(Vt1, Vt2, top_k):
    """Principal angles between top-k right singular subspaces of two models.

    For each (action-step i, noise-step j) block, the principal angles θ₁ ≤ ... ≤ θₖ
    between the two k-dim subspaces in ℝᴰ satisfy:

        cos(θₗ) = lth singular value of  Vt1[i,j,:top_k,:] @ Vt2[i,j,:top_k,:].T

    Args:
        Vt1, Vt2 : (T_action, T_noise, k_full, D) right singular vectors, two models
        top_k    : int, number of leading directions to compare

    Returns:
        angles    : (T_action, T_noise, top_k)  principal angles in [0, π/2], radians
        mean_cos  : (T_action, T_noise)          mean cosine similarity (summary scalar)
    """
    V1 = Vt1[:, :, :top_k, :]   # (T_action, T_noise, top_k, D)
    V2 = Vt2[:, :, :top_k, :]

    def _principal_angles(v1, v2):
        # v1, v2: (top_k, D)
        M     = v1 @ v2.T                       # (top_k, top_k)
        sv    = jnp.linalg.svd(M, compute_uv=False)  # (top_k,)
        sv    = jnp.clip(sv, 0.0, 1.0)
        return jnp.arccos(sv), jnp.mean(sv)    # angles, mean_cos

    angles, mean_cos = jax.vmap(jax.vmap(_principal_angles))(V1, V2)
    # angles   : (T_action, T_noise, top_k)
    # mean_cos : (T_action, T_noise)
    return angles, mean_cos


# ---------------------------------------------------------------------------
# Averaging over states
# ---------------------------------------------------------------------------

def svd_analysis_over_states(agent_dp, obs_procs, noises_pi0):
    """Compute and average per-block SVD results over N (state, noise) pairs.

    singular values s are averaged directly (element-wise mean is well-defined
    and useful for stable rank / spectrum comparisons).

    Right singular vectors Vt are averaged element-wise as an approximation — this
    is only meaningful when the subspaces are consistent across states.  For exact
    subspace comparison between two checkpoints, prefer computing subspace_alignment
    per state (using the lists returned here) and averaging the resulting angles.

    Args:
        agent_dp   : trained pi0 policy
        obs_procs  : list of N preprocessed Observations
        noises_pi0 : list of N (T_noise, D) noise arrays

    Returns:
        mean_s  : (T_action, T_noise, k)    averaged singular values
        mean_Vt : (T_action, T_noise, k, D) averaged right singular vectors
        all_s   : list of N (T_action, T_noise, k) per-state singular values
        all_Vt  : list of N (T_action, T_noise, k, D) per-state Vt matrices
    """
    all_s, all_Vt = [], []
    for obs_proc, noise_pi0 in zip(obs_procs, noises_pi0):
        s_i, Vt_i = compute_jacobian_svd(agent_dp, obs_proc, noise_pi0)
        all_s.append(s_i)
        all_Vt.append(Vt_i)

    mean_s  = jnp.mean(jnp.stack(all_s,  axis=0), axis=0)
    mean_Vt = jnp.mean(jnp.stack(all_Vt, axis=0), axis=0)
    return mean_s, mean_Vt, all_s, all_Vt


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _heatmap(mat, title, xlabel, ylabel, ax=None, cmap="viridis", annotate=True, vmin=None, vmax=None):
    """Shared helper: heatmap with colorbar, labels, optional cell annotation."""
    mat = np.array(mat)
    T_a, T_n = mat.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, T_n * 0.6 + 1),
                                        max(3, T_a * 0.6 + 1)))
    else:
        fig = ax.get_figure()

    im   = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper", vmin=vmin, vmax=vmax,
                     interpolation="nearest")
    # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if annotate and T_a * T_n <= 200:
        vmax = vmax if vmax is not None else float(mat.max())
        for i in range(T_a):
            for j in range(T_n):
                v     = float(mat[i, j])
                color = "white" if v < 0.6 * vmax else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    fig.tight_layout()
    return fig, ax


def plot_stable_rank(sr_mat, title="Stable rank  sr(i,j) = Σσ²/σ_max²",
                     action_label="Action timestep  i",
                     latent_label="Latent timestep  j  (perturbed)",
                     ax=None, cmap="plasma", vmin=None, vmax=None):
    """Heatmap of stable rank across (T_action, T_noise) blocks.

    Args:
        sr_mat : (T_action, T_noise) stable rank array

    Returns:
        fig, ax
    """
    return _heatmap(sr_mat, title, latent_label, action_label, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)


def plot_spectrum_entropy(entropy_mat,
                          title="Spectrum entropy  H(i,j)",
                          action_label="Action timestep  i",
                          latent_label="Latent timestep  j  (perturbed)",
                          ax=None, cmap="inferno", vmin=None, vmax=None):
    """Heatmap of per-block singular value spectrum entropy.

    Args:
        entropy_mat : (T_action, T_noise) entropy array

    Returns:
        fig, ax
    """
    return _heatmap(entropy_mat, title, latent_label, action_label, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)


def plot_normalized_spectra(s_base, i, j, s_ft=None,
                             label_base="base", label_ft="finetuned",
                             ax=None):
    """Bar chart of the normalized singular value spectrum for block (i, j).

    Optionally overlays two models side by side for direct comparison.

    Args:
        s_base  : (T_action, T_noise, k) singular values, base model
        i, j    : block indices to plot
        s_ft    : (T_action, T_noise, k) singular values, finetuned model (optional)
        ax      : existing Axes (created if None)

    Returns:
        fig, ax
    """
    p_base = np.array(normalized_spectrum(s_base[i, j]))  # (k,)
    k      = len(p_base)
    x      = np.arange(k)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, k * 0.5 + 1), 3))
    else:
        fig = ax.get_figure()

    if s_ft is None:
        ax.bar(x, p_base, label=label_base, alpha=0.85)
    else:
        p_ft  = np.array(normalized_spectrum(s_ft[i, j]))
        width = 0.38
        ax.bar(x - width / 2, p_base, width, label=label_base, alpha=0.85)
        ax.bar(x + width / 2, p_ft,   width, label=label_ft,   alpha=0.85)
        ax.legend(fontsize=8)

    ax.set_xlabel("Singular value index", fontsize=9)
    ax.set_ylabel("σ / Σσ", fontsize=9)
    ax.set_title(f"Normalized spectrum  block ({i}, {j})", fontsize=10)
    ax.set_xticks(x)
    fig.tight_layout()
    return fig, ax


def plot_subspace_alignment(mean_cos, title="Subspace alignment  mean cos(θ)",
                             action_label="Action timestep  i",
                             latent_label="Latent timestep  j  (perturbed)",
                             ax=None):
    """Heatmap of mean cosine similarity between base/finetuned subspaces.

    Values close to 1: finetuning preserves the same noise directions.
    Values close to 0: the subspaces are nearly orthogonal.

    Args:
        mean_cos : (T_action, T_noise) mean cosine similarity per block

    Returns:
        fig, ax
    """
    return _heatmap(mean_cos, title, latent_label, action_label, ax=ax,
                    cmap="RdYlGn")


# ---------------------------------------------------------------------------
# Top-level test function
# ---------------------------------------------------------------------------

def svd_analysis_test(
    checkpoint_base,
    checkpoint_ft=None,
    noise_actor_dir=None,
    libero_suite="libero_90",
    task_id=None,
    N=100,
    seed=0,
    top_k=3,
    filename="jacobian_svd",
):
    """Compute Jacobian SVD analysis for one or two pi0 checkpoints.

    When checkpoint_ft is provided the function also computes and plots
    subspace alignment between the base and finetuned models.

    Steps:
      1. Load checkpoint(s) and (optionally) the DSRL noise actor.
      2. Sample N (state, noise) pairs from the LIBERO environment.
      3. Compute per-block SVD, averaged over all states, for each checkpoint.
      4. Derive stable rank, spectrum entropy, and (if two checkpoints) subspace
         alignment heatmaps.
      5. Save figures to plots/plots/svd_analyses/ and raw arrays to
         plots/plots/svd_analyses/<filename>.npz.

    Args:
        checkpoint_base  : key in CHECKPOINTS for the base model
        checkpoint_ft    : key in CHECKPOINTS for the finetuned model (or None)
        noise_actor_dir  : path to PixelSACLearner checkpoint dir (or None → random)
        libero_suite     : LIBERO benchmark suite name
        task_id          : task index within the suite (None → all tasks)
        N                : number of (state, noise) samples to average over
        seed             : RNG seed
        top_k            : number of leading singular vectors for subspace alignment
        filename         : stem for output files

    Returns:
        dict with keys 's_base', 'Vt_base', and (if checkpoint_ft) 's_ft',
        'Vt_ft', 'mean_cos', 'angles'.
    """
    out_dir = pathlib.Path("plots/plots/svd_analyses")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load policy / policies
    # ------------------------------------------------------------------
    def _load_policy(ckpt_key):
        ckpt_dir   = load_pi0_checkpoint(ckpt_key)
        cfg        = _openpi_config.get_config(CHECKPOINTS[ckpt_key]["config"])
        norm_stats = load_norm_stats_for_checkpoint(ckpt_key)
        print(f"Loading openpi policy '{ckpt_key}' from {ckpt_dir}", flush=True)
        return policy_config.create_trained_policy(cfg, ckpt_dir, norm_stats=norm_stats)

    print("Loading base policy...", flush=True)
    agent_base     = _load_policy(checkpoint_base)
    action_horizon = agent_base.action_horizon

    agent_ft = None
    if checkpoint_ft is not None:
        print("Loading finetuned policy...", flush=True)
        agent_ft = _load_policy(checkpoint_ft)

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
        if not state_pool:
            raise ValueError(f"No init states found for suite {libero_suite}")
        print(f"Pool: {task_suite.n_tasks} tasks, {len(state_pool)} init states",
              flush=True)
        task_for_env = state_pool[0][0]
    else:
        suite_task    = task_suite.get_task(task_id)
        init_states   = task_suite.get_task_init_states(task_id)
        if init_states is None:
            raise ValueError(
                f"No init states found for suite {libero_suite}, task_id {task_id}")
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
        task_i, init_state_i    = state_pool[idx % len(state_pool)]
        variant.task_description = task_i.language
        env.reset()
        obs                     = env.set_init_state(init_state_i)
        curr_image              = obs_to_img(obs, variant)
        qpos                    = obs_to_qpos(obs, variant)
        obs_dict = {
            "pixels": curr_image[np.newaxis, ..., np.newaxis],
            "state":  qpos[np.newaxis, ..., np.newaxis],
        }
        return obs_dict, obs_to_pi_zero_input(obs, variant)

    def _get_noise(obs_dict, i):
        if agent_noise is not None:
            noise_cd = agent_noise.sample_actions(
                obs_dict, marginalize_logprobs=False, use_actor_diff=False)
            if agent_noise.action_chunk_shape[0] == 1:
                noise_repeat = np.repeat(
                    noise_cd[-1:, :], action_horizon - noise_cd.shape[0], axis=0)
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
    print(f"Collecting {N} states...", flush=True)
    obs_procs_base, obs_procs_ft, noises_pi0 = [], [], []

    for i in range(N):
        obs_dict_i, obs_pi_zero_i = _collect_obs(i)
        noise_cd_i                = _get_noise(obs_dict_i, i)
        obs_procs_base.append(_preprocess_obs(agent_base, obs_pi_zero_i))
        if agent_ft is not None:
            obs_procs_ft.append(_preprocess_obs(agent_ft, obs_pi_zero_i))
        noises_pi0.append(_prepare_pi0_noise(noise_cd_i, action_horizon)[0])
        if (i + 1) % 10 == 0:
            print(f"  collected {i + 1}/{N}", flush=True)

    # ------------------------------------------------------------------
    # SVD analysis — base model
    # ------------------------------------------------------------------
    print("Computing SVD analysis for base model...", flush=True)
    mean_s_base, mean_Vt_base, all_s_base, all_Vt_base = svd_analysis_over_states(
        agent_base, obs_procs_base, noises_pi0)

    sr_base      = stable_rank(mean_s_base)          # (T_action, T_noise)
    entropy_base = spectrum_entropy(mean_s_base)      # (T_action, T_noise)
    print(f"Base — stable rank range: [{float(sr_base.min()):.3f}, "
          f"{float(sr_base.max()):.3f}]")
    print(f"Base — entropy range:     [{float(entropy_base.min()):.3f}, "
          f"{float(entropy_base.max()):.3f}]")

    # ------------------------------------------------------------------
    # SVD analysis — finetuned model (optional)
    # ------------------------------------------------------------------
    results = {
        "s_base":  np.array(mean_s_base),
        "Vt_base": np.array(mean_Vt_base),
    }

    mean_s_ft = mean_Vt_ft = None
    sr_ft = entropy_ft = mean_cos = angles = None

    if agent_ft is not None:
        print("Computing SVD analysis for finetuned model...", flush=True)
        mean_s_ft, mean_Vt_ft, all_s_ft, all_Vt_ft = svd_analysis_over_states(
            agent_ft, obs_procs_ft, noises_pi0)

        sr_ft      = stable_rank(mean_s_ft)
        entropy_ft = spectrum_entropy(mean_s_ft)
        print(f"Finetuned — stable rank range: [{float(sr_ft.min()):.3f}, "
              f"{float(sr_ft.max()):.3f}]")

        # Per-state subspace alignment, then average angles
        all_angles, all_mean_cos = [], []
        for Vt_b, Vt_f in zip(all_Vt_base, all_Vt_ft):
            ang_i, mc_i = subspace_alignment(Vt_b, Vt_f, top_k)
            all_angles.append(ang_i)
            all_mean_cos.append(mc_i)
        angles   = jnp.mean(jnp.stack(all_angles,   axis=0), axis=0)
        mean_cos = jnp.mean(jnp.stack(all_mean_cos, axis=0), axis=0)
        print(f"Subspace alignment — mean cosine range: "
              f"[{float(mean_cos.min()):.3f}, {float(mean_cos.max()):.3f}]")

        results.update({
            "s_ft":     np.array(mean_s_ft),
            "Vt_ft":    np.array(mean_Vt_ft),
            "mean_cos": np.array(mean_cos),
            "angles":   np.array(angles),
        })

    # ------------------------------------------------------------------
    # Save raw arrays
    # ------------------------------------------------------------------
    npz_path = out_dir / f"{filename}.npz"
    np.savez(npz_path, **results)
    print(f"Saved arrays → {npz_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    T_a  = int(mean_s_base.shape[0])
    T_n  = int(mean_s_base.shape[1])
    base_tag = checkpoint_base
    ft_tag   = checkpoint_ft or ""

    def _tag(metric):
        return (f"{metric}  {base_tag}" if agent_ft is None
                else f"{metric}  {base_tag} vs {ft_tag}")

    def _save(fig, name):
        p = out_dir / f"{filename}_{name}.png"
        fig.savefig(p, dpi=150)
        print(f"Saved → {p}")
        plt.close(fig)

    # Figure 1: stable rank
    ncols = 2 if agent_ft is not None else 1
    sr_vmin = min(float(sr_base.min()), float(sr_ft.min()))
    sr_vmax = max(float(sr_base.max()), float(sr_ft.max()))
    fig1, axes1 = plt.subplots(1, ncols,
                                figsize=(ncols * max(4, T_n * 0.6 + 1),
                                         max(3, T_a * 0.6 + 1)),
                                squeeze=False, constrained_layout=True)
    plot_stable_rank(
        sr_base,
        title=f"Stable rank  —  {base_tag}",
        ax=axes1[0, 0],
        vmin=sr_vmin,
        vmax=sr_vmax,
    )
    if agent_ft is not None:
        plot_stable_rank(
            sr_ft,
            title=f"Stable rank  —  {ft_tag}",
            ax=axes1[0, 1],
            vmin=sr_vmin,
            vmax=sr_vmax,
        )
    fig1.colorbar(axes1[0, 0].images[0], ax=axes1[0, 0], location="right", fraction=0.046, pad=0.02)
    fig1.suptitle(_tag("Stable rank"), fontsize=11)
    fig1.tight_layout()
    _save(fig1, "stable_rank")

    # Figure 2: spectrum entropy
    ent_vmin = min(float(entropy_base.min()), float(entropy_ft.min()))
    ent_vmax = max(float(entropy_base.max()), float(entropy_ft.max()))
    fig2, axes2 = plt.subplots(1, ncols,
                                figsize=(ncols * max(4, T_n * 0.6 + 1),
                                         max(3, T_a * 0.6 + 1)),
                                squeeze=False, constrained_layout=True)
    plot_spectrum_entropy(
        entropy_base,
        title=f"Spectrum entropy  —  {base_tag}",
        ax=axes2[0, 0],
        vmin=ent_vmin,
        vmax=ent_vmax,
    )
    if agent_ft is not None:
        plot_spectrum_entropy(
            entropy_ft,
            title=f"Spectrum entropy  —  {ft_tag}",
            ax=axes2[0, 1],
            vmin=ent_vmin,
            vmax=ent_vmax,
        )
    fig2.suptitle(_tag("Spectrum entropy"), fontsize=11)
    fig2.colorbar(axes2[0, 0].images[0], ax=axes2[0, 0], location="right", fraction=0.046, pad=0.02)
    fig2.tight_layout()
    _save(fig2, "spectrum_entropy")

    # Figure 3: normalized spectrum for a representative block
    rep_i = T_a // 4     # first quarter of action horizon
    rep_j = T_n // 2     # middle of noise horizon
    fig3, ax3 = plt.subplots(figsize=(max(4, int(mean_s_base.shape[-1]) * 0.5 + 1), 3))
    plot_normalized_spectra(
        mean_s_base, rep_i, rep_j,
        s_ft=mean_s_ft,
        label_base=base_tag,
        label_ft=ft_tag,
        ax=ax3,
    )
    _save(fig3, f"spectrum_block_{rep_i}_{rep_j}")

    # Figure 4: subspace alignment (only when two checkpoints)
    if agent_ft is not None:
        fig4, ax4 = plot_subspace_alignment(
            mean_cos,
            title=(f"Subspace alignment  (top-{top_k})  "
                   f"{base_tag} vs {ft_tag}"),
        )
        _save(fig4, "subspace_alignment")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jacobian SVD analysis for pi0 diffusion policy")
    parser.add_argument("--checkpoint_base", type=str, default="pi05_base",
                        help="Base checkpoint key (default: pi05_base)")
    parser.add_argument("--checkpoint_ft", type=str, default=None,
                        help="Finetuned checkpoint key for comparison (optional)")
    parser.add_argument("--noise_actor_dir", type=str, default=None,
                        help="Path to PixelSACLearner checkpoint dir (random if omitted)")
    parser.add_argument("--libero_suite", type=str, default="libero_90")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Task index within suite (all tasks if omitted)")
    parser.add_argument("--N", type=int, default=100,
                        help="Number of states to average over (default: 100)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of leading singular vectors for subspace alignment")
    parser.add_argument("--filename", type=str, default="jacobian_svd",
                        help="Output filename stem (default: jacobian_svd)")
    args = parser.parse_args()

    svd_analysis_test(
        checkpoint_base=args.checkpoint_base,
        checkpoint_ft=args.checkpoint_ft,
        noise_actor_dir=args.noise_actor_dir,
        libero_suite=args.libero_suite,
        task_id=args.task_id,
        N=args.N,
        seed=args.seed,
        top_k=args.top_k,
        filename=args.filename,
    )
