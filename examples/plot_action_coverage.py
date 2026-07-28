#!/usr/bin/env python
"""Plot action-space coverage from data collected by collect_action_coverage.py.

Produces five figures per mode (rollout / openloop):

  fig1_pca_<mode>.pdf        – PCA 2-D scatter + convex hull.  Shows WHERE
                               each model's actions live; hull area is a 2-D
                               proxy, NOT a width metric (use fig4/fig5 for that).

  fig2_std_<mode>.pdf        – Per-dimension standard deviation, grouped bar
                               chart.  X-axis labelled with joint/axis names.

  fig3_saturation_<mode>.pdf – Mean-std-across-dims vs. number of
                               trajectories / noise seeds.  Shows when
                               more data stops increasing coverage.

  fig4_eigenspectrum_<mode>.pdf – Eigenvalue spectrum of each model's action
                               covariance matrix (scree plot).  The model whose
                               curve lies higher is strictly wider.  Flat curves
                               mean variance is spread across many dimensions.
                               This is the primary width-comparison figure.

  fig5_coverage_metrics_<mode>.pdf – Three scalar metrics side-by-side:
                               total variance (trace of Σ), effective rank
                               (exp of eigenvalue entropy), and log-det of Σ
                               (log-volume of the covariance ellipsoid).

All figures are also saved as .png.

Usage (point at a directory created by collect_action_coverage.py):

  python -m examples.plot_action_coverage \\
      --data_dir logs/action_coverage/20240101_120000 \\
      --mode both

Or plot all subfolders in a parent:

  python -m examples.plot_action_coverage \\
      --data_dir logs/action_coverage \\
      --all_runs
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Dimension label maps
# ---------------------------------------------------------------------------

LIBERO_7D_LABELS = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]

ALOHA_14D_LABELS = [
    "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_j6", "L_grip",
    "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_j6", "R_grip",
]

DIM_LABELS = {
    7: LIBERO_7D_LABELS,
    14: ALOHA_14D_LABELS,
}


def get_dim_labels(action_dim: int) -> list[str]:
    if action_dim in DIM_LABELS:
        return DIM_LABELS[action_dim]
    return [f"dim_{i}" for i in range(action_dim)]


# ---------------------------------------------------------------------------
# Colour cycle
# ---------------------------------------------------------------------------

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(data_dir: pathlib.Path) -> dict:
    """Load meta.json and all .npy files from *data_dir*/data/."""
    meta_path = data_dir / "data" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {data_dir / 'data'}")
    meta = json.loads(meta_path.read_text())

    run = {"meta": meta, "checkpoints": {}}
    for ckpt_meta in meta["checkpoints"]:
        name = ckpt_meta["name"]
        entry = {"meta": ckpt_meta, "rollout": None, "openloop": None}
        if "rollout" in ckpt_meta:
            p = data_dir / ckpt_meta["rollout"]["file"]
            if p.exists():
                entry["rollout"] = np.load(str(p))
        if "openloop" in ckpt_meta:
            p = data_dir / ckpt_meta["openloop"]["file"]
            if p.exists():
                entry["openloop"] = np.load(str(p))
            H_common = min(entry["openloop"][0].shape[0], entry["openloop"][1].shape[0])  # e.g. 10
            chunk_common = entry["openloop"][:, :H_common, :]  # (K, H_common, action_dim)
            x_k = chunk_common.reshape(len(entry["openloop"]), -1) 
            entry["openloop"] = x_k
        run["checkpoints"][name] = entry

    return run


# ---------------------------------------------------------------------------
# Fig 1 – PCA scatter + convex hull
# ---------------------------------------------------------------------------

def _convex_hull_area(points_2d: np.ndarray) -> float:
    """Return convex hull area; returns 0 if fewer than 3 unique points."""
    if len(points_2d) < 3:
        return 0.0
    try:
        hull = ConvexHull(points_2d)
        return float(hull.volume)   # in 2-D, scipy uses 'volume' for area
    except Exception:
        return 0.0


def plot_pca_coverage(run: dict, mode: str, out_dir: pathlib.Path, libero_suite: str, task_id: int):
    """Fig 1: PCA 2-D scatter + convex hull."""
    checkpoints = run["checkpoints"]
    ckpt_names = list(checkpoints.keys())

    # Collect arrays for requested mode
    arrays = {}
    for name in ckpt_names:
        arr = checkpoints[name][mode]
        if arr is not None and len(arr) > 0:
            arrays[name] = arr

    if not arrays:
        print(f"  [fig1] No data for mode={mode}, skipping.", flush=True)
        return

    # Fit PCA on all data combined (shared projection = fair comparison)
    all_data = np.concatenate(list(arrays.values()), axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_data)
    var_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, arr) in enumerate(arrays.items()):
        pts = pca.transform(arr)  # (N, 2)
        color = _color(i)

        # Scatter (subsample for visibility if large)
        n_scatter = min(len(pts), 3000)
        idx = np.random.choice(len(pts), n_scatter, replace=False)
        ax.scatter(
            pts[idx, 0], pts[idx, 1],
            s=4, alpha=0.35, color=color, rasterized=True,
        )

        # Convex hull
        area = _convex_hull_area(pts)
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            poly = MplPolygon(
                hull_pts,
                closed=True,
                fill=True,
                facecolor=color,
                alpha=0.12,
                edgecolor=color,
                linewidth=1.5,
                label=f"{name}  (area={area:.3f})",
            )
            ax.add_patch(poly)
        except Exception:
            pass

        # Centroid marker
        ax.scatter(*pts.mean(axis=0), marker="x", s=80, color=color, linewidths=2)

    ax.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% var)", fontsize=11)
    mode_label = "Full rollouts" if mode == "rollout" else "Open-loop noise sweep"
    ax.set_title(f"Action-space coverage — {mode_label} {libero_suite} {task_id}", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    _save_fig(fig, out_dir / f"fig1_pca_{mode}")


# ---------------------------------------------------------------------------
# Fig 2 – Per-dimension std bar chart
# ---------------------------------------------------------------------------

def plot_std_per_dim(run: dict, mode: str, out_dir: pathlib.Path, libero_suite: str, task_id: int):
    """Fig 2: grouped bar chart of per-dimension standard deviation."""
    checkpoints = run["checkpoints"]
    ckpt_names = list(checkpoints.keys())

    arrays = {}
    for name in ckpt_names:
        arr = checkpoints[name][mode]
        if arr is not None and len(arr) > 0:
            arrays[name] = arr

    if not arrays:
        print(f"  [fig2] No data for mode={mode}, skipping.", flush=True)
        return

    # Determine action_dim (should be consistent across checkpoints)
    action_dim = next(iter(arrays.values())).shape[1]
    dim_labels = get_dim_labels(action_dim)
    n_ckpts = len(arrays)

    # Compute per-dim std
    stds = {name: np.std(arr, axis=0) for name, arr in arrays.items()}

    x = np.arange(action_dim)
    bar_width = 0.7 / max(n_ckpts, 1)
    offsets = np.linspace(-(n_ckpts - 1) / 2, (n_ckpts - 1) / 2, n_ckpts) * bar_width

    fig, ax = plt.subplots(figsize=(max(8, action_dim * 0.9), 5))
    for i, (name, std) in enumerate(stds.items()):
        ax.bar(
            x + offsets[i],
            std,
            width=bar_width,
            color=_color(i),
            label=name,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylabel("Standard deviation", fontsize=11)
    ax.set_xlabel("Action dimension", fontsize=11)
    mode_label = "Full rollouts" if mode == "rollout" else "Open-loop noise sweep"
    ax.set_title(f"Per-dimension action spread — {mode_label} {libero_suite} {task_id}", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xlim(-0.6, action_dim - 0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    _save_fig(fig, out_dir / f"fig2_std_{mode}")


# ---------------------------------------------------------------------------
# Fig 3 – Coverage saturation curve
# ---------------------------------------------------------------------------

def _cumulative_mean_std(arr: np.ndarray, chunk_size: int,
                          max_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean-std-across-dims after accumulating 1..N chunks.

    For rollout mode *arr* is already per-step; chunk_size=1 gives per-step.
    For openloop mode each chunk_size=action_horizon.

    Subsamples to at most *max_points* evaluations so the loop is fast
    even when total steps is large.

    Returns (x_ticks, mean_std_values).
    """
    n_total = len(arr)
    chunk = max(1, chunk_size)
    # All valid multiples of chunk_size up to n_total
    all_ends = np.arange(chunk, n_total + 1, chunk)
    if len(all_ends) == 0:
        return np.array([]), np.array([])
    # Subsample to at most max_points, always keeping the last point
    if len(all_ends) > max_points:
        indices = np.round(np.linspace(0, len(all_ends) - 1, max_points)).astype(int)
        indices = np.unique(indices)
        all_ends = all_ends[indices]

    xs, ys = [], []
    for end in all_ends:
        subset = arr[:end]
        ys.append(float(np.mean(np.std(subset, axis=0))))
        xs.append(int(end))
    return np.array(xs), np.array(ys)


def plot_saturation(run: dict, mode: str, out_dir: pathlib.Path, libero_suite: str, task_id: int):
    """Fig 3: coverage saturation — mean std across dims vs. samples seen."""
    checkpoints = run["checkpoints"]
    ckpt_names = list(checkpoints.keys())

    x_label = "Noise seeds" if mode == "openloop" else "Steps"
    fig, ax = plt.subplots(figsize=(7, 5))
    any_data = False

    for i, name in enumerate(ckpt_names):
        arr = checkpoints[name][mode]
        if arr is None or len(arr) == 0:
            continue

        ckpt_meta = checkpoints[name]["meta"]
        horizon = ckpt_meta.get("action_horizon", 1)
        chunk_size = horizon if mode == "openloop" else 1
        xs, ys = _cumulative_mean_std(arr, chunk_size)
        if len(xs) == 0:
            continue

        ax.plot(xs, ys, color=_color(i), label=name, linewidth=1.8)
        any_data = True

    if not any_data:
        print(f"  [fig3] No data for mode={mode}, skipping.", flush=True)
        plt.close(fig)
        return

    mode_label = "Full rollouts" if mode == "rollout" else "Open-loop noise sweep"
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Mean std across action dims", fontsize=11)
    ax.set_title(f"Coverage saturation — {mode_label} {libero_suite} {task_id}", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    _save_fig(fig, out_dir / f"fig3_saturation_{mode}")


# ---------------------------------------------------------------------------
# Coverage metrics helpers
# ---------------------------------------------------------------------------

def _coverage_metrics(arr: np.ndarray) -> dict:
    """Compute full-D-space coverage metrics for an action array (N, D).

    Returns a dict with:
      total_var    – trace(Σ) = sum of per-dim variances.  Overall scale.
      effective_rank – exp(H) where H is the Shannon entropy of the normalised
                       eigenvalue spectrum.  Max = D (all dims equally used).
      log_det      – sum of log(positive eigenvalues).  Log-volume of Σ ellipsoid.
      eigenvalues  – sorted descending eigenvalue array (length D).
    """
    if arr.shape[0] < 2:
        d = arr.shape[1]
        return {
            "total_var": 0.0,
            "effective_rank": 1.0,
            "log_det": float("-inf"),
            "eigenvalues": np.zeros(d),
        }

    cov = np.cov(arr.T)                         # (D, D)
    eigvals = np.linalg.eigvalsh(cov)           # ascending; real for symmetric
    eigvals = np.clip(eigvals, 0, None)         # numerical noise can give tiny negatives
    eigvals_desc = np.sort(eigvals)[::-1]       # descending for scree plot

    total_var = float(np.sum(eigvals))

    pos = eigvals[eigvals > 1e-12]
    if len(pos) > 0:
        p = pos / pos.sum()
        entropy = float(-np.sum(p * np.log(p)))
        effective_rank = float(np.exp(entropy))
        log_det = float(np.sum(np.log(pos)))
    else:
        effective_rank = 1.0
        log_det = float("-inf")

    return {
        "total_var": total_var,
        "effective_rank": effective_rank,
        "log_det": log_det,
        "eigenvalues": eigvals_desc,
    }


# ---------------------------------------------------------------------------
# Fig 4 – Eigenvalue spectrum (scree plot)
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(run: dict, mode: str, out_dir: pathlib.Path,
                             libero_suite: str, task_id: int):
    """Fig 4: scree plot — primary width-comparison figure.

    The model whose curve lies higher at every component is strictly wider.
    A flat curve means variance is spread evenly across many dimensions.
    """
    checkpoints = run["checkpoints"]
    ckpt_names = list(checkpoints.keys())

    arrays = {
        name: checkpoints[name][mode]
        for name in ckpt_names
        if checkpoints[name][mode] is not None and len(checkpoints[name][mode]) > 0
    }
    if not arrays:
        print(f"  [fig4] No data for mode={mode}, skipping.", flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_abs, ax_rel = axes

    for i, (name, arr) in enumerate(arrays.items()):
        m = _coverage_metrics(arr)
        eigvals = m["eigenvalues"]
        d = len(eigvals)
        xs = np.arange(1, d + 1)

        color = _color(i)
        ax_abs.plot(xs, eigvals, marker="o", markersize=4, color=color,
                    linewidth=1.8, label=name)
        ax_rel.plot(xs, np.cumsum(eigvals) / max(eigvals.sum(), 1e-12),
                    marker="o", markersize=4, color=color,
                    linewidth=1.8, label=name)

    mode_label = "Full rollouts" if mode == "rollout" else "Open-loop noise sweep"

    ax_abs.set_xlabel("Component index", fontsize=11)
    ax_abs.set_ylabel("Eigenvalue (variance explained)", fontsize=11)
    ax_abs.set_title(
        f"Eigenvalue spectrum — {mode_label}\n{libero_suite} task {task_id}",
        fontsize=11,
    )
    ax_abs.set_yscale("log")
    ax_abs.legend(fontsize=9, framealpha=0.85)
    ax_abs.set_xlim(0.5, None)

    ax_rel.set_xlabel("Component index", fontsize=11)
    ax_rel.set_ylabel("Cumulative variance explained", fontsize=11)
    ax_rel.set_title("Cumulative variance explained", fontsize=11)
    ax_rel.axhline(0.9, color="grey", linestyle="--", linewidth=1, label="90%")
    ax_rel.legend(fontsize=9, framealpha=0.85)
    ax_rel.set_xlim(0.5, None)
    ax_rel.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_fig(fig, out_dir / f"fig4_eigenspectrum_{mode}")


# ---------------------------------------------------------------------------
# Fig 5 – Scalar coverage metrics bar chart
# ---------------------------------------------------------------------------

def plot_coverage_metrics(run: dict, mode: str, out_dir: pathlib.Path,
                          libero_suite: str, task_id: int):
    """Fig 5: side-by-side bar chart of three full-D-space width metrics.

    All three are computed from the full covariance matrix — no projection loss.
    """
    checkpoints = run["checkpoints"]
    ckpt_names = list(checkpoints.keys())

    names, total_vars, eff_ranks, log_dets = [], [], [], []
    for name in ckpt_names:
        arr = checkpoints[name][mode]
        if arr is None or len(arr) == 0:
            continue
        m = _coverage_metrics(arr)
        names.append(name)
        total_vars.append(m["total_var"])
        eff_ranks.append(m["effective_rank"])
        log_dets.append(m["log_det"])

    if not names:
        print(f"  [fig5] No data for mode={mode}, skipping.", flush=True)
        return

    x = np.arange(len(names))
    colors = [_color(i) for i in range(len(names))]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    def _bar(ax, values, ylabel, title, fmt=".4f"):
        bars = ax.bar(x, values, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=8,
            )

    _bar(axes[0], total_vars, "trace(Σ)",
         "Total variance\n(wider = larger)", fmt=".4f")
    _bar(axes[1], eff_ranks, "exp(H[eigenvalues])",
         "Effective rank\n(max = action_dim)", fmt=".2f")
    _bar(axes[2], log_dets, "log|Σ|",
         "Log-det of covariance\n(wider = larger)", fmt=".2f")

    mode_label = "Full rollouts" if mode == "rollout" else "Open-loop noise sweep"
    fig.suptitle(
        f"Coverage metrics — {mode_label} — {libero_suite} task {task_id}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, out_dir / f"fig5_coverage_metrics_{mode}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, stem: pathlib.Path):
    for ext in ("pdf", "png"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_run(data_dir: pathlib.Path, modes: list[str], libero_suite: str, task_id: int, out_dir: pathlib.Path | None = None):
    """Generate all figures for one run directory."""
    run = load_run(data_dir)
    meta = run["meta"]

    if out_dir is None:
        out_dir = data_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Plotting: {data_dir}", flush=True)
    print(
        f"  Checkpoints: {[c['name'] for c in meta['checkpoints']]}",
        flush=True,
    )
    print(f"  Figures → {out_dir}", flush=True)

    for mode in modes:
        print(f"\n  -- mode: {mode} --", flush=True)
        plot_pca_coverage(run, mode, out_dir, libero_suite, task_id)
        plot_std_per_dim(run, mode, out_dir, libero_suite, task_id)
        plot_saturation(run, mode, out_dir, libero_suite, task_id)
        plot_eigenvalue_spectrum(run, mode, out_dir, libero_suite, task_id)
        plot_coverage_metrics(run, mode, out_dir, libero_suite, task_id)


def main(argv=None):
    args = parse_args(argv)

    data_root = pathlib.Path(args.data_dir)
    libero_suite = args.libero_suite
    task_id = args.task_id
    # Determine which modes to plot
    if args.mode == "both":
        modes = ["rollout", "openloop"]
    else:
        modes = [args.mode]

    if args.all_runs:
        # Iterate over all timestamped subdirs
        run_dirs = sorted(
            p for p in data_root.iterdir()
            if p.is_dir() and (p / "data" / "meta.json").exists()
        )
        if not run_dirs:
            print(f"No run directories found under {data_root}", flush=True)
            sys.exit(1)
        for run_dir in run_dirs:
            try:
                plot_run(run_dir, modes, libero_suite, task_id)
            except Exception as exc:
                print(f"  [ERROR] {run_dir}: {exc}", flush=True)
    else:
        # Single run
        if not (data_root / "data" / "meta.json").exists():
            # Maybe data_root IS the data/ folder
            candidate = data_root.parent
            if (candidate / "data" / "meta.json").exists():
                data_root = candidate
            else:
                raise FileNotFoundError(
                    f"Cannot find data/meta.json under {data_root}. "
                    "Use --data_dir to point at the timestamped run directory."
                )
        out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
        plot_run(data_root, modes, libero_suite, task_id, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot action-space coverage from collect_action_coverage output."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the timestamped run directory created by collect_action_coverage.py",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["rollout", "openloop", "both"],
        help="Which collection mode's data to plot (default: both)",
    )
    parser.add_argument(
        "--all_runs",
        action="store_true",
        help=(
            "Iterate over all timestamped subdirectories inside --data_dir "
            "and generate figures for each."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override figure output directory (default: <data_dir>/figs/)",
    )
    parser.add_argument(
        "--libero_suite",
        required=True,
        help="Libero suite",
    )
    parser.add_argument(
        "--task_id",
        required=True,
        help="Task ID",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
