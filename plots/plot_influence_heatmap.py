#!/usr/bin/env python3
"""Combined multi-task heatmap of the per-latent-dimension influence metric.

Reads one or more ``*_metrics.json`` files produced by ``jaxrl2/tests/jacobian.py``
(each holding a (typically 32-element) "influence" vector, plus "checkpoint",
"libero_suite", and "task_id") and renders a single figure with one column per
task and one row per latent dimension (0..D-1) on the y-axis. Columns are
separated by a visible gap; rows within a column are drawn back-to-back with
no gap. Each column is headed by its task id.

Usage
-----
python plots/plot_influence_heatmap.py \\
    --metrics_json plots/plots/jacobians/*_metrics.json \\
    --out_dir plots/plots/jacobians \\
    --filename influence_heatmap_all_tasks
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "mathtext.fontset": "cm",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(path: pathlib.Path) -> dict:
    """Load a jacobian ``*_metrics.json`` file and sanity-check its "influence" field.

    Args:
        path : path to a metrics JSON file (as saved by jaxrl2/tests/jacobian.py)

    Returns:
        metrics : the parsed JSON dict
    """
    with open(path) as f:
        metrics = json.load(f)

    influence = metrics.get("influence")
    if influence is None:
        raise ValueError(f"{path}: no 'influence' field found in metrics JSON")
    if len(influence) != 32:
        print(
            f"Warning: {path}: expected 32 influence values, found {len(influence)}. "
            "Plotting whatever is present.",
            flush=True,
        )
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_influence_heatmap(
    tasks,
    title=None,
    cmap="viridis",
    log_scale=True,
    gap=0.4,
):
    """Combined heatmap of the per-latent-dimension influence metric, one column per task.

    Each task is drawn as its own column of stacked colored cells (one cell per
    latent dimension, back-to-back with no vertical gap). Columns are separated
    by a visible horizontal gap, and each column is headed by its task id, so
    influence patterns can be compared side-by-side across tasks.

    Args:
        tasks        : list of metrics dicts (as returned by load_metrics), each
                       expected to have "influence", and optionally "checkpoint",
                       "libero_suite", "task_id"
        title        : overall figure title; auto-generated from libero_suite /
                       checkpoint if None (and if consistent across tasks)
        cmap         : matplotlib colormap name
        log_scale    : if True, color-scale the cells on a log (rather than linear)
                       norm, since influence values typically span many orders of
                       magnitude
        gap          : gap (in cell-widths) left blank between consecutive task
                       columns (0 = no gap, touching columns). There is no gap
                       between rows within a column.

    Returns:
        fig, ax
    """
    if not tasks:
        raise ValueError("plot_influence_heatmap: no tasks provided")

    # Sort by task_id (left to right) when every task has one.
    if all(t.get("task_id") is not None for t in tasks):
        tasks = sorted(tasks, key=lambda t: t["task_id"])

    D = min(len(t["influence"]) for t in tasks)
    N = len(tasks)
    # (D, N) matrix: row d = latent dim, column t = task
    influence_matrix = np.stack(
        [np.asarray(t["influence"], dtype=float)[:D] for t in tasks], axis=1
    )

    all_values = influence_matrix.flatten()
    if log_scale:
        positive = all_values[all_values > 0]
        vmin = float(positive.min()) if positive.size else 1e-12
        vmax = float(all_values.max()) if all_values.size else 1.0
        vmax = max(vmax, vmin * 10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=float(all_values.min()), vmax=float(all_values.max()))

    cmap_obj = plt.get_cmap(cmap)
    mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(all_values)

    cell_size = 1.0
    step = cell_size + gap  # horizontal spacing between task columns
    header_room = 2.0       # vertical space reserved above row 0 for column headers

    fig_width = max(5.0, 1.0 * N + 2.0)
    fig_height = max(5.0, 0.28 * D + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for t_idx, task in enumerate(tasks):
        x0 = t_idx * step

        for d in range(D):
            value = float(influence_matrix[d, t_idx])
            y_top = -d
            color_value = value if (not log_scale or value > 0) else norm.vmin
            color = mappable.to_rgba(color_value)
            ax.add_patch(Rectangle(
                (x0, y_top - cell_size), cell_size, cell_size,
                facecolor=color, edgecolor="none",
            ))
            r, g, b, _ = color
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            # ax.text(
            #     x0 + cell_size / 2, y_top - cell_size / 2,
            #     f"{value:.1e}",
            #     ha="center", va="center", fontsize=6, color=text_color,
            # )

        task_id = task.get("task_id")
        header = f"Task {task_id}" if task_id is not None else f"Task {t_idx}"
        ax.text(
            x0 + cell_size / 2, cell_size + 0.15, header,
            ha="center", va="bottom", fontsize=13, 
            fontweight="bold",
        )

    ax.set_xlim(-gap / 2, N * step - gap / 2)
    ax.set_ylim(-(D - 1) - cell_size - 0.2, cell_size + header_room)
    ax.set_xticks([])
    ax.set_yticks([-d - cell_size / 2 for d in range(D)])
    ax.set_yticklabels([str(d) for d in range(D)], fontsize=8)
    ax.set_ylabel("Latent dimension  d", fontsize=11)

    if title is None:
        suites = {t.get("libero_suite") for t in tasks if t.get("libero_suite")}
        checkpoints = {t.get("checkpoint") for t in tasks if t.get("checkpoint")}
        parts = []
        if len(suites) == 1:
            parts.append(next(iter(suites)))
        if len(checkpoints) == 1:
            parts.append(next(iter(checkpoints)))
        title = "Influence metric  " + "  ".join(parts) if parts else "Influence metric across tasks"
    ax.set_title(title, fontsize=18, pad=15)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(mappable, ax=ax, fraction=0.08, pad=0.08 / max(N, 1) + 0.02)
    cbar.set_label(r"$\mathrm{Infl}_d(w)$", fontsize=11)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combined multi-task heatmap of the per-latent-dimension influence "
            "metric, read from one or more jacobian *_metrics.json files. "
            "Produces a single figure with one column per task."
        )
    )
    parser.add_argument(
        "--metrics_json", type=str, nargs="+", required=True,
        help=(
            "One or more paths to jacobian *_metrics.json files "
            "(as saved by jaxrl2/tests/jacobian.py), one per task. Shell globs "
            "work too, e.g. 'plots/plots/jacobians/*_metrics.json'."
        ),
    )
    parser.add_argument(
        "--out_dir", type=str, default="plots/plots/jacobians",
        help="Directory to save the output SVG (default: plots/plots/jacobians).",
    )
    parser.add_argument(
        "--filename", type=str, default="influence_heatmap_all_tasks",
        help="Output filename stem (default: influence_heatmap_all_tasks).",
    )
    parser.add_argument(
        "--cmap", type=str, default="viridis",
        help="Matplotlib colormap name (default: viridis).",
    )
    parser.add_argument(
        "--log_scale", type=int, default=1,
        help=(
            "If 1, color-scale the cells on a log norm, which is usually needed "
            "since influence values span many orders of magnitude (default: 1)."
        ),
    )
    parser.add_argument(
        "--gap", type=float, default=0.4,
        help=(
            "Gap (in cell-widths) left blank between consecutive task columns "
            "(default: 0.4). There is no gap between rows within a column."
        ),
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Overall figure title (default: None, auto-generated from libero_suite / checkpoint if consistent across tasks).",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for json_path_str in args.metrics_json:
        json_path = pathlib.Path(json_path_str)
        print(f"Loading {json_path}...", flush=True)
        tasks.append(load_metrics(json_path))

    fig, ax = plot_influence_heatmap(
        tasks,
        title=args.title,
        cmap=args.cmap,
        log_scale=bool(args.log_scale),
        gap=args.gap,
    )

    out_path = out_dir / f"{args.filename}.svg"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}", flush=True)


if __name__ == "__main__":
    main()
