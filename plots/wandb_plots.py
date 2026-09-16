#!/usr/bin/env python3
"""Plot W&B success rates for selected runs."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from plot_fonts import PLOT_FONT_FAMILY
from plot_layout import (
    AXES_H,
    AXES_W,
    BOTTOM_PAD,
    LEFT_PAD,
    SUBPLOT_W,
    XLABEL_H,
    add_top_decorations,
    apply_axes_grid,
    figure_size,
)

DEFAULT_PROJECT = "DSRL_pi0_Libero"
DEFAULT_METRIC = "evaluation/success_rate"
DEFAULT_X_AXIS = "_step"
DEFAULT_EMA_HALFLIFE = 50_000

# Publication-quality styling
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# COLOR_MAP = {
#     "DSRL-SAC (Baseline)": "#D55E00",  # orange 
#     "RDS-NA": "#009E73",          # blue
# }

COLOR_MAP = {
    "DSRL-SAC (Baseline)": "#D55E00",  # orange 
    "FDTS-Chunk MLP Actor + MLP Critic": "#00bed5",          # blue
    "FDTS-Chunk MLP Actor + MLP Critic (hidden dim 2048)": "#CC79A7",          # blue
    "FDTS-Chunk MLP Actor + Transformer Critic": "#56B4E9",    # sky blue
    # "FDTS-Chunk MLP Actor + Transformer Critic (Autoregressive)": "#009E73", # bluish green     
}

def setup_plot_style():
    sns.set_theme(
        style="ticks",
        context="paper",
    )

    plt.rcParams.update({
        "font.family": PLOT_FONT_FAMILY,
        "figure.titlesize": 32,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 22,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.2,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
    })


def format_training_steps(value: float, _pos: int) -> str:
    """Format step counts as plain numbers, K, or M labels."""
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:g}M"
    if abs_value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:g}K"
    return f"{int(value)}"


def nice_step_ticks(max_step: float) -> list[float]:
    if max_step <= 0:
        return [0.0]
    nice_units = [
        50_000, 100_000, 200_000, 250_000, 400_000, 500_000, 600_000, 750_000, 800_000,
        1_000_000, 2_000_000, 2_500_000, 5_000_000,
    ]
    target_ticks = 5
    best_ticks = [0.0, max_step]
    for unit in nice_units:
        ticks = [float(i * unit) for i in range(int(max_step / unit) + 2) if i * unit <= max_step]
        if target_ticks <= len(ticks) <= target_ticks + 2:
            return ticks
        if len(ticks) >= 4 and abs(len(ticks) - target_ticks) < abs(len(best_ticks) - target_ticks):
            best_ticks = ticks
    return best_ticks


def time_weighted_ema(values: np.ndarray, steps: np.ndarray, halflife: float) -> np.ndarray:
    """Smooth a series with an EMA that respects irregular step spacing."""
    if len(values) == 0 or halflife <= 0:
        return values

    tau = halflife / math.log(2)
    smoothed = np.empty(len(values), dtype=float)
    smoothed[0] = float(values[0])
    for i in range(1, len(values)):
        dt = max(float(steps[i] - steps[i - 1]), 0.0)
        alpha = 1.0 - math.exp(-dt / tau)
        smoothed[i] = alpha * float(values[i]) + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def apply_smoothing(df: pd.DataFrame, halflife: float) -> pd.DataFrame:
    if halflife <= 0:
        return df

    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("label", sort=False):
        group = group.sort_values("step").copy()
        group["success_rate"] = time_weighted_ema(
            group["success_rate"].to_numpy(),
            group["step"].to_numpy(),
            halflife,
        )
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def configure_wandb_auth(entity: str | None) -> str | None:
    """Load credentials from jaxrl2.utils.wandb_config when available."""
    try:
        from jaxrl2.utils.wandb_config import get_wandb_config

        wandb_config = get_wandb_config()
        os.environ.setdefault("WANDB_API_KEY", wandb_config["WANDB_API_KEY"])
        os.environ.setdefault("WANDB_USER_EMAIL", wandb_config["WANDB_EMAIL"])
        os.environ.setdefault("WANDB_USERNAME", wandb_config["WANDB_USERNAME"])
        if entity is None and wandb_config.get("WANDB_TEAM"):
            entity = wandb_config["WANDB_TEAM"]
    except ImportError:
        pass
    return entity or os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_TEAM")


def parse_label_mapping(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(
                f"Invalid label mapping {item!r}. Expected format: run_id=Label,run_id2=Label2"
            )
        run_id, label = item.split("=", 1)
        mapping[run_id.strip()] = label.strip()
    return mapping


def metric_ylabel(metric: str) -> str:
    """Derive a y-axis label from a W&B metric key (e.g. evaluation/success_rate -> Success Rate)."""
    name = metric.rsplit("/", 1)[-1]
    return name.replace("_", " ").title()


def short_label(identifier: str) -> str:
    """Use the suffix after the timestamp when the run id follows project naming."""
    match = re.search(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{4}--s-\d+(?:_(.+))?$", identifier)
    if match and match.group(1):
        return match.group(1)
    if len(identifier) > 48:
        return identifier[:24] + "..." + identifier[-20:]
    return identifier


def fetch_run_history(
    *,
    entity: str | None,
    project: str,
    identifier: str,
    metric: str,
    x_axis: str,
) -> pd.DataFrame:
    """Fetch scalar history for one run from W&B."""
    api = wandb.Api()
    run_path = f"{entity}/{project}/{identifier}" if entity else f"{project}/{identifier}"
    try:
        run = api.run(run_path)
    except wandb.errors.CommError as exc:
        try:
            entity = "noragorhan-karlsruhe-institute-of-technology"
            project = "DSRL_pi0_Libero"
            run_path = f"{entity}/{project}/{identifier}" if entity else f"{project}/{identifier}"
            run = api.run(run_path)
        except wandb.errors.CommError as exc2:
            entity = "noras-masterarbeit"
            project = "DSRL_pi05_Libero"
            run_path = f"{entity}/{project}/{identifier}" if entity else f"{project}/{identifier}"
            run = api.run(run_path)
        except wandb.errors.CommError as exc2:
            raise ValueError(f"Could not find run {identifier!r} in {project!r}.") from exc2

    columns = [x_axis, metric]
    history = run.history(keys=columns, pandas=True)
    if history.empty:
        raise ValueError(f"Run {identifier!r} has no logged values for {metric!r}.")

    history = history.dropna(subset=[metric])
    if history.empty:
        raise ValueError(f"Run {identifier!r} only contains NaN values for {metric!r}.")

    history = history.rename(columns={metric: "success_rate", x_axis: "step"})
    history["identifier"] = identifier
    history["run_name"] = run.name or identifier
    return history


def load_local_run_history(
    run_dir: Path,
    metric: str,
    x_axis: str,
) -> pd.DataFrame:
    """Load scalar history from a local offline W&B run directory."""
    run = wandb.Api().from_path(str(run_dir))
    identifier = run.id or run.name or run_dir.name
    columns = [x_axis, metric]
    history = run.history(keys=columns, pandas=True)
    if history.empty:
        raise ValueError(f"Local run {run_dir} has no logged values for {metric!r}.")

    history = history.dropna(subset=[metric])
    history = history.rename(columns={metric: "success_rate", x_axis: "step"})
    history["identifier"] = identifier
    history["run_name"] = run.name or identifier
    return history


def collect_histories(
    identifiers: Iterable[str],
    *,
    entity: str | None,
    project: str,
    metric: str,
    x_axis: str,
    label_mapping: dict[str, str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for identifier in identifiers:
        path = Path(identifier)
        if path.exists():
            history = load_local_run_history(path, metric, x_axis)
            key = history["identifier"].iloc[0]
        else:
            history = fetch_run_history(
                entity=entity,
                project=project,
                identifier=identifier,
                metric=metric,
                x_axis=x_axis,
            )
            key = identifier

        label = label_mapping.get(key) or label_mapping.get(identifier) or short_label(key)
        history["label"] = label
        frames.append(history)

    return pd.concat(frames, ignore_index=True)


def clip_to_shortest_run(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only steps up to where the shortest run ends."""
    max_step = df.groupby("label", sort=False)["step"].max().min()
    return df.loc[df["step"] <= max_step].copy()


def plot_success_rates(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str | None,
    output: Path | None,
    show: bool,
    ylim: tuple[float, float] | None,
    ema_halflife: float,
    clip_to_shortest: bool,
    max_steps: float | None = None,
    sft_baseline: float | None = None,
) -> None:
    if clip_to_shortest:
        df = clip_to_shortest_run(df)
    if max_steps is not None:
        df = df.loc[df["step"] <= max_steps]
    df = apply_smoothing(df, ema_halflife)

    setup_plot_style()
    hue_order = df["label"].drop_duplicates().tolist()
    palette = {
        lbl: COLOR_MAP.get(lbl, OKABE_ITO[i % len(OKABE_ITO)])
        for i, lbl in enumerate(hue_order)
    }

    ymin_val = 0.0 if ylim is None else ylim[0]
    ymax_val = 1.0 if ylim is None else ylim[1]
    bottom_pad = 0.02 if ymin_val <= 0.0 else 0.0
    top_pad = 0.02 if ymax_val >= 1.0 else 0.0
    y_bottom = ymin_val - bottom_pad
    y_top = ymax_val + top_pad

    n_rows, n_cols = 1, 1
    n_fig_cols = max(n_cols, 2)
    n_legend_labels = len(hue_order) + (1 if sft_baseline is not None else 0)
    n_legend_cols = min(n_legend_labels, 2)
    fig_w, fig_h, suptitle_h, legend_h = figure_size(
        n_rows,
        n_fig_cols,
        suptitle=bool(title),
        n_legend_labels=n_legend_labels,
        n_legend_cols=n_legend_cols,
    )
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=True)
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)
    ax = axes[0, 0]

    sns.lineplot(
        data=df,
        x="step",
        y="success_rate",
        hue="label",
        hue_order=hue_order,
        palette=palette,
        linewidth=2.5,
        ax=ax,
        legend=False,
    )

    n_lines = len(ax.lines)
    for i, line in enumerate(ax.lines):
        line.set_zorder(3 + n_lines - i)
        line.set_clip_on(False)

    if sft_baseline is not None:
        ax.axhline(y=sft_baseline, color="0.4", linestyle="--", linewidth=2.0, zorder=2)

    # Use the requested cap (if any) as the axis ceiling rather than the
    # actual last sampled step. wandb's history() sub-samples points, so
    # the last row after filtering to `max_steps` can land slightly below
    # it (e.g. 799000 instead of 800000), which would otherwise cause
    # nice_step_ticks to silently drop the tick at `max_steps`.
    axis_max = float(max_steps) if max_steps is not None else float(df["step"].max())
    ax.set_xlim(0, axis_max)
    ax.set_xticks(nice_step_ticks(axis_max))
    ax.xaxis.set_major_formatter(FuncFormatter(format_training_steps))
    ax.set_ylim(y_bottom, y_top)

    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel(metric_ylabel(metric))
    ax.tick_params(direction="out", width=1.2, length=4)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="0.90", linewidth=0.8)
    sns.despine(ax=ax)

    legend_handles = [
        Line2D([0], [0], color=palette[lbl], linewidth=2.5, label=lbl)
        for lbl in hue_order
    ]
    if sft_baseline is not None:
        legend_handles.insert(
            0,
            Line2D([0], [0], color="0.4", linewidth=2.0, linestyle="--", label="$\pi_{0.5}$"),
        )

    n_legend_cols = min(len(legend_handles), 2)
    apply_axes_grid(axes, n_rows=n_rows, n_cols=n_cols, fig_w=fig_w, fig_h=fig_h)
    if n_cols < n_fig_cols:
        cell_left = (fig_w - n_cols * SUBPLOT_W) / 2
        for col in range(n_cols):
            left = (cell_left + col * SUBPLOT_W + LEFT_PAD) / fig_w
            axes[0, col].set_position(
                [left, (BOTTOM_PAD + XLABEL_H) / fig_h, AXES_W / fig_w, AXES_H / fig_h]
            )
    add_top_decorations(
        fig,
        suptitle=title,
        legend_handles=legend_handles,
        fig_h=fig_h,
        suptitle_h=suptitle_h,
        legend_h=legend_h,
        n_legend_cols=n_legend_cols,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        print(f"Saved plot to {output}")

    if show or output is None:
        plt.show()
    else:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot evaluation success rates from selected W&B runs. "
            "Identifiers can be run ids (experiment ids), or local offline run directories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python wandb_plots.py \\\n"
            "    --identifiers dsrl_pi05_libero_90_task28_2026_06_30_13_50_48_0000--s-0_criticgpt \\\n"
            "                  dsrl_pi05_libero_90_task28_2026_06_30_14_00_00_0000--s-0_baseline\n\n"
            "  python wandb_plots.py \\\n"
            "    --project DSRL_pi0_Libero \\\n"
            "    --identifiers run_a run_b \\\n"
            "    --labels run_a=Trafo actor,run_b=MLP actor \\\n"
            "    --output plots/task28_success.png\n"
        ),
    )
    parser.add_argument(
        "--identifiers",
        nargs="+",
        required=True,
        help="W&B run ids (experiment ids) or local offline run directories.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"W&B project name (default: {DEFAULT_PROJECT}).",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity/team. Defaults to WANDB_TEAM from jaxrl2.utils.wandb_config.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric key to plot (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--x-axis",
        default=DEFAULT_X_AXIS,
        help=f"X-axis column from W&B history (default: {DEFAULT_X_AXIS}).",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional legend labels as comma-separated run_id=Label pairs.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the plot (e.g. plots/success_rates.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Minimum y-axis value (default: 0.0).",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
        help="Maximum y-axis value (default: 1.0).",
    )
    parser.add_argument(
        "--ema-halflife",
        type=float,
        default=DEFAULT_EMA_HALFLIFE,
        help=(
            "Halflife in training steps for time-weighted EMA smoothing "
            f"(default: {DEFAULT_EMA_HALFLIFE}). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--clip-to-shortest-run",
        action="store_true",
        help=(
            "Truncate all runs and the x-axis at the last step of the shortest run "
            "(useful for fair comparisons when runs differ in length)."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=float,
        default=None,
        help="Maximum number of steps to plot (default: None).",
    )
    parser.add_argument(
        "--sft-baselines",
        nargs="+",
        default=None,
        help=(
            "Optional SFT baseline success-rate value to draw as a horizontal line. "
            "Only the first value is used for single-plot mode."
        ),
    )
    return parser


def parse_sft_baseline(raw: list[str] | None) -> float | None:
    if raw is None:
        return None
    value = raw[0].strip()
    if value == "":
        return None
    return float(value)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    entity = configure_wandb_auth(args.entity)
    label_mapping = parse_label_mapping(args.labels)

    try:
        sft_baseline = parse_sft_baseline(args.sft_baselines)
    except ValueError as exc:
        print(f"Error: invalid --sft-baselines value: {exc}", file=sys.stderr)
        return 1

    try:
        df = collect_histories(
            args.identifiers,
            entity=entity,
            project=args.project,
            metric=args.metric,
            x_axis=args.x_axis,
            label_mapping=label_mapping,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    plot_success_rates(
        df,
        metric=args.metric,
        title=args.title,
        output=args.output,
        show=args.show,
        ylim=(args.ymin, args.ymax),
        ema_halflife=args.ema_halflife,
        clip_to_shortest=args.clip_to_shortest_run,
        max_steps=args.max_steps,
        sft_baseline=sft_baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
