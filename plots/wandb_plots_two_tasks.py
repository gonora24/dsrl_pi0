#!/usr/bin/env python3
"""Plot W&B success rates for 4 tasks in a 2x2 subplot figure with a shared legend."""

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
from matplotlib.ticker import FuncFormatter

from matplotlib import font_manager

font_dir = Path(os.environ["CONDA_PREFIX"]) / "fonts"

for font in font_manager.findSystemFonts(fontpaths=[str(font_dir)]):
    font_manager.fontManager.addfont(font)

DEFAULT_PROJECT = "DSRL_pi0_Libero"
DEFAULT_METRIC = "evaluation/success_rate"
DEFAULT_X_AXIS = "_step"
DEFAULT_EMA_HALFLIFE = 50_000
DEFAULT_RUNS_PER_TASK = 4

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

METHOD_COLORS = [
    "#D55E00",  # Baseline
    "#00bed5",  # Method 2
    # "#d50053", # only for ablation overlapping
    "#0041d6",  # Method 3
    "#003756",  # Method 4  005889
]

DIM_COLORS = [
    # "#D55E00",
    "#5e00d5",  # rds
    # "#d50077",  # t-rds
    "#005889",  # gt-rds
    # "#e691a6",  # t-rds residual noise 
    # "#560030",  # t-rds residual mean
    "#003d35",  # gt-rds residual noise
    "#00d599",  # gt-rds residual mlp
]


def setup_plot_style():
    sns.set_theme(
        style="ticks",
        context="paper",
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.titlesize": 30,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 20,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.2,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
    })


def format_training_steps(value: float, _pos: int) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{int(value)}"


def nice_step_ticks(max_step: float) -> list[float]:
    if max_step <= 0:
        return [0.0]

    # Candidate divisors for ~5 ticks
    divisors = [3, 4, 5]

    nice_units = [
        50_000, 100_000, 150_000, 200_000, 250_000,
        300_000, 400_000, 500_000,
        1_000_000, 2_000_000, 2_500_000, 5_000_000,
    ]

    # Add divisions of the maximum itself
    for d in divisors:
        unit = max_step / d
        nice_units.append(round(unit / 50_000) * 50_000)

    nice_units = sorted(set(nice_units))

    target_ticks = 5
    best_ticks = None
    best_score = float("inf")

    for unit in nice_units:
        ticks = [i * unit for i in range(int(max_step // unit) + 1)]
        if ticks[-1] != max_step:
            ticks.append(max_step)

        score = abs(len(ticks) - target_ticks)
        if score < best_score:
            best_score = score
            best_ticks = ticks

    return best_ticks


def time_weighted_ema(values: np.ndarray, steps: np.ndarray, halflife: float) -> np.ndarray:
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
    # Group by individual run (identifier) within each label so that seeds are
    # smoothed independently before seaborn averages them across runs.
    for _, group in df.groupby(["label", "identifier"], sort=False):
        group = group.sort_values("step").copy()
        group["success_rate"] = time_weighted_ema(
            group["success_rate"].to_numpy(),
            group["step"].to_numpy(),
            halflife,
        )
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def configure_wandb_auth(entity: str | None) -> str | None:
    try:
        from jaxrl2.utils.wandb_config import get_wandb_config
        # wandb_config = get_wandb_config()
        # os.environ.setdefault("WANDB_API_KEY", wandb_config["WANDB_API_KEY"])
        # os.environ.setdefault("WANDB_USER_EMAIL", wandb_config["WANDB_EMAIL"])
        # os.environ.setdefault("WANDB_USERNAME", wandb_config["WANDB_USERNAME"])
        os.environ.setdefault("WANDB_TEAM", entity)
        # if entity is None and wandb_config.get("WANDB_TEAM"):
        #     entity = wandb_config["WANDB_TEAM"]
    except ImportError:
        pass
    return entity


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
    name = metric.rsplit("/", 1)[-1]
    return name.replace("_", " ").title()


def short_label(identifier: str) -> str:
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
    api = wandb.Api()
    run_path = f"{entity}/{project}/{identifier}" if entity else f"{project}/{identifier}"
    try:
        run = api.run(run_path)
    except wandb.errors.CommError as exc:
        raise ValueError(f"Could not find run {identifier!r} in {project!r}.") from exc
    if metric not in run.summary:
        metric2 = "evaluation/Reward >= 1"
        columns = [x_axis, metric2]
    else:
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


def load_local_run_history(run_dir: Path, metric: str, x_axis: str) -> pd.DataFrame:
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
        if identifier == "":
            continue
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


def clip_df_to_shortest_run(df: pd.DataFrame) -> pd.DataFrame:
    max_step = df.groupby("label", sort=False)["step"].max().min()
    return df.loc[df["step"] <= max_step].copy()


def plot_multi_task(
    task_dfs: list[pd.DataFrame],
    task_titles: list[str],
    *,
    metric: str,
    suptitle: str | None,
    output: Path | None,
    show: bool,
    ylim: tuple[float, float] | None,
    ema_halflife: float,
    clip_to_shortest: bool,
    errorbar: str = "ci",
    max_steps: float | None = None,
    dim_colors: int = 0,
    sft_baselines: list[float | None] | None = None,
) -> None:
    setup_plot_style()

    # Process each task dataframe
    processed: list[pd.DataFrame] = []
    for df in task_dfs:
        if clip_to_shortest:
            df = clip_df_to_shortest_run(df)
        if max_steps is not None:
            df = df.loc[df["step"] <= max_steps]
        df = apply_smoothing(df, ema_halflife)
        processed.append(df)

    # Determine consistent hue order and palette from across all tasks
    all_labels: list[str] = []
    for df in processed:
        for lbl in df["label"].drop_duplicates().tolist():
            if lbl not in all_labels:
                all_labels.append(lbl)
    if dim_colors == 1:
        print("Using dimension-specific colors")
        task_palette = {lbl: DIM_COLORS[i] for i, lbl in enumerate(all_labels)}
    else:
        print("Using method-specific colors")
        task_palette = {lbl: METHOD_COLORS[i] for i, lbl in enumerate(all_labels)}
    # Build palette: use COLOR_MAP when label is known, fall back to OKABE_ITO
    palette: dict[str, str] = {}
    fallback_idx = 0
    for lbl in all_labels:
        if lbl in task_palette:
            palette[lbl] = task_palette[lbl]
        else:
            palette[lbl] = OKABE_ITO[fallback_idx % len(OKABE_ITO)]
            fallback_idx += 1

    ymin_val = 0.0 if ylim is None else ylim[0]
    ymax_val = 1.0 if ylim is None else ylim[1]
    bottom_pad = 0.02 if ymin_val <= 0.0 else 0.0
    top_pad = 0.02 if ymax_val >= 1.0 else 0.0
    y_bottom = ymin_val - bottom_pad
    y_top = ymax_val + top_pad

    n_tasks = len(processed)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 7), sharey=True)
    axes = np.atleast_1d(axes)

    for idx, (df, title) in enumerate(zip(processed, task_titles)):
        ax = axes[idx]

        # Only use labels that actually appear in this task's data
        task_labels = df["label"].drop_duplicates().tolist()
        task_hue_order = [lbl for lbl in all_labels if lbl in task_labels]
        task_palette = {lbl: palette[lbl] for lbl in task_hue_order}

        sns.lineplot(
            data=df,
            x="step",
            y="success_rate",
            hue="label",
            hue_order=task_hue_order,
            palette=task_palette,
            linewidth=2.5,
            ax=ax,
            legend=False,
            errorbar=None if errorbar == "none" else errorbar,
        )

        # Assign z-order/clipping to the method lines before adding the SFT
        # baseline line, so the loop below doesn't clobber its explicit z-order.
        n_lines = len(ax.lines)
        for i, line in enumerate(ax.lines):
            line.set_zorder(3 + n_lines - i)
            line.set_clip_on(False)

        sft_value = sft_baselines[idx] if sft_baselines is not None else None
        if sft_value is not None:
            ax.axhline(y=sft_value, color="0.4", linestyle="--", linewidth=2.0, zorder=2)

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

        ax.set_title(title, pad=6)
        ax.set_xlabel("Training Steps")
        # Only left column gets the y-axis label
        ax.set_ylabel(metric_ylabel(metric) if idx % 2 == 0 else "")

        ax.tick_params(direction="out", width=1.2, length=4)
        ax.set_axisbelow(True)
        ax.grid(which="major", color="0.90", linewidth=0.8)
        sns.despine(ax=ax)

    # Shared figure title
    if suptitle:
        fig.suptitle(suptitle, y=1.01)

    # Shared legend below the 2x2 grid using proxy artists so it is independent
    # of any individual axes legend state.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=palette[lbl], linewidth=2.5, label=lbl)
        for lbl in all_labels
    ]

    has_sft = sft_baselines is not None and any(v is not None for v in sft_baselines)
    if has_sft:
        legend_handles.insert(0,
            Line2D([0], [0], color="0.4", linewidth=2.0, linestyle="--", label="$\pi_0$")
        )

    n_legend_cols = min(len(legend_handles), 1)
    fig.tight_layout(rect=[0, 0.35, 1, 1])
    # fig.tight_layout(rect=[0, 0.23, 1, 1])
    # fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=n_legend_cols,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="0.6",
        facecolor="white",
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved plot to {output}")

    if show or output is None:
        plt.show()
    else:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot evaluation success rates for 4 tasks in a 2x2 subplot figure. "
            "Pass all run identifiers flat via --identifiers (N tasks × runs-per-task). "
            "Identifiers can be W&B run ids or local offline run directories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python wandb_plots_multi_tasks.py \\\n"
            "    --identifiers task1_run1 task1_run2 task1_run3 task1_run4 \\\n"
            "                  task2_run1 task2_run2 task2_run3 task2_run4 \\\n"
            "                  task3_run1 task3_run2 task3_run3 task3_run4 \\\n"
            "                  task4_run1 task4_run2 task4_run3 task4_run4 \\\n"
            "    --task-titles 'Task 28' 'Task 42' 'Task 57' 'Task 83' \\\n"
            "    --runs-per-task 4 \\\n"
            "    --labels 'task1_run1=Baseline,task1_run2=Method B,...' \\\n"
            "    --output plots/multi_task_success.svg\n"
        ),
    )
    parser.add_argument(
        "--identifiers",
        nargs="+",
        required=True,
        help=(
            "Flat list of W&B run ids or local run dirs, ordered by task. "
            "Must have length == number-of-tasks × --runs-per-task."
        ),
    )
    parser.add_argument(
        "--task-titles",
        nargs="+",
        required=True,
        help="One title per task subplot (must match the number of tasks derived from --identifiers).",
    )
    parser.add_argument(
        "--runs-per-task",
        type=int,
        default=DEFAULT_RUNS_PER_TASK,
        help=f"Number of runs per task (default: {DEFAULT_RUNS_PER_TASK}).",
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
        "--suptitle",
        default=None,
        help="Optional super-title for the whole figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the plot (e.g. plots/multi_task_success.svg).",
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
            f"Halflife in training steps for EMA smoothing (default: {DEFAULT_EMA_HALFLIFE}). "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--clip-to-shortest-run",
        action="store_true",
        help="Truncate each task's x-axis at the last step of its shortest run.",
    )
    parser.add_argument(
        "--errorbar",
        default="ci",
        choices=["ci", "sd", "se", "none"],
        help=(
            "Error band style when multiple runs share the same label (seeds). "
            "ci = 95%% bootstrap CI, sd = std dev, se = std error, none = no band "
            "(default: ci)."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=float,
        default=None,
        help="Maximum number of steps to plot (default: None).",
    )
    parser.add_argument(
        "--dim-colors",
        type=int,
        default=0,
        help="Use dimension-specific colors (default: 0).",
    )
    parser.add_argument(
        "--sft-baselines",
        nargs="+",
        default=None,
        help=(
            "Optional per-task SFT baseline success-rate values, one per task, in the "
            "same order as --task-titles. Use an empty string to skip a task."
        ),
    )
    return parser


def parse_sft_baselines(raw: list[str] | None, n_tasks: int) -> list[float | None]:
    if raw is None:
        return [None] * n_tasks
    if len(raw) != n_tasks:
        raise ValueError(
            f"--sft-baselines has {len(raw)} entries but {n_tasks} tasks were provided."
        )
    return [float(v) if v.strip() != "" else None for v in raw]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    n_ids = len(args.identifiers)
    rpt = args.runs_per_task
    if n_ids % rpt != 0:
        print(
            f"Error: --identifiers length ({n_ids}) is not divisible by "
            f"--runs-per-task ({rpt}).",
            file=sys.stderr,
        )
        return 1

    n_tasks = n_ids // rpt
    if len(args.task_titles) != n_tasks:
        print(
            f"Error: {len(args.task_titles)} --task-titles provided but {n_tasks} tasks "
            f"inferred from --identifiers / --runs-per-task.",
            file=sys.stderr,
        )
        return 1

    try:
        sft_baselines = parse_sft_baselines(args.sft_baselines, n_tasks)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    entity = configure_wandb_auth(args.entity)
    print(f"Entity: {entity}")
    print(f"Project: {args.project}")
    label_mapping = parse_label_mapping(args.labels)

    task_dfs: list[pd.DataFrame] = []
    for t in range(n_tasks):
        task_ids = args.identifiers[t * rpt : (t + 1) * rpt]
        print(f"Fetching task {t + 1}/{n_tasks}: {args.task_titles[t]} ...")
        try:
            df = collect_histories(
                task_ids,
                entity=entity,
                project=args.project,
                metric=args.metric,
                x_axis=args.x_axis,
                label_mapping=label_mapping,
            )
        except ValueError as exc:
            print(f"Error (task {t + 1}): {exc}", file=sys.stderr)
            return 1
        task_dfs.append(df)

    plot_multi_task(
        task_dfs,
        args.task_titles,
        metric=args.metric,
        suptitle=args.suptitle,
        output=args.output,
        show=args.show,
        ylim=(args.ymin, args.ymax),
        ema_halflife=args.ema_halflife,
        clip_to_shortest=args.clip_to_shortest_run,
        errorbar=args.errorbar,
        max_steps=args.max_steps,
        dim_colors=args.dim_colors,
        sft_baselines=sft_baselines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
