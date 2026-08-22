#!/usr/bin/env python3
"""Plot residual-prediction norms (inferred from checkpoints) for 4/5/6 tasks
in a shared-legend grid figure, reusing the exact plotting logic of
plots/wandb_plots_multi_tasks.py.

Data source is different from wandb_plots_multi_tasks.py: instead of fetching
a metric's history from the W&B API, this script reads the long-format CSVs
produced by plots/eval_residual_norms.py (columns: step, metric, value,
rollout, query_idx, ar_step) and filters them down to a single --metric
(one of "residual_norm", "residual_mean_norm", "residual_log_std_norm").

The filtered rows are renamed to the schema plot_multi_task() expects
(step, success_rate, label, identifier) so the grid layout, coloring, EMA
smoothing, and CI-band logic are shared verbatim with the W&B plotting script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from wandb_plots_multi_tasks import (
    DEFAULT_RUNS_PER_TASK,
    parse_label_mapping,
    parse_sft_baselines,
    plot_multi_task,
    short_label,
)

DEFAULT_METRIC = "residual_norm"
DEFAULT_DATA_DIR = Path("plots/data/residual_norms")


def resolve_csv_path(identifier: str, data_dir: Path) -> Path:
    """Identifiers may be a run name (looked up under data_dir) or a direct CSV path."""
    path = Path(identifier)
    if path.exists():
        return path
    candidate = data_dir / f"{identifier}.csv"
    if candidate.exists():
        return candidate
    raise ValueError(
        f"Could not find a residual-norm CSV for identifier {identifier!r} "
        f"(tried {path} and {candidate}). Run plots/eval_residual_norms.py first."
    )


def load_residual_history(csv_path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"step", "metric", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")

    history = df.loc[df["metric"] == metric, ["step", "value"]].copy()
    if history.empty:
        available = sorted(df["metric"].unique())
        raise ValueError(
            f"{csv_path} has no rows for metric {metric!r}. Available metrics: {available}"
        )
    history = history.rename(columns={"value": "success_rate"})
    history["identifier"] = csv_path.stem
    history["run_name"] = csv_path.stem
    return history


def collect_histories(
    identifiers: Iterable[str],
    *,
    metric: str,
    data_dir: Path,
    label_mapping: dict[str, str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for identifier in identifiers:
        if identifier == "":
            continue
        csv_path = resolve_csv_path(identifier, data_dir)
        history = load_residual_history(csv_path, metric)
        label = (
            label_mapping.get(identifier)
            or label_mapping.get(csv_path.stem)
            or short_label(csv_path.stem)
        )
        history["label"] = label
        frames.append(history)
    if not frames:
        raise ValueError("No non-empty identifiers were provided for this task.")
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot residual-prediction norms inferred from checkpoints for N tasks "
            "in a shared-legend grid figure. Pass all run identifiers flat via "
            "--identifiers (N tasks x runs-per-task); identifiers are run names "
            "(looked up under --data-dir/<name>.csv) or direct CSV paths."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python plots/plot_residual_norms_multi_tasks.py \\\n"
            "    --identifiers task1_run1 task1_run2 task2_run1 task2_run2 \\\n"
            "    --task-titles 'Task 28' 'Task 43' \\\n"
            "    --runs-per-task 2 \\\n"
            "    --metric residual_norm \\\n"
            "    --labels 'task1_run1=Residual-Noise,...' \\\n"
            "    --output plots/plots_multi_tasks/residual_norm.svg\n"
        ),
    )
    parser.add_argument(
        "--identifiers",
        nargs="+",
        required=True,
        help=(
            "Flat list of run names or CSV paths, ordered by task. "
            "Must have length == number-of-tasks x --runs-per-task."
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
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing <run_name>.csv files (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=["residual_norm", "residual_mean_norm", "residual_log_std_norm"],
        help=f"Which residual-norm metric to plot (default: {DEFAULT_METRIC}).",
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
        help="Path to save the plot (e.g. plots/plots_multi_tasks/residual_norm.svg).",
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
        default=0.0,
        help=(
            "Halflife in training steps for EMA smoothing (default: 0, disabled). "
            "Checkpoints are sparse and each step already carries many rollout "
            "samples, so smoothing is off by default; the per-step spread is shown "
            "via --errorbar instead."
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
            "Error band style across rollout/query/AR-step samples sharing a "
            "checkpoint step (and across seeds sharing a label). "
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
        "--residual-colors",
        type=int,
        default=0,
        help="Use residual-specific colors (default: 0).",
    )
    parser.add_argument(
        "--sft-baselines",
        nargs="+",
        default=None,
        help=(
            "Optional per-task SFT baseline values, one per task, in the same "
            "order as --task-titles. Use an empty string to skip a task."
        ),
    )
    return parser


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

    label_mapping = parse_label_mapping(args.labels)

    task_dfs: list[pd.DataFrame] = []
    for t in range(n_tasks):
        task_ids = args.identifiers[t * rpt : (t + 1) * rpt]
        print(f"Loading task {t + 1}/{n_tasks}: {args.task_titles[t]} ...")
        try:
            df = collect_histories(
                task_ids,
                metric=args.metric,
                data_dir=args.data_dir,
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
        label_mapping=label_mapping,
        dim_colors=args.dim_colors,
        residual_colors=args.residual_colors,
        sft_baselines=sft_baselines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
