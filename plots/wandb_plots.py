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
from matplotlib.ticker import FuncFormatter


DEFAULT_PROJECT = "DSRL_pi0_Libero"
DEFAULT_METRIC = "evaluation/success_rate"
DEFAULT_X_AXIS = "_step"
DEFAULT_EMA_HALFLIFE = 50_000


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
    """Pick readable major ticks such as 0, 250K, 500K, 1M, 2M."""
    if max_step <= 0:
        return [0.0]

    nice_units = [
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        2_500_000,
        5_000_000,
    ]
    target_ticks = 6
    best_ticks = [0.0, max_step]
    for unit in nice_units:
        ticks = [float(i * unit) for i in range(int(max_step / unit) + 2) if i * unit <= max_step]
        if 4 <= len(ticks) <= 10:
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
        raise ValueError(f"Could not find run {identifier!r} in {project!r}.") from exc

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


def plot_success_rates(
    df: pd.DataFrame,
    *,
    title: str | None,
    output: Path | None,
    show: bool,
    ylim: tuple[float, float] | None,
    ema_halflife: float,
) -> None:
    df = apply_smoothing(df, ema_halflife)

    sns.set_theme(style="whitegrid", context="notebook")
    n_runs = df["label"].nunique()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="step",
        y="success_rate",
        hue="label",
        linewidth=1.8,
        ax=ax,
    )

    max_step = float(df["step"].max())
    ax.set_xlim(0, max_step)
    ax.set_xticks(nice_step_ticks(max_step))
    ax.xaxis.set_major_formatter(FuncFormatter(format_training_steps))

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate")
    ax.set_title(title or "Evaluation success rate", fontsize=14)

    ymin_val = 0.0 if ylim is None else ylim[0]
    ymax_val = 1.0 if ylim is None else ylim[1]
    bottom_pad = 0.02 if ymin_val <= 0.0 else 0.0
    top_pad = 0.02 if ymax_val >= 1.0 else 0.0
    ax.set_ylim(ymin_val - bottom_pad, ymax_val + top_pad)
    ax.set_axisbelow(True)
    for line in ax.lines:
        line.set_zorder(3)
        line.set_clip_on(False)
    ax.legend(
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(n_runs, 3),
        frameon=True,
        handlelength=1.5,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.28 if n_runs <= 3 else 0.34)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    entity = configure_wandb_auth(args.entity)
    label_mapping = parse_label_mapping(args.labels)

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
        title=args.title,
        output=args.output,
        show=args.show,
        ylim=(args.ymin, args.ymax),
        ema_halflife=args.ema_halflife,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
