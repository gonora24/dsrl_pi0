"""Shared figure layout constants and helpers for W&B multi-task plots.

Vertical stacking order (top-down):
    SUPTITLE_H          space for the figure suptitle
    TITLE_LEGEND_GAP    gap between suptitle and legend
    legend_h            legend box height (varies with number of labels/columns)
    LEGEND_SUBPLOT_GAP  gap between legend bottom and subplot title tops
    row_span            subplot rows + ROW_GAP between rows
    BOTTOM_PAD          room for x-axis labels
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

SUBPLOT_W = 7.0    # inches per subplot column
SUBPLOT_H = 4.5    # inches per subplot row (title + axes + xlabel)

LEGEND_ROW_H = 0.38        # inches per legend row
LEGEND_PAD = 0.20          # padding inside legend box (top+bottom combined)

SUPTITLE_H = 0.55          # vertical space reserved for the figure suptitle
TITLE_LEGEND_GAP = 0.15    # gap between suptitle bottom and legend top
LEGEND_SUBPLOT_GAP = 0.40  # gap between legend bottom and subplot title tops
BOTTOM_PAD = 0.35          # room for x-axis tick labels below subplot grid

# Fixed per-subplot decoration sizes so AXES_H is identical across grid shapes.
TITLE_H = 0.42    # height of subplot axes title
XLABEL_H = 0.38   # height below axes box for x-axis label + ticks
AXES_H = SUBPLOT_H - TITLE_H - XLABEL_H   # pure axes box height

LEFT_PAD = 0.55   # left margin per column (y-axis label + ticks)
CELL_PAD = 0.15   # right/inner padding per column
AXES_W = SUBPLOT_W - LEFT_PAD - CELL_PAD
ROW_GAP = 0.40


def row_span_inches(n_rows: int) -> float:
    """Total height occupied by the subplot grid rows, in inches."""
    return n_rows * SUBPLOT_H + max(0, n_rows - 1) * ROW_GAP


def legend_height(n_labels: int, ncol: int) -> float:
    """Height of the legend box in inches."""
    if n_labels <= 0:
        n_labels = 1
    ncol = max(1, ncol)
    n_legend_rows = math.ceil(n_labels / ncol)
    return LEGEND_PAD + LEGEND_ROW_H * n_legend_rows


def figure_size(
    n_rows: int,
    n_cols: int,
    *,
    suptitle: bool,
    n_legend_labels: int,
    n_legend_cols: int,
) -> tuple[float, float, float, float]:
    """Return (fig_w, fig_h, suptitle_h, legend_h).

    The top margin stacks top-down as:
        suptitle_h + TITLE_LEGEND_GAP + legend_h + LEGEND_SUBPLOT_GAP
    """
    leg_h = legend_height(n_legend_labels, n_legend_cols)
    suptitle_h = SUPTITLE_H if suptitle else 0.0
    top_margin = suptitle_h + TITLE_LEGEND_GAP + leg_h + LEGEND_SUBPLOT_GAP
    fig_w = n_cols * SUBPLOT_W
    fig_h = row_span_inches(n_rows) + top_margin + BOTTOM_PAD
    return fig_w, fig_h, suptitle_h, leg_h


def axes_position(
    row: int,
    col: int,
    *,
    n_rows: int,
    fig_w: float,
    fig_h: float,
) -> list[float]:
    """Return [left, bottom, width, height] in figure fractions for an axes patch.

    All subplots have identical width and height regardless of grid shape.
    """
    left = (col * SUBPLOT_W + LEFT_PAD) / fig_w
    width = AXES_W / fig_w
    # Row 0 is the topmost; row (n_rows-1) is the bottom row.
    row_bottom = BOTTOM_PAD + (n_rows - 1 - row) * (SUBPLOT_H + ROW_GAP)
    bottom = (row_bottom + XLABEL_H) / fig_h
    height = AXES_H / fig_h
    return [left, bottom, width, height]


def apply_axes_grid(
    axes: np.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    fig_w: float,
    fig_h: float,
) -> None:
    """Place every axes patch at a fixed, identical size."""
    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_position(
            axes_position(row, col, n_rows=n_rows, fig_w=fig_w, fig_h=fig_h)
        )


def add_top_decorations(
    fig: plt.Figure,
    *,
    suptitle: str | None,
    legend_handles,
    fig_h: float,
    suptitle_h: float,
    legend_h: float,
    n_legend_cols: int,
) -> None:
    """Add suptitle and shared legend anchored top-down.

    Suptitle sits at the very top, legend is placed directly below it with
    TITLE_LEGEND_GAP separating them.  The gap between the legend bottom and
    the subplot titles is LEGEND_SUBPLOT_GAP.
    """
    if suptitle:
        # Centre of suptitle text sits half a SUPTITLE_H below the figure top.
        fig.suptitle(suptitle, y=1.0 - (suptitle_h * 0.5) / fig_h)

    # legend_top_y: figure-fraction y of the legend's top edge.
    legend_top_y = (fig_h - suptitle_h - TITLE_LEGEND_GAP) / fig_h
    # Anchor the upper-centre of the legend at legend_top_y.
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_top_y),
        ncol=n_legend_cols,
        frameon=False,
        fancybox=False,
        framealpha=1.0,
        edgecolor="0.6",
        facecolor="white",
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=1.0,
    )
