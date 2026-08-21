#!/usr/bin/env python3
"""Plot the first frame of two or three videos side by side, each with its own title.

Reads frame 0 from two (or optionally three) video files (e.g. rollout
``.mp4`` files produced by the eval scripts under ``examples/``) and renders
them as side-by-side panels in a single figure, with a title above each panel
and a configurable gap between panels. Each panel can also show a smaller,
wrapped subtitle below its image (e.g. the language instruction for that
task); subtitles are automatically wrapped in quotation marks when rendered,
so pass the raw instruction text without adding quotes yourself.

Usage
-----
python plots/plot_video_first_frames.py \\
    --video1 path/to/rollout_a.mp4 --title1 "Standard" \\
    --subtitle1 "pick up the black bowl and place it on the plate" \\
    --video2 path/to/rollout_b.mp4 --title2 "Modified" \\
    --subtitle2 "pick up the red mug and place it in the sink" \\
    --video3 path/to/rollout_c.mp4 --title3 "Another Modification" \\
    --subtitle3 "pick up the green cup and place it on the shelf" \\
    --output plots/plots/frame_comparison.png \\
    --wspace 0.4
"""

from __future__ import annotations

import argparse
import pathlib
import textwrap

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "mathtext.fontset": "cm",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_first_frame(path: pathlib.Path) -> np.ndarray:
    """Read the first frame (frame 0) of a video file.

    Args:
        path : path to a video file (e.g. .mp4)

    Returns:
        frame : (H, W, 3) uint8 RGB array
    """
    reader = imageio.get_reader(str(path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    return frame


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_first_frames(
    frames: list[np.ndarray],
    titles: list[str],
    subtitles: list[str | None] | None = None,
    wspace: float = 0.3,
    figsize: tuple[float, float] | None = None,
    subtitle_wrap_width: int = 40,
):
    """Plot two or three frames side by side, each with its own title above it.

    Args:
        frames               : list of 2 or 3 (H, W, 3) uint8 RGB arrays
        titles               : list of title strings, one per frame, drawn
                               above the corresponding panel
        subtitles            : optional list of smaller captions, one per
                               frame, drawn below the corresponding panel
                               (e.g. its language instruction), automatically
                               wrapped in quotation marks; a panel's subtitle
                               is omitted entirely if None/empty
        wspace               : horizontal space between panels, passed to
                               ``fig.subplots_adjust`` (0 = panels touching;
                               larger values leave more of a gap)
        figsize              : optional (width, height) override; by default
                               it is derived from the frames' aspect ratios
        subtitle_wrap_width  : max characters per line before a subtitle is
                               wrapped onto the next line (default: 40)

    Returns:
        fig, axes
    """
    n = len(frames)
    if n not in (2, 3):
        raise ValueError(f"plot_first_frames: expected 2 or 3 frames, got {n}")
    if len(titles) != n:
        raise ValueError(f"plot_first_frames: expected {n} titles, got {len(titles)}")
    if subtitles is None:
        subtitles = [None] * n
    if len(subtitles) != n:
        raise ValueError(f"plot_first_frames: expected {n} subtitles, got {len(subtitles)}")

    if figsize is None:
        panel_height = 4.0
        panel_widths = [panel_height * (f.shape[1] / f.shape[0]) for f in frames]
        figsize = (sum(panel_widths), panel_height)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    for ax, frame, title, subtitle in zip(axes, frames, titles, subtitles):
        ax.imshow(frame)
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis("off")
        if subtitle:
            wrapped = textwrap.fill(f'"{subtitle}"', width=subtitle_wrap_width)
            ax.text(
                0.5, -0.04, wrapped,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=10,
            )

    fig.subplots_adjust(wspace=wspace)
    return fig, axes


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the first frame of two or three videos side by side, each "
            "with its own title above it and a configurable gap between "
            "panels."
        )
    )
    parser.add_argument(
        "--video1", type=str, required=True,
        help="Path to the first video file.",
    )
    parser.add_argument(
        "--title1", type=str, required=True,
        help="Title drawn above the first video's frame.",
    )
    parser.add_argument(
        "--subtitle1", type=str, default=None,
        help=(
            "Optional smaller caption drawn below the first video's frame "
            "(e.g. its language instruction), automatically wrapped in "
            "quotation marks. Omitted if not given."
        ),
    )
    parser.add_argument(
        "--video2", type=str, required=True,
        help="Path to the second video file.",
    )
    parser.add_argument(
        "--title2", type=str, required=True,
        help="Title drawn above the second video's frame.",
    )
    parser.add_argument(
        "--subtitle2", type=str, default=None,
        help=(
            "Optional smaller caption drawn below the second video's frame "
            "(e.g. its language instruction), automatically wrapped in "
            "quotation marks. Omitted if not given."
        ),
    )
    parser.add_argument(
        "--video3", type=str, default=None,
        help="Optional path to a third video file, plotted alongside the first two.",
    )
    parser.add_argument(
        "--title3", type=str, default=None,
        help="Title drawn above the third video's frame (required if --video3 is given).",
    )
    parser.add_argument(
        "--subtitle3", type=str, default=None,
        help=(
            "Optional smaller caption drawn below the third video's frame "
            "(e.g. its language instruction), automatically wrapped in "
            "quotation marks. Omitted if not given."
        ),
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output image path (e.g. plots/plots/frame_comparison.png or .svg).",
    )
    parser.add_argument(
        "--wspace", type=float, default=0.3,
        help="Horizontal space between panels (default: 0.3).",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
        help="Optional figure size override in inches (default: derived from frame aspect ratios).",
    )
    parser.add_argument(
        "--subtitle_wrap_width", type=int, default=40,
        help="Max characters per line before a subtitle wraps onto the next line (default: 40).",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Resolution (dots per inch) used when saving the figure (default: 200).",
    )
    args = parser.parse_args()

    if args.video3 and not args.title3:
        parser.error("--title3 is required when --video3 is given.")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_paths = [args.video1, args.video2]
    titles = [args.title1, args.title2]
    subtitles = [args.subtitle1, args.subtitle2]
    if args.video3:
        video_paths.append(args.video3)
        titles.append(args.title3)
        subtitles.append(args.subtitle3)

    frames = []
    for video_path in video_paths:
        print(f"Reading first frame of {video_path}...", flush=True)
        frames.append(read_first_frame(pathlib.Path(video_path)))

    fig, _ = plot_first_frames(
        frames,
        titles=titles,
        subtitles=subtitles,
        wspace=args.wspace,
        figsize=tuple(args.figsize) if args.figsize is not None else None,
        subtitle_wrap_width=args.subtitle_wrap_width,
    )

    fig.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved figure → {out_path}", flush=True)


if __name__ == "__main__":
    main()
