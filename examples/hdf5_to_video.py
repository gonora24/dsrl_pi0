#!/usr/bin/env python3
"""Generate an MP4 video from a rollout stored in trajectories.hdf5."""

import argparse
import os
import sys

import h5py
import imageio
import numpy as np


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def load_demo_frames(hdf5_path, demo_idx):
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise ValueError(
                f"'data' group not found in {hdf5_path}. "
                "Expected a trajectories.hdf5 file from collect_trajectories.py."
            )

        data_grp = f["data"]
        demo_key = f"demo_{demo_idx}"
        if demo_key not in data_grp:
            available = sorted(k for k in data_grp.keys() if k.startswith("demo_"))
            raise ValueError(
                f"'{demo_key}' not found in {hdf5_path}. "
                f"Available demos: {available or 'none'}"
            )

        demo = data_grp[demo_key]
        pixels_key = "obs/pixels"
        if pixels_key not in demo:
            raise ValueError(
                f"'{pixels_key}' not found in {hdf5_path}:{demo_key}. "
                "This HDF5 file may use a different schema (e.g. LIBERO demo.hdf5 "
                "without saved pixel observations)."
            )

        pixels = demo[pixels_key][:]
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError(
                f"Expected pixel shape (T, H, W, 3), got {pixels.shape} "
                f"in {hdf5_path}:{demo_key}/{pixels_key}."
            )
        if pixels.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 pixels, got dtype {pixels.dtype} "
                f"in {hdf5_path}:{demo_key}/{pixels_key}."
            )

        meta = {key: _decode_attr(data_grp.attrs[key]) for key in data_grp.attrs}
        demo_meta = {
            "is_success": bool(demo.attrs.get("is_success", False)),
            "env_steps": int(demo.attrs.get("env_steps", len(pixels) - 1)),
        }

    return pixels, meta, demo_meta


def default_output_path(hdf5_path, demo_idx):
    hdf5_dir = os.path.dirname(os.path.abspath(hdf5_path))
    return os.path.join(hdf5_dir, f"demo_{demo_idx}.mp4")


def write_video(frames, output_path, fps):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    imageio.mimwrite(output_path, [np.asarray(frame) for frame in frames], fps=fps)


def main():
    parser = argparse.ArgumentParser(
        description="Generate an MP4 video from a rollout in trajectories.hdf5."
    )
    parser.add_argument(
        "--hdf5_path",
        required=True,
        help="Path to trajectories.hdf5",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path (default: <hdf5_dir>/demo_<demo_idx>.mp4)",
    )
    parser.add_argument(
        "--demo_idx",
        type=int,
        default=0,
        help="Rollout index to render (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frames per second (default: 10)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5_path):
        print(f"Error: HDF5 file not found: {args.hdf5_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or default_output_path(args.hdf5_path, args.demo_idx)

    try:
        frames, meta, demo_meta = load_demo_frames(args.hdf5_path, args.demo_idx)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    task_description = meta.get("task_description", "unknown task")
    print(f"Task: {task_description}")
    print(f"Demo: demo_{args.demo_idx}")
    print(f"Success: {demo_meta['is_success']}")
    print(f"Env steps: {demo_meta['env_steps']}")
    print(f"Frames: {len(frames)} ({frames.shape[1]}x{frames.shape[2]})")

    write_video(frames, output_path, args.fps)
    print(f"Wrote video to {output_path}")


if __name__ == "__main__":
    main()
