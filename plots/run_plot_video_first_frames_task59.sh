#!/bin/bash
# set -euo pipefail

# Plot the first frame of two videos side by side, each with its own title.

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

# --- Videos to compare ---
VIDEO1="/hkfs/work/workspace/scratch/eu3660-dsrl/PI05_Libero_Tests/multimodality_eval/20260817_140016_libero_object_task_5_pi05_libero/part1_standard/rollout_001_success_1.mp4"
TITLE1="\$\pi_{0.5}\$ LIBERO-Object Task 5"
SUBTITLE1="pick up the tomato sauce and place it in the basket"  # optional language instruction shown below the frame, e.g. "put the bowl on the plate"

VIDEO2="/hkfs/work/workspace/scratch/eu3660-dsrl/PI05_Libero_Tests/multimodality_eval/20260817_135509_libero_10_task_0_pi05_libero/part1_standard/rollout_003_success_1.mp4"
TITLE2="\$\pi_{0.5}\$ LIBERO-10 Task 0"
SUBTITLE2="put both the alphabet soup and the tomato sauce in the basket"  # optional language instruction shown below the frame

# Optional third video, plotted alongside the first two. Leave VIDEO3 empty
# to only plot two panels.
VIDEO3="/hkfs/work/workspace/scratch/eu3660-dsrl/PI05_Libero_Tests/multimodality_eval/20260817_134955_libero_90_task_59_pi05_libero/part1_standard/rollout_010_success_1.mp4"
TITLE3="\$\pi_{0.5}\$ LIBERO-90 Task 59"
SUBTITLE3="pick up the tomato sauce and put it in the tray"

# --- Output settings ---
OUTPUT="plots/plots/frame_comparison_task59.svg"
WSPACE=0.4
SUBTITLE_WRAP_WIDTH=40
DPI=200

EXTRA_ARGS=()
if [ -n "${VIDEO3}" ]; then
    EXTRA_ARGS+=(--video3 "${VIDEO3}" --title3 "${TITLE3}" --subtitle3 "${SUBTITLE3}")
fi

python3 plots/plot_video_first_frames.py \
    --video1 "${VIDEO1}" --title1 "${TITLE1}" --subtitle1 "${SUBTITLE1}" \
    --video2 "${VIDEO2}" --title2 "${TITLE2}" --subtitle2 "${SUBTITLE2}" \
    "${EXTRA_ARGS[@]}" \
    --output "${OUTPUT}" \
    --wspace ${WSPACE} \
    --subtitle_wrap_width ${SUBTITLE_WRAP_WIDTH} \
    --dpi ${DPI}
