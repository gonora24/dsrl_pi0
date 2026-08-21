#!/bin/bash
# set -euo pipefail

# Plot the first frame of two videos side by side, each with its own title.

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

# --- Videos to compare ---
VIDEO1="/hkfs/work/workspace/scratch/eu3660-dsrl/PI05_Libero_Tests/multimodality_eval/20260817_134112_libero_goal_task_4_pi05_libero/part1_standard/rollout_000_success_1.mp4"
TITLE1="\$\pi_{0.5}\$ LIBERO-Goal Task 4"
SUBTITLE1="put the bowl on top of the cabinet"  # optional language instruction shown below the frame, e.g. "put the bowl on the plate"

VIDEO2="/hkfs/work/workspace/scratch/eu3660-dsrl/PI05_Libero_Tests/multimodality_eval/20260817_134353_libero_90_task_43_pi05_libero/part1_standard/rollout_003_success_0.mp4"
TITLE2="\$\pi_{0.5}\$ LIBERO-90 Task 43"
SUBTITLE2="put the white bowl on top of the cabinet"  # optional language instruction shown below the frame

# Optional third video, plotted alongside the first two. Leave VIDEO3 empty
# to only plot two panels.
VIDEO3=""
TITLE3=""
SUBTITLE3=""

# --- Output settings ---
OUTPUT="plots/plots/frame_comparison_task43.svg"
WSPACE=0.4
SUBTITLE_WRAP_WIDTH=60
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
