#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=multimodality_eval_pi05
#
# Multimodality evaluation for pi0.5 on the LIBERO frying-pan task.
#
# Usage (interactive):
#   bash examples/scripts/run_multimodality_eval.sh
#
# Usage (Slurm):
#   sbatch examples/scripts/run_multimodality_eval.sh
#
# Override defaults via env vars before submitting, e.g.:
#   PAN_ROTATION_DEG=180 sbatch examples/scripts/run_multimodality_eval.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module load devel/miniforge 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate dsrl_pi0

device_id=0
export CUDA_VISIBLE_DEVICES=$device_id
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

export OUTPUT_DIR=/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/PI05_Libero_Tests
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/LIBERO"

# Keep model versions in sync with the main training script.
pip install mujoco==3.3.1 --quiet
pip install "transformers==4.53.2" --quiet

# ---------------------------------------------------------------------------
# Experiment parameters (override via env vars as needed)
# ---------------------------------------------------------------------------
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
TASK_ID="${TASK_ID:-64}"              # 18 = KITCHEN_SCENE3_put_the_frying_pan_on_the_stove
LIBERO_SUITE="${LIBERO_SUITE:-libero_90}"
PI0_CHECKPOINT="rlinf_hf_pi05"
QUERY_FREQ="${QUERY_FREQ:-5}"
SEED="${SEED:-0}"

# Part 2: how many degrees to rotate the pan handle around Z, also book for task 77
PAN_ROTATION_DEG="25.0"

# Part 3: how far to shift the stove (target) in metres, or desk caddy for task 77
TARGET_OFFSET_X="0.0"
TARGET_OFFSET_Y="0.10"

# Which parts to run (space-separated, e.g. "1 2 3")
PARTS="${PARTS:-1 2 3}"

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Multimodality evaluation — pi0.5"
echo "  Suite:            ${LIBERO_SUITE}"
echo "  Task ID:          ${TASK_ID}"
echo "  Rollouts / part:  ${NUM_ROLLOUTS}"
echo "  Pan rotation:     ${PAN_ROTATION_DEG} deg (Part 2)"
echo "  Target offset:    dx=${TARGET_OFFSET_X} m, dy=${TARGET_OFFSET_Y} m (Part 3)"
echo "  Parts:            ${PARTS}"
echo "  Output dir:       ${OUTPUT_DIR}/multimodality_eval"
echo "========================================================"

# shellcheck disable=SC2086
python3 examples/eval_multimodality_pi05.py \
  --libero_suite     "${LIBERO_SUITE}" \
  --task_id          "${TASK_ID}" \
  --num_rollouts     "${NUM_ROLLOUTS}" \
  --pi0_checkpoint   "${PI0_CHECKPOINT}" \
  --query_freq       "${QUERY_FREQ}" \
  --seed             "${SEED}" \
  --pan_rotation_deg "${PAN_ROTATION_DEG}" \
  --target_offset_x  "${TARGET_OFFSET_X}" \
  --target_offset_y  "${TARGET_OFFSET_Y}" \
  --parts            ${PARTS}
