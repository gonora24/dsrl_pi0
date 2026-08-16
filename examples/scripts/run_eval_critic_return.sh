#!/bin/bash
# Evaluate a trained DSRL-NA critic against realized returns from pi0.5 rollouts
# on a LIBERO task.
#
# Required environment variable:
#   DSRL_CHECKPOINT  path to the checkpointN directory
#                    (e.g. /path/to/run_xyz/checkpoint941)
#
# Optional environment variables (with defaults):
#   PI0_CHECKPOINT   pi0 preset or local path  (default: pi05_libero)
#   TASK_ID          LIBERO task index          (default: 59)
#   LIBERO_SUITE     LIBERO suite name          (default: libero_90)
#   NUM_ROLLOUTS     number of rollouts         (default: 10)
#   SEED             random seed                (default: 0)
#   OUTPUT_JSON      path to write JSON results (default: auto-generated)
#
# query_freq is auto-derived from the checkpoint config (no need to specify).
#
# Usage:
#   DSRL_CHECKPOINT=/path/to/checkpoint941 bash examples/scripts/run_eval_critic_return.sh
#   DSRL_CHECKPOINT=/path/to/checkpoint941 TASK_ID=42 NUM_ROLLOUTS=20 \
#       bash examples/scripts/run_eval_critic_return.sh

set -euo pipefail


DSRL_CHECKPOINT="/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_NA_Libero/dsrl_na_libero_90_task59_2026_08_12_19_44_15_0000--s-0_singleactorcritic_offline/checkpoint30000"
PI0_CHECKPOINT="${PI0_CHECKPOINT:-pi05_libero}"
TASK_ID="${TASK_ID:-59}"
LIBERO_SUITE="${LIBERO_SUITE:-libero_90}"
NUM_ROLLOUTS="10"
SEED="${SEED:-0}"

module load devel/miniforge 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate dsrl_pi0

export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/LIBERO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=./openpi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXTRA_ARGS=()
if [[ -n "${OUTPUT_JSON:-}" ]]; then
    EXTRA_ARGS+=(--output_json "${OUTPUT_JSON}")
fi

python3 examples/eval_critic_return.py \
    --dsrl_checkpoint "${DSRL_CHECKPOINT}" \
    --pi0_checkpoint  "${PI0_CHECKPOINT}" \
    --task_id         "${TASK_ID}" \
    --libero_suite    "${LIBERO_SUITE}" \
    --num_rollouts    "${NUM_ROLLOUTS}" \
    --seed            "${SEED}" \
    "${EXTRA_ARGS[@]}"
