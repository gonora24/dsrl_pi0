#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=repeat_noise_eval
#
# Evaluate pi0 with a DSRL noise actor whose predicted noise vector is held
# fixed for QUERY_REPEAT consecutive action chunks, then re-queried.
#
# Usage (interactive):
#   NOISE_ACTOR_DIR=/path/to/checkpoint bash examples/scripts/run_repeat_noise_eval.sh
#
# Usage (Slurm):
#   NOISE_ACTOR_DIR=/path/to/checkpoint sbatch examples/scripts/run_repeat_noise_eval.sh
#
# Override any variable via the environment before running, e.g.:
#   QUERY_REPEAT=10 TASK_ID=59 NOISE_ACTOR_DIR=... sbatch examples/scripts/run_repeat_noise_eval.sh

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
# Path to the PixelSACLearner checkpoint directory — must be set by the caller.
NOISE_ACTOR_DIR="/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero/dsrl_pi05_libero_90_task28_2026_07_06_17_08_06_0000--s-0_baseline/checkpoint500000"

# How many consecutive action chunks reuse the same noise vector.
# 1 = re-query every chunk (baseline, no repeat).
QUERY_REPEAT="${QUERY_REPEAT:-30}"


NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
TASK_ID="${TASK_ID:-28}"
LIBERO_SUITE="${LIBERO_SUITE:-libero_90}"
PI0_CHECKPOINT="${PI0_CHECKPOINT:-pi05_libero}"
# Env steps per action chunk (must match the value used during training).
QUERY_FREQ="${QUERY_FREQ:-5}"
SEED="${SEED:-0}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Noise-repeat evaluation — pi0 + DSRL noise actor"
echo "  Suite:              ${LIBERO_SUITE}"
echo "  Task ID:            ${TASK_ID}"
echo "  Rollouts:           ${NUM_ROLLOUTS}"
echo "  query_freq:         ${QUERY_FREQ}"
echo "  query_repeat:       ${QUERY_REPEAT}"
echo "  Noise actor:   ${NOISE_ACTOR_DIR}"
echo "  Pi0 ckpt:      ${PI0_CHECKPOINT}"
echo "  Output dir:         ${OUTPUT_DIR}/repeat_noise_eval"
echo "========================================================"

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
python3 examples/eval_repeat_pi.py \
  --noise_actor_dir  "${NOISE_ACTOR_DIR}" \
  --query_repeat     "${QUERY_REPEAT}" \
  --libero_suite     "${LIBERO_SUITE}" \
  --task_id          "${TASK_ID}" \
  --num_rollouts     "${NUM_ROLLOUTS}" \
  --pi0_checkpoint   "${PI0_CHECKPOINT}" \
  --query_freq       "${QUERY_FREQ}" \
  --seed             "${SEED}"
