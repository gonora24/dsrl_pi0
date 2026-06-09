#!/bin/bash
# Evaluate Pi0.5 on all tasks in a LIBERO suite (10 rollouts per task by default).
#
# Usage:
#   bash examples/scripts/run_eval_pi05_libero.sh libero_90
#   bash examples/scripts/run_eval_pi05_libero.sh libero_spatial

set -euo pipefail

suite="libero_object"
num_rollouts="${2:-10}"

module load devel/miniforge 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate dsrl_pi0

export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/LIBERO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=./openpi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 examples/eval_pi05_libero.py \
  --libero_suite "${suite}" \
  --num_rollouts "${num_rollouts}" \
  --query_freq 5 \
  --pi0_checkpoint pi05_libero \
  --seed 0
