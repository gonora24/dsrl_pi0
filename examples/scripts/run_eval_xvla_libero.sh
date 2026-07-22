#!/bin/bash
# Evaluate pretrained XVLA on all tasks in a LIBERO suite (10 rollouts per task by default).
#
# Usage:
#   bash examples/scripts/run_eval_xvla_libero.sh libero_90

set -euo pipefail

suite="libero_90"
num_rollouts=10

module load devel/miniforge 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate dsrl_pi0

export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/LIBERO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=./openpi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 examples/eval_xvla_libero.py \
  --libero_suite "${suite}" \
  --num_rollouts ${num_rollouts} \
  --query_freq 30 \
  --checkpoint 2toINF/X-VLA-Libero \
  --seed 0
