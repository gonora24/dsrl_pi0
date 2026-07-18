#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=collect_pi05_libero_90_task59

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

device_id=0

export OUTPUT_DIR="./logs"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}:${REPO_ROOT}/LIBERO"

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

pip install mujoco==3.3.1
pip install "transformers==4.53.2"

python3 examples/collect_trajectories.py \
  --env libero \
  --prefix "collect_pi05_libero_90_task${libero_task_id}" \
  --libero_suite libero_90 \
  --libero_task_id 59 \
  --pi0_ckpt pi05_libero \
  --num_trajectories 6000 \
  --query_freq 10 \
  --seed 0 \
  --resize_image 64 \
  --add_states 1 \
  --output_file trajectories.hdf5 \
  --add_noise_to_actions_percentage 2 \
  --noise_action_freq 0.7

# percentage as in every 5th trajectory add noise to the actions
