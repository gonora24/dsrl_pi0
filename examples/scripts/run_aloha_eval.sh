#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_pi0_aloha_sim

module load devel/miniforge
conda deactivate
conda activate dsrl_pi0

device_id=0

export PYTHONPATH="${PYTHONPATH}:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/"

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

pip install mujoco==2.3.7

python3 examples/eval_pi0_aloha_sim.py \
  --tasks transfer_cube insertion \
  --num_rollouts 10 \
  --query_freq 10 \
  --pi0_checkpoint pi0_aloha_sim \
  --seed 0
