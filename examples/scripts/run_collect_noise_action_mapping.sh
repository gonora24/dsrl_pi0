#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=collect_noise_pi05_libero_90_task60

module load devel/miniforge 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate dsrl_pi0
device_id=0

export OUTPUT_DIR="/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_NA_noise_action_mapping"
export PYTHONPATH="${PYTHONPATH}:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/LIBERO
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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

python3 examples/collect_noise_action_mapping.py \
  --env libero \
  --libero_suite libero_90 \
  --libero_task_id 59 \
  --pi0_ckpt pi05_libero \
  --num_mappings 2000000 \
  --query_freq 5 \
  --seed 0 \
  --output_file noise_action_mapping_task59.h5 \
