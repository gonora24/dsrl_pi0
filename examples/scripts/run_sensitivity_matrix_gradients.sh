#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=sensitivity_matrix

# module load devel/miniforge
# conda deactivate
# conda activate dsrl_pi0

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

python3 jaxrl2/tests/gradient_sensitivity.py \
    --checkpoint pi05_libero \
    --libero_suite libero_90 \
    --task_id 43 \
    --N 1 \
    --seed 0 \
    --filename "gradient_sensitivity_pi05_libero_90_task43_1state"