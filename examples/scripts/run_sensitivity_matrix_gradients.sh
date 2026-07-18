#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=sensitivity_matrix

module load devel/miniforge
conda deactivate
conda activate dsrl_pi0

export PYTHONPATH="${PYTHONPATH}:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/LIBERO

export NOISE_ACTOR_DIR="/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero/dsrl_pi05_libero_90_task28_2026_07_06_17_08_06_0000--s-0_baseline/checkpoint500000"

python3 jaxrl2/tests/gradient_sensitivity.py \
    --libero_suite libero_90 \
    --task_id 28 \
    --N 300 \
    --seed 0 \
    --filename "gradient_sensitivity_libero_90_task28_300states"