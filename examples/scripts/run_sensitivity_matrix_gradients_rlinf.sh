#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=sensitivity_matrix_rlinf

module load devel/miniforge
conda deactivate
conda activate dsrl_pi0

export PYTHONPATH="${PYTHONPATH}:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/LIBERO

export NOISE_ACTOR_DIR=

python3 jaxrl2/tests/gradient_sensitivity_pytorch.py \
    --libero_suite libero_90 \
    --task_id 59 \
    --N 100 \
    --seed 0 \
    --filename "gradient_sensitivity_libero_90_task59_rlinf_hf_pi05_100states_randomnoise" \
    --checkpoint rlinf_hf_pi05
