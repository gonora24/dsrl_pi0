#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=jacobian_noise_action_dims

module load devel/miniforge
conda deactivate
conda activate dsrl_pi0

export PYTHONPATH="${PYTHONPATH}:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/LIBERO

# Optional: set to a PixelSACLearner checkpoint directory to use learned noise.
# Leave empty to use random Gaussian noise.
export NOISE_ACTOR_DIR="/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero/dsrl_pi05_libero_90_task59_2026_07_07_09_46_10_0000--s-0_baseline/checkpoint1000000"

# Base checkpoint (required).  Set checkpoint_ft to enable base-vs-finetuned comparison.
export CHECKPOINT_BASE=pi05_base
export CHECKPOINT_FT=pi05_libero          # e.g. pi05_libero

python3 jaxrl2/tests/jacobian.py \
    --checkpoint "${CHECKPOINT_FT}" \
    --noise_actor_dir "${NOISE_ACTOR_DIR}" \
    --libero_suite libero_90 \
    --N 100 \
    --seed 0 \
    --action_idx 0 \
    --noise_idx 0 \
    --task_id 59 \
    --num_actions 7 \
    --filename "jacobian_noise_action_dims_${CHECKPOINT_FT}_libero90_task59_N100_baselineactor_action0_noise0_cropped"
