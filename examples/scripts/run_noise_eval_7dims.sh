#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=jacobian_noise_action_dims

# module load devel/miniforge
# conda deactivate
# conda activate dsrl_pi0

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

# Optional: set to a PixelSACLearner checkpoint directory to use learned noise.
# Leave empty to use random Gaussian noise.
export NOISE_ACTOR_DIR=""
#"/hkfs/work/workspace/scratch/eu3660-dsrl/DSRL_pi0_Libero/dsrl_pi05_libero_90_task29_2026_08_01_11_15_21_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps/checkpoint2000000"
# /hkfs/work/workspace/scratch/eu3660-dsrl/DSRL_pi0_Libero/dsrl_pi05_libero_90_task29_2026_07_31_08_06_09_0000--s-0_chunkrewardcriticactor_mlp_7dims/checkpoint2000000
# /hkfs/work/workspace/scratch/eu3660-dsrl/DSRL_pi0_Libero/dsrl_pi05_libero_90_task38_2026_07_30_17_24_53_0000--s-0_chunkrewardcriticactor_mlp_7dims/checkpoint1600000
# /hkfs/work/workspace/scratch/eu3660-dsrl/DSRL_pi0_Libero/dsrl_pi05_libero_90_task64_2026_07_30_15_25_17_0000--s-0_chunkrewardcriticactor_mlp_7dims/checkpoint1000000
# /hkfs/work/workspace/scratch/eu3660-dsrl/DSRL_pi0_Libero/dsrl_pi05_libero_90_task38_2026_08_04_08_04_14_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps_residualmlp/checkpoint1800000

# Base checkpoint (required).  Set checkpoint_ft to enable base-vs-finetuned comparison.
export CHECKPOINT_FT=pi05_libero          # e.g. pi05_libero

python3 examples/eval_noise_7dims.py \
    --checkpoint "${CHECKPOINT_FT}" \
    --noise_actor_dirs "${NOISE_ACTOR_DIR}" \
    --task_id 38 \
    --num_rollouts 1 \
    --max_timesteps 400 \
    --filename "noise_eval_7dims_${CHECKPOINT_FT}_libero90_task38_residual" \
    --no-cross_actor
