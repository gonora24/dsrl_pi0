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
export NOISE_ACTOR_DIR=
#"/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero/dsrl_pi05_libero_90_task59_2026_07_07_09_46_10_0000--s-0_baseline/checkpoint1000000"

# Finetuned checkpoint (required).  Set checkpoint_ft to enable base-vs-finetuned comparison.
export CHECKPOINT_FT=pi05_libero          # e.g. pi05_libero

# Set to 1 to switch to gripper-close event mode: instead of the usual
# whole-trajectory-averaged output, roll out a trajectory and, whenever the
# gripper starts closing (action[num_actions-1] crosses
# GRIPPER_CLOSE_THRESHOLD), save a chunk-averaged Jacobian/influence for that
# event under plots/plots/jacobian/<filename>/. Requires --run_over_trajectory 1
# and --num_actions to be set.
export GRIPPER_CLOSE_MODE=1
export GRIPPER_CLOSE_THRESHOLD=0.5

python3 jaxrl2/tests/jacobian.py \
    --checkpoint "${CHECKPOINT_FT}" \
    --libero_suite libero_90 \
    --N 100 \
    --seed 0 \
    --action_idx 0 \
    --noise_idx 0 \
    --task_id 64 \
    --num_actions 7 \
    --run_over_trajectory 1 \
    --query_frequency 5 \
    --filename "jacobian_noise_action_dims_${CHECKPOINT_FT}_libero90_task64_trajectory10rollouts_averagechunk" \
    --title_str "Jacobian over trajectory of \$\pi_{0.5}\$ LIBERO-90 Task 64" \
    --num_rollouts 1 \
    --compute_influence 1 \
    --plot 1 \
    --average_over_chunk 1 \
    --gripper_close_mode "${GRIPPER_CLOSE_MODE}" \
    --gripper_close_threshold "${GRIPPER_CLOSE_THRESHOLD}"
