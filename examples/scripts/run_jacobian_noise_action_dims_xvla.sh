#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=jacobian_noise_action_dims_xvla

# module load devel/miniforge
# conda deactivate
# conda activate dsrl_pi0

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

# Optional: set to a PixelSACLearner checkpoint directory to use learned noise.
# Leave empty to use random Gaussian noise.
export NOISE_ACTOR_DIR=

# X-VLA checkpoint key: xvla_libero or xvla_base.
export CHECKPOINT=xvla_libero

# Set to 1 to switch to gripper-close event mode: instead of the usual
# whole-trajectory-averaged output, roll out a trajectory and, whenever the
# executed (7-dim) gripper action starts closing (crosses
# GRIPPER_CLOSE_THRESHOLD), save a chunk-averaged Jacobian/influence for that
# event under plots/plots/jacobian/<filename>/. Requires --run_over_trajectory 1.
export GRIPPER_CLOSE_MODE=0
export GRIPPER_CLOSE_THRESHOLD=0.5

# --query_frequency is intentionally omitted: X-VLA's absolute ee6d chunks
# must be executed in full before replanning, so jacobian_xvla.py always
# overrides it to the policy's action_horizon.
NOISE_ACTOR_ARGS=()
if [ -n "${NOISE_ACTOR_DIR}" ]; then
    NOISE_ACTOR_ARGS=(--noise_actor_dir "${NOISE_ACTOR_DIR}")
fi

python3 jaxrl2/tests/jacobian_xvla.py \
    --checkpoint "${CHECKPOINT}" \
    "${NOISE_ACTOR_ARGS[@]}" \
    --libero_suite libero_90 \
    --N 100 \
    --seed 0 \
    --action_idx 0 \
    --noise_idx 0 \
    --task_id 64 \
    --num_actions 10 \
    --run_over_trajectory 1 \
    --filename "jacobian_noise_action_dims_${CHECKPOINT}_libero90_task64_trajectory10rollouts_averagechunk" \
    --title_str "Jacobian over trajectory of XVLA LIBERO-90 Task 64" \
    --num_rollouts 10 \
    --compute_influence 1 \
    --plot 1 \
    --average_over_chunk 1 \
    --gripper_close_mode "${GRIPPER_CLOSE_MODE}" \
    --gripper_close_threshold "${GRIPPER_CLOSE_THRESHOLD}"
