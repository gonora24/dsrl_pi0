#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=dsrl_pi05_90_59_frozen_residual

# Frozen-baseline residual mode:
#
#   The warm-started MLP (restored from a baseline checkpoint) is frozen via
#   stop_gradient and always predicts a single 32-dim noise vector, identical
#   to the baseline.  A second trainable residual MLP adds corrections to
#   residual_n_vectors extra copies of that frozen vector.
#
#   Total action dim = (1 + residual_n_vectors) * 32.
#   Default (residual_n_vectors=1): 64-dim = 2 noise vectors.
#     vector 0: v_frozen  (baseline, no gradient)
#     vector 1: v_frozen + v_residual  (learned correction)
#
#   Expansion to the pi0 horizon uses noise_repeats_per_vector as usual.
#   Example: residual_n_vectors=1, noise_repeats_per_vector=5 -> 10-step horizon.

module load devel/miniforge
conda deactivate
conda activate dsrl_pi0

export WANDB_API_KEY='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'
export WANDB_EMAIL='noragorhan@gmail.com'
export WANDB_USERNAME='noragorhan'
export WANDB_TEAM='noragorhan-karlsruhe-institute-of-technology'
proj_name=DSRL_pi0_Libero
device_id=0
wandb_mode=offline  # online or offline
export WANDB_MODE=${wandb_mode}

export OUTPUT_DIR=/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero

export PYTHONPATH="${PYTHONPATH}:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/LIBERO

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

pip install mujoco==3.3.1
pip install "transformers==4.53.2"

python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi05_libero_90_task59 \
--suffix frozen_residual_K1_R5 \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 1000000  \
--eval_interval 10000 \
--checkpoint_interval 1000000 \
--initialize_weights_from /pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero/dsrl_pi05_libero_90_task59_2026_07_07_09_46_10_0000--s-0_baseline/checkpoint1000000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 5 \
--hidden_dims 128 128 128 \
--critic_hidden_dims 512 256 128 \
--libero_suite "libero_90" \
--libero_task_id 59 \
--pi0_checkpoint pi05_libero \
--chunk_reward 0 \
--overlap_transitions 0 \
--use_frozen_baseline_residual 1 \
--residual_n_vectors 1 \
--num_noise_vectors 2 \
--noise_repeats_per_vector 5 \
--marginalize_logprobs 0 \
