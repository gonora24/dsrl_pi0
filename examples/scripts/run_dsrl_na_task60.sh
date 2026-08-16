#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=dsrl_na_singleactorcritic_task60

module load devel/miniforge 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate dsrl_pi0
device_id=0

export PYTHONPATH="${PYTHONPATH}:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/LIBERO

proj_name=DSRL_NA_Libero
wandb_mode=online   # online or offline
export WANDB_API_KEY='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'
export WANDB_EMAIL='noragorhan@gmail.com'
export WANDB_USERNAME='noragorhan'
export WANDB_TEAM='noragorhan-karlsruhe-institute-of-technology'
export WANDB_MODE=${wandb_mode}
export OUTPUT_DIR="/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_NA_Libero"

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

python3 examples/launch_train_sim.py \
--algorithm dsrl_na \
--trajectory_hdf5_path /pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_NA_trajectories/libero_90_task_60/trajectories_task60_7000_combined.hdf5 \
--env libero \
--prefix dsrl_na_libero_90_task60 \
--suffix singleactorcritic_offline \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--backup_entropy 1 \
--seed 0 \
--max_steps 1000000 \
--num_offline_steps 1000000 \
--eval_interval 500 \
--checkpoint_interval 2500 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 10 \
--hidden_dims 128 128 128 \
--num_qs 2 \
--online_buffer_size 1000000 \
--libero_suite "libero_90" \
--libero_task_id 60 \
--pi0_checkpoint pi05_libero \
--pi0_microbatch_size 0 \
--chunk_reward 1 \
--use_chunky_actor_critic 0 \
--critic_hidden_dims 128 128 128 \
--transformer_n_embd 512 \
--transformer_n_head 4 \
--transformer_n_layer 3 \
--transformer_weight_norm 1 \
--transformer_use_bias 0 \
--use_noise_mapping_distill 1 \
--noise_mapping_distill_path /pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_NA_noise_action_mapping/noise_action_mapping_task60_repeat.h5 \
--freeze_latent_models 35000 \
--only_predict_dims_until 7 \

# set online steps as difference between max_steps and num_offline_steps
# backup entropy important, used in original dsrl-na 
# chunk reward is used for outer real action space, not for noisy action space
# use_chunky_actor_critic is used for noisy action space

# --critic_hidden_dims 512 512 512 512 \