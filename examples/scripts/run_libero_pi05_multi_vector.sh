#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=dsrl_pi05_90_59_multivec

# Multi-vector noise mode: the SAC actor predicts N independent 32-d noise
# vectors; each is tiled K times before being passed to pi0.5 so the full
# action horizon (10) is covered.
#
# Current setting:  N=2, K=5  ->  [v1]*5 + [v2]*5 = horizon 10  (exact)
#
# Other combinations to try:
#   --num_noise_vectors 5  --noise_repeats_per_vector 2  -> 10  (exact)
#   --num_noise_vectors 3  --noise_repeats_per_vector 3  ->  9  (last vec padded once)
#   --num_noise_vectors 3  --noise_repeats_per_vector 4  -> 12  (truncated to 10)
#   --num_noise_vectors 4  --noise_repeats_per_vector 3  -> 12  (truncated to 10)
#   --num_noise_vectors 10 --noise_repeats_per_vector 1  -> 10  (equivalent to chunky)

module load devel/miniforge
conda deactivate
conda activate dsrl_pi0

export WANDB_API_KEY='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'
export WANDB_EMAIL='noragorhan@gmail.com'
export WANDB_USERNAME='noragorhan'
export WANDB_TEAM='noragorhan-karlsruhe-institute-of-technology'
proj_name=DSRL_pi0_Libero
device_id=0
wandb_mode=online  # online or offline
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
--suffix multivec_N5_K2_mlp \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 2000000  \
--eval_interval 10000 \
--checkpoint_interval 200000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 10 \
--hidden_dims 128 128 128 \
--critic_hidden_dims 512 256 128 \
--libero_suite "libero_90" \
--libero_task_id 59 \
--pi0_checkpoint pi05_libero \
--chunk_reward 0 \
--overlap_transitions 0 \
--num_noise_vectors 5 \
--noise_repeats_per_vector 2 \
--marginalize_logprobs 1 \
