#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=dsrl_pi05_90_59_multivec_interp

# Multi-vector noise mode: the SAC actor predicts N independent 32-d noise
# vectors.  Two expansion modes are available:
#
#   Hard tiling (interpolate_noise_vectors 0):
#     Each vector is repeated K times: [v1]*K + [v2]*K + ...
#     N=2, K=5  ->  [v1]*5 + [v2]*5 = horizon 10  (exact)
#
#   Interpolation (interpolate_noise_vectors 1):
#     N vectors become evenly-spaced anchors at positions linspace(0, H-1, N);
#     piecewise linear interpolation fills the remaining H steps.
#     N=2  ->  v1 at t=0, v2 at t=9, smooth blend in between
#     N=3  ->  v1, blend, v2 (middle), blend, v3
#     N=5  ->  v1 at t=0, v5 at t=9, v2..v4 evenly spaced between them
#     noise_repeats_per_vector is ignored when interpolation is active.
#
# Other combinations to try (hard tiling):
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
--suffix multivec_N5_K2_mlp_warmstart \
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
--query_freq 10 \
--hidden_dims 128 128 128 \
--critic_hidden_dims 512 256 128 \
--libero_suite "libero_90" \
--libero_task_id 59 \
--pi0_checkpoint pi05_libero \
--chunk_reward 0 \
--overlap_transitions 0 \
--num_noise_vectors 2 \
--noise_repeats_per_vector 5 \
--marginalize_logprobs 0 \
