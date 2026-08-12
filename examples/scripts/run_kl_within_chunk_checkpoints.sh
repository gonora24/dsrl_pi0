#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=kl_within_chunk_ckpts

# module load devel/miniforge
# conda deactivate
# conda activate dsrl_pi0

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

# ---------------------------------------------------------------------------
# Noise actor checkpoints to compare — add / remove entries as needed.
# Task ID is auto-extracted from the path via regex task(\d+) if not passed
# explicitly.  All checkpoints below should correspond to the same task.
# ---------------------------------------------------------------------------

CKPT_BASE=/hkfs/work/workspace/scratch/eu3660-dsrl/DSRL_pi0_Libero

CKPT1="${CKPT_BASE}/dsrl_pi05_libero_90_task29_2026_07_31_08_06_09_0000--s-0_chunkrewardcriticactor_mlp_7dims/checkpoint2000000"
CKPT2="${CKPT_BASE}/dsrl_pi05_libero_90_task38_2026_07_30_17_24_53_0000--s-0_chunkrewardcriticactor_mlp_7dims/checkpoint1600000"
CKPT3="${CKPT_BASE}/dsrl_pi05_libero_90_task43_2026_07_30_10_12_54_0000--s-0_chunkrewardcriticactor_mlp_7dims/checkpoint1000000"
CKPT4="${CKPT_BASE}/dsrl_pi05_libero_90_task64_2026_07_30_15_25_17_0000--s-0_chunkrewardcriticactor_mlp_7dims/checkpoint1000000"

# Human-readable labels for the bar chart (one per checkpoint, same order).
# Remove --labels to fall back to auto-generated names.
LABELS=("Task 29" "Task 38" "Task 43" "Task 64")

# Base pi0 policy key
CHECKPOINT=pi05_libero

python3 examples/plot_kl_within_chunk_checkpoints.py \
    --checkpoint "${CHECKPOINT}" \
    --noise_actor_dirs "${CKPT1}" "${CKPT2}" "${CKPT3}" "${CKPT4}" \
    --labels "${LABELS[@]}" \
    --num_rollouts 20 \
    --max_timesteps 400 \
    --filename "kl_multi_ckpt_${CHECKPOINT}_libero90_chunked_noerrorbar" \
    --out_dir "plots/plots/noise_7dims/" \
    --random_baseline_samples 2000 \
    --errorbar 0
