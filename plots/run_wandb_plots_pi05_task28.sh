#!/bin/bash
# set -euo pipefail

# --- W&B settings ---
proj_name="DSRL_pi0_Libero"
entity="noragorhan-karlsruhe-institute-of-technology"  # leave empty to use WANDB_TEAM from jaxrl2/utils/wandb_config.py

# --- Runs to compare (W&B run ids / experiment ids, or local offline run dirs) ---
identifiers=(
  "dsrl_pi05_libero_90_task28_2026_06_21_04_44_15_0000--s-0_baseline_10numqs_5replan"
  "dsrl_pi05_libero_90_task28_2026_06_27_13_25_52_0000--s-0_chunkrewardcriticactor_mlp_10replan_criticdim512_256_128_clip_marginlp"
  "dsrl_pi05_libero_90_task28_2026_07_01_11_02_41_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_criticdim512"
  "dsrl_pi05_libero_90_task28_2026_06_23_05_18_35_0000--s-0_criticgpt_aractor_10replan"
)

# --- Optional legend labels (comma-separated run_id=Label pairs; leave empty for auto labels) ---
labels=(
  "${identifiers[0]}=DSRL-SAC Baseline"
  "${identifiers[1]}=Chunked MLP Critic + MLP Actor"
  "${identifiers[2]}=Chunked Critic Transformer + MLP Actor"
  "${identifiers[3]}=Chunked Critic Transformer + Autoregressive Actor Transformer"
)

# --- Plot settings ---
metric="evaluation/success_rate"
x_axis="_step"
title="$\pi_{0.5}$ LIBERO-90 Task 28"
output="plots/pi05_libero90_task28_success.svg"
show_plot=0   # set to 1 to display interactively
ymin=0.0
ymax=1.0
ema_halflife=25000  # training steps; set to 0 to disable smoothing
clip_to_shortest_run=0  # set to 0 to plot until the longest run ends

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

# --show and --clip-to-shortest-run are boolean flags (store_true), so only pass them when enabled
extra_args=()
[[ ${show_plot} -eq 1 ]] && extra_args+=(--show)
[[ ${clip_to_shortest_run} -eq 1 ]] && extra_args+=(--clip-to-shortest-run)

# --labels expects a single comma-separated string, not multiple shell arguments
labels_str=$(IFS=','; echo "${labels[*]}")

# --- Environment (adjust if needed) ---
python3 plots/wandb_plots.py \
  --project ${proj_name} \
  --entity ${entity} \
  --identifiers "${identifiers[@]}" \
  --metric ${metric} \
  --x-axis ${x_axis} \
  --ymin ${ymin} \
  --ymax ${ymax} \
  --ema-halflife ${ema_halflife} \
  --output ${output} \
  --labels "${labels_str}" \
  --title "${title}" \
  "${extra_args[@]}"