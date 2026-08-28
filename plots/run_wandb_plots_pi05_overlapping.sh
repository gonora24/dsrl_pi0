#!/bin/bash
# set -euo pipefail

# --- W&B settings ---
proj_name="DSRL_pi0_Libero"
entity="noragorhan-karlsruhe-institute-of-technology"

# ---------------------------------------------------------------------------
# seeds_per_method: how many runs (seeds) represent each method.
#   Set to 1 for single-seed runs (original behaviour).
#   Set to 2 (or more) to average seeds — runs sharing the same label are
#   averaged by seaborn with a shaded confidence band (see --errorbar).
# Each task block must contain  seeds_per_method × 4  run IDs, ordered so
# that seeds of the same method are listed consecutively.
# ---------------------------------------------------------------------------
seeds_per_method=2

task1_ids=(
  "dsrl_pi05_libero_90_task28_2026_07_06_17_08_06_0000--s-0_baseline"
  "dsrl_pi05_libero_90_task28_2026_08_22_04_00_17_0000--s-0_baseline"
  "dsrl_pi05_libero_90_task28_2026_06_21_10_16_01_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task28_2026_06_20_01_12_06_0000--s-0_chunkrewardcriticactor_mlp_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task28_chunk_2026_08_10_17_29_23_0000--s-0_chunkrewardcriticactor_mlp_realactionchunking"
  "dsrl_pi05_libero_90_task28_chunk_2026_07_31_16_36_41_0000--s-0_chunkrewardcriticactor_mlp_realactionchunking"
)
task1_title="Task 28"
task1_sft="0.49"

# --- Task 1 ---
task2_ids=(
  "dsrl_pi05_libero_90_task59_2026_08_22_03_01_52_0000--s-0_baseline"
  "dsrl_pi05_libero_90_task59_2026_08_24_11_41_34_0000--s-0_baseline_10replan"
  "dsrl_pi05_libero_90_task59_2026_07_07_09_46_10_0000--s-0_chunkrewardcriticactor_mlp_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task59_2026_07_08_08_19_37_0000--s-0_chunkreward_singleactorcritic_mlp_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task59_2026_07_30_16_48_55_0000--s-0_chunkrewardcriticactor_mlp_realactionchunking"
  "dsrl_pi05_libero_90_task59_2026_07_17_13_32_50_0000--s-0_chunkrewardcriticactor_mlp_realactionchunking"
)
task2_title="Task 59"
task2_sft="0.34"



# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  "DSRL-SAC (Baseline)"
  "FDTS-Chunk MLP (non-overlapping)"
  "FDTS-Chunk MLP (overlapping)"
)

all_task_ids=(
  "${task1_ids[@]}"
  "${task2_ids[@]}"
)

sft_baselines=(
  "${task1_sft}"
  "${task2_sft}"
)
labels=()
for task_ids_ref in task1_ids task2_ids; do
  declare -n _ids="$task_ids_ref"
  for i in "${!_ids[@]}"; do
    method_idx=$(( i / seeds_per_method ))
    labels+=("${_ids[$i]}=${method_labels[$method_idx]}")
  done
done

# --- Plot settings ---
metric="evaluation/success_rate"
x_axis="_step"
suptitle="\$\pi_{0.5}\$ LIBERO-90"
output="plots/plots_two_tasks/pi05_libero90_overlapping.svg"
show_plot=0
ymin=0.0
ymax=1.0
ema_halflife=25000
clip_to_shortest_run=0

export LD_LIBRARY_PATH=
export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

extra_args=()
[[ ${show_plot} -eq 1 ]] && extra_args+=(--show)
[[ ${clip_to_shortest_run} -eq 1 ]] && extra_args+=(--clip-to-shortest-run)

labels_str=$(IFS=','; echo "${labels[*]}")

python3 plots/wandb_plots_two_tasks.py \
  --project "${proj_name}" \
  --entity "${entity}" \
  --identifiers "${all_task_ids[@]}" \
  --task-titles "${task1_title}" "${task2_title}" \
  --runs-per-task "${#task1_ids[@]}" \
  --metric "${metric}" \
  --x-axis "${x_axis}" \
  --ymin "${ymin}" \
  --ymax "${ymax}" \
  --ema-halflife "${ema_halflife}" \
  --output "${output}" \
  --labels "${labels_str}" \
  --suptitle "${suptitle}" \
  --errorbar ci \
  --max-steps 1000000 \
  --dim-colors 0 \
  --sft-baselines "${sft_baselines[@]}" \
  "${extra_args[@]}"
