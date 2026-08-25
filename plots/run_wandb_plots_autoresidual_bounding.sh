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
seeds_per_method=1

task1_ids=(
  "dsrl_pi05_libero_90_task28_2026_08_22_04_00_17_0000--s-0_baseline" #_10replan
  "dsrl_pi05_libero_90_task28_2026_08_05_09_10_55_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  "dsrl_pi05_libero_90_task28_2026_08_07_22_46_43_0000--s-0_criticgpt_aractor_diff_mean"
  "dsrl_pi05_libero_90_task28_2026_08_22_08_23_09_0000--s-0_criticgpt_aractor_diff_0.1bound"
  "dsrl_pi05_libero_90_task28_2026_08_22_04_00_17_0000--s-0_criticgpt_aractor_diff_mean_bounded0.2"
)
task1_title="Task 28"
task1_sft="0.49"  # SFT baseline success rate for this task (0-1 scale); leave "" to skip

# --- Task 1 ---
task2_ids=(
  "dsrl_pi05_libero_90_task43_2026_06_28_04_53_53_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task43_2026_08_05_09_15_05_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  "dsrl_pi05_libero_90_task43_2026_08_07_22_46_43_0000--s-0_criticgpt_aractor_diff_mean"
  "dsrl_pi05_libero_90_task43_2026_08_22_13_02_49_0000--s-0_criticgpt_aractor_diff_0.1bound"
  "dsrl_pi05_libero_90_task43_2026_08_22_04_00_17_0000--s-0_criticgpt_aractor_diff_mean_bounded0.2"
)
task2_title="Task 43"
task2_sft="0.14"  # SFT baseline success rate for this task (0-1 scale); leave "" to skip

# --- Task 2: seeds_per_method × 4 methods ---
task3_ids=(
  # DSRL-SAC Baseline (seed 1 & 2)
  "dsrl_pi05_libero_90_task59_2026_08_22_03_01_52_0000--s-0_baseline" #_10replan
  # residual without freezing
  "dsrl_pi05_libero_90_task59_2026_07_12_09_53_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  "dsrl_pi05_libero_90_task59_2026_08_08_16_10_19_0000--s-0_criticgpt_aractor_diff_mean"
  "dsrl_pi05_libero_90_task59_2026_08_22_13_02_49_0000--s-0_criticgpt_aractor_diff_0.1bound"
  ""
) 
task3_title="Task 59"
task3_sft="0.34"  # SFT baseline success rate for this task (0-1 scale); leave "" to skip


task4_ids=(
  # DSRL-SAC Baseline (seed 1 & 2)
  "dsrl_pi05_libero_90_task60_2026_06_28_04_53_54_0000--s-0_baseline_10numqs_10replan"
  # residual without freezing
  "dsrl_pi05_libero_90_task60_2026_08_04_16_38_56_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  "dsrl_pi05_libero_90_task60_2026_08_08_16_11_50_0000--s-0_criticgpt_aractor_diff_mean"
  "dsrl_pi05_libero_90_task60_2026_08_22_08_23_09_0000--s-0_criticgpt_aractor_diff_0.1bound"
  "dsrl_pi05_libero_90_task60_2026_08_22_04_22_12_0000--s-0_criticgpt_aractor_diff_mean_bounded0.2"
) 
task4_title="Task 60"
task4_sft="0.49"  # SFT baseline success rate for this task (0-1 scale); leave "" to skip


# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  "DSRL-SAC (Baseline)"
  "FDTS-Residual-Noise"
  "FDTS-Residual-Distribution"
  "FDTS-Residual-Noise (bounded)"
  "FDTS-Residual-Distribution (bounded)"
)

all_task_ids=(
  "${task1_ids[@]}"
  "${task2_ids[@]}"
  "${task3_ids[@]}"
  "${task4_ids[@]}"
)

sft_baselines=(
  "${task1_sft}"
  "${task2_sft}"
  "${task3_sft}"
  "${task4_sft}"
)

labels=()
for task_ids_ref in task1_ids task2_ids task3_ids task4_ids; do
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
output="plots/plots_multi_tasks/pi05_libero90_autoresidual_bounded.svg"
show_plot=0
ymin=0.0
ymax=1.0
ema_halflife=50000
clip_to_shortest_run=0

export LD_LIBRARY_PATH=
export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

extra_args=()
[[ ${show_plot} -eq 1 ]] && extra_args+=(--show)
[[ ${clip_to_shortest_run} -eq 1 ]] && extra_args+=(--clip-to-shortest-run)

labels_str=$(IFS=','; echo "${labels[*]}")

python3 plots/wandb_plots_residual_auto.py \
  --project "${proj_name}" \
  --entity "${entity}" \
  --identifiers "${all_task_ids[@]}" \
  --task-titles "${task1_title}" "${task2_title}" "${task3_title}" "${task4_title}" \
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
  --max-steps 800000 \
  --sft-baselines "${sft_baselines[@]}" \
  "${extra_args[@]}"
