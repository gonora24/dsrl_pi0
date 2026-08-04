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
  "dsrl_pi05_libero_90_task28_2026_06_18_07_59_34_0000--s-0_baseline"
  ""
  ""
  ""
    # residual with freezing
  "dsrl_pi05_libero_90_task28_2026_07_29_17_18_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  ""
)
task1_title="Task 28"

# --- Task 1 ---
task2_ids=(
  "dsrl_pi05_libero_90_task43_2026_07_30_07_38_49_0000--s-0_baseline"
  ""
  ""
  ""
    # residual with freezing
  "dsrl_pi05_libero_90_task43_2026_07_29_17_18_34_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  ""
)
task2_title="Task 43"

# --- Task 2: seeds_per_method × 4 methods ---
task3_ids=(
  # DSRL-SAC Baseline (seed 1 & 2)
  "dsrl_pi05_libero_90_task59_2026_06_19_16_13_16_0000--s-0_baseline"
  ""
  # residual without freezing
  "dsrl_pi05_libero_90_task59_2026_07_12_09_53_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  ""
  # residual with freezing
  "dsrl_pi05_libero_90_task59_2026_07_29_17_18_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  ""
) 
task3_title="Task 59"


task4_ids=(
  # DSRL-SAC Baseline (seed 1 & 2)
  "dsrl_pi05_libero_90_task60_2026_07_30_07_39_33_0000--s-0_baseline_only7dims"
  ""
  # residual without freezing
  ""
  ""
  # residual with freezing
  "dsrl_pi05_libero_90_task60_2026_07_29_17_18_38_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  ""
) 
task4_title="Task 60"


# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  "DSRL-SAC (Baseline)"
  "Autoregressive Residual Actor Transformer (no freezing)"
  "Autoregressive Residual Actor Transformer (with freezing)"
)

all_task_ids=(
  "${task1_ids[@]}"
  "${task2_ids[@]}"
  "${task3_ids[@]}"
  "${task4_ids[@]}"
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
output="plots/plots_two_tasks/pi05_libero90_autoresidual_differentcolors.svg"
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
  "${extra_args[@]}"
