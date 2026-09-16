#!/bin/bash
# set -euo pipefail

# --- W&B settings ---
proj_name="DSRL_NA_Libero"
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
  "dsrl_pi05_libero_90_task59_2026_08_23_08_03_43_0000--s-0_baseline_7dims_10replan"
  "dsrl_na_libero_90_task59_2026_08_04_21_09_04_0000--s-0_singleactorcritic_offline"
)
task1_title="Task 59"
task1_sft="0.34"

# --- Task 1 ---
task2_ids=(
  "dsrl_pi05_libero_90_task60_2026_08_22_04_00_17_0000--s-0_baseline"
  "dsrl_na_libero_90_task60_2026_08_20_15_38_26_0000--s-0_singleactorcritic_offline_nofreezing"
)
task2_title="Task 60"
task2_sft="0.49"


# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  "DSRL-SAC (Baseline)"
  "RDS-NA"
)

all_task_ids=(
  "${task1_ids[@]}"
  "${task2_ids[@]}"
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
suptitle="RDS-NA on \$\pi_{0.5}\$ LIBERO-90"
output="plots/plots_two_tasks/pi05_libero90_fdts_na.svg"
show_plot=0
ymin=0.0
ymax=1.0
ema_halflife=1000
clip_to_shortest_run=0
dim_colors=1
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
  --max-steps 30000 \
  --dim-colors "${dim_colors}" \
  --sft-baselines "${task1_sft}" "${task2_sft}" \
  "${extra_args[@]}"
