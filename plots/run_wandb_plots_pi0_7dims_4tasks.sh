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
  "dsrl_pi0_libero_90_task29_2026_08_12_15_54_29_0000--s-0_baseline"
  ""
  "dsrl_pi0_libero_90_task29_2026_08_12_15_54_25_0000--s-0_baseline_7dims"
  ""
)
task1_title="Task 29"

task2_ids=(
  "dsrl_pi0_libero_90_task33_2026_08_02_14_39_26_0000--s-0_baseline"
  ""
  "dsrl_pi0_libero_90_task33_2026_08_02_12_54_10_0000--s-0_baseline_7dims"
  ""
)
task2_title="Task 33"

# --- Task 1 ---
task3_ids=(
  "dsrl_pi0_libero_90_task38_2026_08_02_13_18_03_0000--s-0_baseline"
  ""
  "dsrl_pi0_libero_90_task38_2026_08_02_12_57_05_0000--s-0_baseline_7dims"
  ""
)
task3_title="Task 38"

task4_ids=(
  "dsrl_pi0_libero_90_task47_2026_08_12_15_32_54_0000--s-0_baseline"
  ""
  "dsrl_pi0_libero_90_task47_2026_08_12_15_32_54_0000--s-0_baseline_7dims"
  ""
)
task4_title="Task 47"



# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  "DSRL-SAC (Baseline)"
  "RDS"
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
suptitle="\$\pi_0\$ LIBERO-90"
output="plots/plots_multi_tasks/pi0_libero90_7dims.svg"
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

python3 plots/wandb_plots_multi_tasks.py \
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
  --max-steps 1000000 \
  --dim-colors 1 \
  "${extra_args[@]}"
