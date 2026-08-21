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
  "dsrl_pi05_libero_90_task29_2026_07_31_08_04_03_0000--s-0_baseline_7dims"
  "dsrl_pi05_libero_90_task29_2026_08_12_13_18_25_0000--s-0_baseline_7dims"
  "dsrl_pi05_libero_90_task29_2026_08_01_11_15_21_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps"
  "dsrl_pi05_libero_90_task29_2026_08_02_15_14_37_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps"
  "dsrl_pi05_libero_90_task29_2026_08_16_17_41_03_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task29_2026_08_03_19_53_57_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps_residualmlp"
  ""
)
task1_title="Task 29"

# --- Task 1 ---
task2_ids=(
  "dsrl_pi05_libero_90_task38_2026_07_30_08_01_40_0000--s-0_baseline_only7dims"
  "dsrl_pi05_libero_90_task38_2026_07_31_08_02_05_0000--s-0_baseline_7dims"
  "dsrl_pi05_libero_90_task38_2026_08_02_09_30_38_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps"
  ""
  "dsrl_pi05_libero_90_task38_2026_08_16_17_19_53_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task38_2026_08_04_08_04_14_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps_residualmlp"
  ""
)
task2_title="Task 38"



# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  "RDS"
  "GT-RDS"
  "GT-RDS-Residual-Noise"
  "GT-RDS-Residual-Mean-MLP"
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
suptitle="\$\pi_{0.5}\$ LIBERO-90"
output="plots/plots_two_tasks/pi05_libero90_7dims_diff_gtrds.svg"
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
  --dim-colors 1 \
  "${extra_args[@]}"
