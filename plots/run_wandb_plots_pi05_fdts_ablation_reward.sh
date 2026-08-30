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
  "dsrl_pi05_libero_90_task59_2026_06_19_16_13_16_0000--s-0_baseline"
  "dsrl_pi05_libero_90_task59_2026_07_07_09_46_10_0000--s-0_baseline"
  "dsrl_pi05_libero_90_task59_2026_07_08_08_19_37_0000--s-0_chunkreward_singleactorcritic_mlp_10replan_criticdim512_256_128"
  ""
  "dsrl_pi05_libero_90_task59_2026_07_07_14_13_06_0000--s-0_chunkycriticactor_mlp_singlereward_10replan_criticdim512_256_128"
  ""
  "dsrl_pi05_libero_90_task59_2026_07_07_09_46_10_0000--s-0_chunkrewardcriticactor_mlp_10replan_criticdim512_256_128"
  ""
)
task1_title="Task 59"
task1_sft="0.34"


# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  "DSRL-SAC (Baseline)"
  "DSRL-SAC (Chunk Reward)"
  "FDTS-Chunk MLP Actor + MLP Critic (Single Reward)"
  "FDTS-Chunk MLP Actor + MLP Critic (Chunk Reward)"
)

all_task_ids=(
  "${task1_ids[@]}"
)

labels=()
for task_ids_ref in task1_ids; do
  declare -n _ids="$task_ids_ref"
  for i in "${!_ids[@]}"; do
    method_idx=$(( i / seeds_per_method ))
    labels+=("${_ids[$i]}=${method_labels[$method_idx]}")
  done
done

# --- Plot settings ---
metric="evaluation/success_rate"
x_axis="_step"
suptitle="FDTS-Chunk Ablation on \$\pi_{0.5}\$ LIBERO-90"
output="plots/plots_two_tasks/pi05_libero90_fdts_ablation_reward.svg"
show_plot=0
ymin=0.0
ymax=1.0
ema_halflife=25000
clip_to_shortest_run=0
dim_colors=0
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
  --task-titles "${task1_title}" \
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
  --dim-colors "${dim_colors}" \
  --sft-baselines "${task1_sft}" \
  "${extra_args[@]}"
