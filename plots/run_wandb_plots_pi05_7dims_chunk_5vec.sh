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
  # "dsrl_pi05_libero_90_task29_2026_08_23_08_05_46_0000--s-0_baseline_10replan"
  # "dsrl_pi05_libero_90_task29_2026_08_22_03_31_18_0000--s-0_baseline" # 10 replan
  "dsrl_pi05_libero_90_task29_2026_08_22_12_40_31_0000--s-0_baseline_7dims_10replan"
  "dsrl_pi05_libero_90_task29_2026_08_23_08_05_04_0000--s-0_baseline_7dims_10replan"
  "dsrl_pi05_libero_90_task29_2026_07_31_08_06_09_0000--s-0_chunkrewardcriticactor_mlp_7dims"
  "dsrl_pi05_libero_90_task29_2026_08_13_07_44_58_0000--s-0_chunkrewardcriticactor_mlp_7dims"
  "dsrl_pi05_libero_90_task29_2026_08_01_11_15_21_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps"
  "dsrl_pi05_libero_90_task29_2026_08_02_15_14_37_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps"
  "dsrl_pi05_libero_90_task29_2026_06_21_10_25_52_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task29_2026_06_21_10_16_01_0000--s-0_chunkrewardcriticactor_mlp_10numqs_5replan_criticdim512_256_128"
)
task1_title="Task 29"
task1_sft="0.50"


# --- Task 2 ---
task2_ids=(
  # "dsrl_pi05_libero_90_task38_2026_06_28_11_00_03_0000--s-0_baseline_10numqs_10replan"
  # "dsrl_pi05_libero_90_task38_2026_06_28_04_37_47_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task38_2026_08_22_12_51_08_0000--s-0_baseline_7dims_10replan"
  "dsrl_pi05_libero_90_task38_2026_08_23_07_36_43_0000--s-0_baseline_7dims_10replan"
  "dsrl_pi05_libero_90_task38_2026_07_30_17_24_53_0000--s-0_chunkrewardcriticactor_mlp_7dims"
  "dsrl_pi05_libero_90_task38_2026_08_13_07_44_58_0000--s-0_chunkrewardcriticactor_mlp_7dims"
  "dsrl_pi05_libero_90_task38_2026_08_02_09_30_38_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps"
  ""
  "dsrl_pi05_libero_90_task38_2026_06_29_03_07_24_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task38_2026_06_29_00_36_11_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
)
task2_title="Task 38"
task2_sft="0.61"

task3_ids=(
  # "dsrl_pi05_libero_90_task43_2026_06_28_05_42_04_0000--s-0_baseline_10numqs_10replan"
  # "dsrl_pi05_libero_90_task43_2026_06_28_11_00_03_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task43_2026_08_23_07_33_46_0000--s-0_baseline_7dims_10replan"
  "dsrl_pi05_libero_90_task43_2026_08_23_07_33_50_0000--s-0_baseline_7dims_10replan"
  "dsrl_pi05_libero_90_task43_2026_07_30_10_12_54_0000--s-0_chunkrewardcriticactor_mlp_7dims"
  "dsrl_pi05_libero_90_task43_2026_08_13_07_44_58_0000--s-0_chunkrewardcriticactor_mlp_7dims"
  "dsrl_pi05_libero_90_task43_2026_08_15_18_02_24_0000--s-0_chunkrewardcriticactor_mlp_5vecs_2reps"
  ""
  "dsrl_pi05_libero_90_task43_2026_06_29_03_52_19_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task43_2026_06_28_11_05_20_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
)
task3_title="Task 43"
task3_sft="0.14"
# task3_ids=(
#   "dsrl_pi05_libero_90_task59_2026_08_22_03_01_52_0000--s-0_baseline" #10 replan
#   "dsrl_pi05_libero_90_task59_2026_07_07_09_46_10_0000--s-0_baseline"
#   "dsrl_pi05_libero_90_task59_2026_08_22_12_55_47_0000--s-0_baseline_7dims_10replan"
#   "dsrl_pi05_libero_90_task59_2026_08_12_13_07_58_0000--s-0_baseline_7dims"
#   "dsrl_pi05_libero_90_task59_2026_07_30_10_06_45_0000--s-0_chunkrewardcriticactor_mlp_7dims"
#   ""
#   "dsrl_pi05_libero_90_task59_2026_08_03_17_16_09_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps"
#   ""
# )
# task3_title="Task 59"
# task3_sft="0.34"

task4_ids=(
  # "dsrl_pi05_libero_90_task64_2026_06_28_04_53_13_0000--s-0_baseline_10numqs_10replan"
  # "dsrl_pi05_libero_90_task64_2026_06_28_05_44_43_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task64_2026_08_22_13_14_27_0000--s-0_baseline_7dims_10replan"
  "dsrl_pi05_libero_90_task64_2026_08_23_07_33_50_0000--s-0_baseline_7dims_10replan"
  "dsrl_pi05_libero_90_task64_2026_07_30_15_25_17_0000--s-0_chunkrewardcriticactor_mlp_7dims"
  "dsrl_pi05_libero_90_task64_2026_07_30_17_25_51_0000--s-0_chunkrewardcriticactor_mlp_7dims"
  "dsrl_pi05_libero_90_task64_2026_08_02_09_34_44_0000--s-0_chunkrewardcriticactor_mlp_7dims_5vecs_2reps"
  ""
  "dsrl_pi05_libero_90_task64_2026_06_29_01_20_38_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task64_2026_06_28_11_05_17_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
)
task4_title="Task 64"
task4_sft="0.42"
# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  # "DSRL-SAC (Baseline)"
  "RDS"
  "T-RDS"
  "GT-RDS"
  "FDTS-Chunk MLP Actor + MLP Critic"
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

dashed_labels=(
  "FDTS-Chunk MLP Actor + MLP Critic"
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
suptitle="GT-RDS on \$\pi_{0.5}\$ LIBERO-90"
output="plots/plots_multi_tasks/pi05_libero90_7dims_chunk_gt_trds_nodsrlsac.svg"
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
  --sft-baselines "${sft_baselines[@]}" \
  --dashed-labels "${dashed_labels[@]}" \
  "${extra_args[@]}"
