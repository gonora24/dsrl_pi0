#!/bin/bash
# set -euo pipefail

# --- W&B settings ---
proj_name="DSRL_pi05_Libero"
entity="noras-masterarbeit"

# ---------------------------------------------------------------------------
# seeds_per_method: how many runs (seeds) represent each method.
#   Set to 1 for single-seed runs (original behaviour).
#   Set to 2 (or more) to average seeds — runs sharing the same label are
#   averaged by seaborn with a shaded confidence band (see --errorbar).
# Each task block must contain  seeds_per_method × 4  run IDs, ordered so
# that seeds of the same method are listed consecutively.
# ---------------------------------------------------------------------------
seeds_per_method=2

# --- Task 1 ---
task1_ids=(
  "dsrl_pi05_libero_90_task31_2026_06_28_05_38_27_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task31_2026_06_28_11_00_03_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task31_2026_06_29_02_28_09_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task31_2026_06_29_00_05_41_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task31_2026_06_29_06_40_55_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
  "dsrl_pi05_libero_90_task31_2026_06_29_08_42_40_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
  "dsrl_pi05_libero_90_task31_2026_08_04_14_47_42_0000--s-0_criticgpt_aractor_10replan"
  ""
)
task1_title="Task 31"
task1_sft="0.14"  # SFT baseline success rate for this task (0-1 scale); leave "" to skip

# --- Task 2: seeds_per_method × 4 methods ---
task2_ids=(
  # DSRL-SAC Baseline (seed 1 & 2)
  "dsrl_pi05_libero_90_task38_2026_06_28_11_00_03_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task38_2026_06_28_04_37_47_0000--s-0_baseline_10numqs_10replan"
  # Chunked MLP Critic + MLP Actor (seed 1 & 2)
  "dsrl_pi05_libero_90_task38_2026_06_28_11_06_19_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task38_2026_06_29_00_36_11_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  # Chunked Critic Transformer + MLP Actor (seed 1 & 2)
  "dsrl_pi05_libero_90_task38_2026_06_29_06_43_09_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
  "dsrl_pi05_libero_90_task38_2026_06_29_08_55_15_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
  # Chunked Critic Transformer + AR Actor (seed 1 & 2)
  "dsrl_pi05_libero_90_task38_2026_08_03_22_24_57_0000--s-0_criticgpt_aractor_10replan"
  ""
) #dsrl_pi05_libero_90_task38_2026_08_03_22_24_57_0000--s-0_criticgpt_aractor_10replan
task2_title="Task 38"
task2_sft="0.61"  # SFT baseline success rate for this task (0-1 scale); leave "" to skip

# --- Task 3 ---
task3_ids=(
  # DSRL-SAC Baseline
  "dsrl_pi05_libero_90_task43_2026_06_28_04_53_53_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task43_2026_06_28_11_00_03_0000--s-0_baseline_10numqs_10replan"
  # Chunked MLP Critic + MLP Actor
  "dsrl_pi05_libero_90_task43_2026_06_29_01_06_06_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task43_2026_06_28_11_06_19_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  # Chunked Critic Transformer + MLP Actor
  "dsrl_pi05_libero_90_task43_2026_06_29_06_48_16_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
  "dsrl_pi05_libero_90_task43_2026_06_29_08_55_16_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
  # Chunked Critic Transformer + AR Actor
  "dsrl_pi05_libero_90_task43_2026_08_03_22_08_28_0000--s-0_criticgpt_aractor_10replan"
  ""
) #dsrl_pi05_libero_90_task43_2026_08_03_22_08_28_0000--s-0_criticgpt_aractor_10replan
task3_title="Task 43"
task3_sft="0.14"  # SFT baseline success rate for this task (0-1 scale); leave "" to skip

# --- Task 4 ---
task4_ids=(
  "dsrl_pi05_libero_90_task64_2026_06_28_05_44_43_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task64_2026_06_28_04_53_13_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task64_2026_06_29_05_17_14_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task64_2026_06_28_11_05_20_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task64_2026_06_29_07_04_06_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
  "dsrl_pi05_libero_90_task64_2026_06_29_09_15_53_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
  "dsrl_pi05_libero_90_task64_2026_08_03_22_08_28_0000--s-0_criticgpt_aractor_10replan"
  ""
) #dsrl_pi05_libero_90_task64_2026_08_03_22_08_28_0000--s-0_criticgpt_aractor_10replan
task4_title="Task 64"
task4_sft="0.42"  # SFT baseline success rate for this task (0-1 scale); leave "" to skip

# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method  (i / seeds_per_method)  (integer division).  Seeds of the same
# method share an identical label so that seaborn averages them together.
# ---------------------------------------------------------------------------
method_labels=(
  "DSRL-SAC (Baseline)"
  "FDTS-Chunk MLP Actor + MLP Critic"
  "FDTS-Chunk MLP Actor + Transformer Critic"
  "FDTS-Chunk AR Transformer Actor + Transformer Critic"
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
suptitle="FDTS-Chunk on \$\pi_{0.5}\$ LIBERO-90"
output="plots/plots_multi_tasks/pi05_libero90_multi_task_success_chunkedsac_fdts.svg"
show_plot=0
ymin=0.0
ymax=1.0
ema_halflife=25000
clip_to_shortest_run=0
max_steps=1000000

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
  --max-steps "${max_steps}" \
  --sft-baselines "${sft_baselines[@]}" \
  "${extra_args[@]}"
