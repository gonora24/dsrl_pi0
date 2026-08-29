#!/bin/bash
# set -euo pipefail

# ---------------------------------------------------------------------------
# Mirrors plots/run_wandb_plots_autoresidual_mean_mean.sh, but plots the
# residual-prediction norms inferred from checkpoints (via
# plots/eval_residual_norms.py / plots/run_eval_residual_norms.sh) instead of
# a metric already logged to W&B.
#
# Produces two figures:
#   - residual_norm_mean.svg   : ||Δ action|| (use_actor_diff) and
#                                ||Δ mean|| (use_actor_diff_mean) on one plot
#   - residual_norm_logstd.svg : ||Δ log_std|| for use_actor_diff_mean runs
#
# Run identifiers below are the same run names used in
# plots/run_eval_residual_norms.sh; each is resolved to
# plots/data/residual_norms/<run_name>.csv.
# ---------------------------------------------------------------------------

data_dir="plots/data/residual_abs_means"

task1_ids=(
  "dsrl_pi05_libero_90_task29_2026_08_16_17_41_03_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task29_2026_08_21_23_56_23_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec2reps"
  ""
  "dsrl_pi05_libero_90_task29_2026_08_21_04_24_00_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps_0.1bound"
  ""
  "dsrl_pi05_libero_90_task29_2026_08_20_13_29_55_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec2reps_0.1bound"
  ""
)
task1_title="Task 29"
task1_task_id=29
task1_sft="0.50"


# --- Task 2 ---
task2_ids=(
  "dsrl_pi05_libero_90_task38_2026_08_16_17_19_53_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task38_2026_08_19_00_37_07_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task38_2026_08_20_14_53_41_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps_0.1bound"
  ""
  "dsrl_pi05_libero_90_task38_2026_08_21_04_24_00_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec2reps_0.1bound"
  ""
)
task2_title="Task 38"
task2_sft="0.61"
task2_task_id=38

task3_ids=(
  "dsrl_pi05_libero_90_task59_2026_08_23_09_00_36_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task59_2026_08_23_09_00_36_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task59_2026_08_20_14_53_41_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps_0.1bound"
  ""
  "dsrl_pi05_libero_90_task59_2026_08_24_03_15_03_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec_2reps_bounded0.2"
  ""
)
task3_title="Task 59"
task3_sft="0.34"
task3_task_id=59

task4_ids=(
  "dsrl_pi05_libero_90_task64_2026_08_19_00_09_20_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task64_2026_08_19_00_37_08_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec_2reps"
  ""
  "dsrl_pi05_libero_90_task64_2026_08_22_00_25_00_0000--s-0_criticgpt_aractor_diff_7dims_5vec2reps_0.1bound"
  ""
  "dsrl_pi05_libero_90_task64_2026_08_20_13_25_02_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec2reps_0.1bound"
  ""
)
task4_title="Task 64"
task4_sft="0.42"
task4_task_id=64

# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method (i / seeds_per_method) (integer division), same convention as
# plots/run_wandb_plots_autoresidual_mean_mean.sh.
# ---------------------------------------------------------------------------
seeds_per_method=2
method_labels=(
  "GT-RDS-Residual-Noise"
  "GT-RDS-Residual-Distribution"
  "GT-RDS-Residual-Noise (bounded)"
  "GT-RDS-Residual-Distribution (bounded)"
)

# Indices (within each task block) holding use_actor_diff runs vs.
# use_actor_diff_mean runs. The noise-residual metric only exists in CSVs
# produced for the former; the mean/log-std-residual metrics only for the
# latter (see plots/eval_residual_norms.py).
diff_indices=(0 4)
diff_mean_indices=(2 6)
combined_indices=(0 2 4 6)

filter_indices() {
  # filter_indices <array_name> <idx> [<idx> ...]
  # Prints one line per element of <array_name>, blanking every index not
  # listed in <idx>... (preserves array length/order for --runs-per-task).
  local -n _src="$1"; shift
  local keep=("$@")
  for i in "${!_src[@]}"; do
    local matched=0
    for k in "${keep[@]}"; do
      [[ "$i" == "$k" ]] && matched=1 && break
    done
    if [[ "$matched" -eq 1 ]]; then
      printf '%s\n' "${_src[$i]}"
    else
      printf '\n'
    fi
  done
}

build_labels() {
  # build_labels <array_name> -> prints "run_id=Label" for each non-empty entry.
  local -n _src="$1"
  for i in "${!_src[@]}"; do
    local run_id="${_src[$i]}"
    [[ -z "${run_id}" ]] && continue
    local method_idx=$(( i / seeds_per_method ))
    printf '%s=%s\n' "${run_id}" "${method_labels[$method_idx]}"
  done
}

build_metric_map() {
  # build_metric_map <array_name> -> prints "run_id=metric" for each non-empty
  # entry. use_actor_diff runs (diff_indices) use residual_norm; diff_mean runs
  # use residual_mean_norm.
  local -n _src="$1"
  for i in "${!_src[@]}"; do
    local run_id="${_src[$i]}"
    [[ -z "${run_id}" ]] && continue
    local metric="residual_abs_mean"
    for k in "${diff_mean_indices[@]}"; do
      if [[ "$i" == "$k" ]]; then
        metric="residual_mean_abs_mean"
        break
      fi
    done
    printf '%s=%s\n' "${run_id}" "${metric}"
  done
}

task_var_names=(task1_ids task2_ids task3_ids task4_ids)
task_titles=("${task1_title}" "${task2_title}" "${task3_title}" "${task4_title}")
runs_per_task="${#task1_ids[@]}"

all_ids_combined=()
all_ids_diff_mean=()
all_labels=()
all_metric_map=()
for name in "${task_var_names[@]}"; do
  readarray -t _filtered_combined < <(filter_indices "${name}" "${combined_indices[@]}")
  readarray -t _filtered_diff_mean < <(filter_indices "${name}" "${diff_mean_indices[@]}")
  all_ids_combined+=("${_filtered_combined[@]}")
  all_ids_diff_mean+=("${_filtered_diff_mean[@]}")
  readarray -t _labels < <(build_labels "${name}")
  all_labels+=("${_labels[@]}")
  readarray -t _metric_map < <(build_metric_map "${name}")
  all_metric_map+=("${_metric_map[@]}")
done
labels_str=$(IFS=','; echo "${all_labels[*]}")
metric_map_str=$(IFS=','; echo "${all_metric_map[*]}")

# --- Plot settings ---
suptitle="GT-RDS Residual Means on \$\pi_{0.5}\$ LIBERO-90"
show_plot=0
ema_halflife=0
clip_to_shortest_run=0
max_steps=1000000

export LD_LIBRARY_PATH=
export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

extra_args=()
[[ ${show_plot} -eq 1 ]] && extra_args+=(--show)
[[ ${clip_to_shortest_run} -eq 1 ]] && extra_args+=(--clip-to-shortest-run)

echo "=== Plotting mean-residual norms (use_actor_diff + use_actor_diff_mean) ==="
python3 plots/plot_residual_norms_multi_tasks.py \
  --identifiers "${all_ids_combined[@]}" \
  --task-titles "${task_titles[@]}" \
  --runs-per-task "${runs_per_task}" \
  --data-dir "${data_dir}" \
  --metric residual_abs_mean \
  --metric-map "${metric_map_str}" \
  --ymin 0.0 \
  --ymax 0.7 \
  --ema-halflife "${ema_halflife}" \
  --output "plots/plots_multi_tasks/residual_mean_gtrds.svg" \
  --labels "${labels_str}" \
  --suptitle "${suptitle}" \
  --errorbar none \
  --max-steps "${max_steps}" \
  --dim-colors 1 \
  "${extra_args[@]}"

echo "=== Plotting log-std-residual norm (use_actor_diff_mean) ==="
python3 plots/plot_residual_norms_multi_tasks.py \
  --identifiers "${all_ids_diff_mean[@]}" \
  --task-titles "${task_titles[@]}" \
  --runs-per-task "${runs_per_task}" \
  --data-dir "${data_dir}" \
  --metric residual_log_std_abs_mean \
  --ymin 0.0 \
  --ymax 3.0 \
  --ema-halflife "${ema_halflife}" \
  --output "plots/plots_multi_tasks/residual_logstd_mean_gtrds.svg" \
  --labels "${labels_str}" \
  --suptitle "${suptitle}" \
  --errorbar none \
  --max-steps "${max_steps}" \
  --dim-colors 1 \
  "${extra_args[@]}"

# NOTE: --ymin/--ymax above are just starting points based on this repo's
# default residual_bound / residual_mean_bound / residual_log_std_high/low
# (see examples/scripts/run_libero_pi05_criticgpt_aractor_diff*.sh) — adjust
# to taste once you see the actual inferred norms.
