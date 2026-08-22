#!/bin/bash
# set -euo pipefail

# ---------------------------------------------------------------------------
# Mirrors plots/run_wandb_plots_autoresidual_mean_mean.sh, but plots the
# residual-prediction norms inferred from checkpoints (via
# plots/eval_residual_norms.py / plots/run_eval_residual_norms.sh) instead of
# a metric already logged to W&B.
#
# Produces three figures:
#   - residual_norm_noise.svg  : ||residual|| for use_actor_diff runs
#   - residual_norm_mean.svg   : ||residual_mean||     for use_actor_diff_mean runs
#   - residual_norm_logstd.svg : ||residual_log_std||  for use_actor_diff_mean runs
#
# Run identifiers below are the same run names used in
# plots/run_eval_residual_norms.sh; each is resolved to
# plots/data/residual_norms/<run_name>.csv.
# ---------------------------------------------------------------------------

data_dir="plots/data/residual_norms"

# --- Task blocks: same run names / grouping as plots/run_eval_residual_norms.sh ---
task1_ids=(
  ""
  ""
  "dsrl_pi05_libero_90_task28_2026_08_05_09_10_55_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  ""
  "dsrl_pi05_libero_90_task28_2026_07_29_17_18_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  ""
  "dsrl_pi05_libero_90_task28_2026_08_07_22_46_43_0000--s-0_criticgpt_aractor_diff_mean"
  ""
)
task1_title="Task 28"

task2_ids=(
  ""
  ""
  "dsrl_pi05_libero_90_task43_2026_08_05_09_15_05_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  ""
  "dsrl_pi05_libero_90_task43_2026_07_29_17_18_34_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  ""
  "dsrl_pi05_libero_90_task43_2026_08_07_22_46_43_0000--s-0_criticgpt_aractor_diff_mean"
  ""
)
task2_title="Task 43"

task3_ids=(
  ""
  ""
  "dsrl_pi05_libero_90_task59_2026_07_12_09_53_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  ""
  "dsrl_pi05_libero_90_task59_2026_07_29_17_18_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  ""
  "dsrl_pi05_libero_90_task59_2026_08_08_16_10_19_0000--s-0_criticgpt_aractor_diff_mean"
  ""
)
task3_title="Task 59"

task4_ids=(
  ""
  ""
  "dsrl_pi05_libero_90_task60_2026_08_04_16_38_56_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  ""
  "dsrl_pi05_libero_90_task60_2026_07_29_17_18_38_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  ""
  "dsrl_pi05_libero_90_task60_2026_08_08_16_11_50_0000--s-0_criticgpt_aractor_diff_mean"
  ""
)
task4_title="Task 60"

# ---------------------------------------------------------------------------
# Shared legend labels — runs at position i within a task block belong to
# method (i / seeds_per_method) (integer division), same convention as
# plots/run_wandb_plots_autoresidual_mean_mean.sh.
# ---------------------------------------------------------------------------
seeds_per_method=2
method_labels=(
  ""
  "FDTS-Residual-Noise (no freezing)"
  "FDTS-Residual-Noise (with freezing)"
  "FDTS-Residual-Distribution (no freezing)"
)

# Indices (within each task block) holding use_actor_diff runs vs.
# use_actor_diff_mean runs. The noise-residual metric only exists in CSVs
# produced for the former; the mean/log-std-residual metrics only for the
# latter (see plots/eval_residual_norms.py).
diff_indices=(2 4)
diff_mean_indices=(6)

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

task_var_names=(task1_ids task2_ids task3_ids task4_ids)
task_titles=("${task1_title}" "${task2_title}" "${task3_title}" "${task4_title}")
runs_per_task="${#task1_ids[@]}"

all_ids_diff=()
all_ids_diff_mean=()
all_labels=()
for name in "${task_var_names[@]}"; do
  readarray -t _filtered_diff < <(filter_indices "${name}" "${diff_indices[@]}")
  readarray -t _filtered_diff_mean < <(filter_indices "${name}" "${diff_mean_indices[@]}")
  all_ids_diff+=("${_filtered_diff[@]}")
  all_ids_diff_mean+=("${_filtered_diff_mean[@]}")
  readarray -t _labels < <(build_labels "${name}")
  all_labels+=("${_labels[@]}")
done
labels_str=$(IFS=','; echo "${all_labels[*]}")

# --- Plot settings ---
suptitle="\$\pi_{0.5}\$ LIBERO-90"
show_plot=0
ema_halflife=0          # checkpoints are sparse; rely on --errorbar for spread instead
clip_to_shortest_run=0
max_steps=1000000

export LD_LIBRARY_PATH=
export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

extra_args=()
[[ ${show_plot} -eq 1 ]] && extra_args+=(--show)
[[ ${clip_to_shortest_run} -eq 1 ]] && extra_args+=(--clip-to-shortest-run)

echo "=== Plotting noise-residual norm (use_actor_diff) ==="
python3 plots/plot_residual_norms_multi_tasks.py \
  --identifiers "${all_ids_diff[@]}" \
  --task-titles "${task_titles[@]}" \
  --runs-per-task "${runs_per_task}" \
  --data-dir "${data_dir}" \
  --metric residual_norm \
  --ymin 0.0 \
  --ymax 1.0 \
  --ema-halflife "${ema_halflife}" \
  --output "plots/plots_multi_tasks/residual_norm_noise.svg" \
  --labels "${labels_str}" \
  --suptitle "${suptitle}" \
  --errorbar ci \
  --max-steps "${max_steps}" \
  --residual-colors 1 \
  "${extra_args[@]}"

echo "=== Plotting mean-residual norm (use_actor_diff_mean) ==="
python3 plots/plot_residual_norms_multi_tasks.py \
  --identifiers "${all_ids_diff_mean[@]}" \
  --task-titles "${task_titles[@]}" \
  --runs-per-task "${runs_per_task}" \
  --data-dir "${data_dir}" \
  --metric residual_mean_norm \
  --ymin 0.0 \
  --ymax 0.3 \
  --ema-halflife "${ema_halflife}" \
  --output "plots/plots_multi_tasks/residual_norm_mean.svg" \
  --labels "${labels_str}" \
  --suptitle "${suptitle}" \
  --errorbar ci \
  --max-steps "${max_steps}" \
  --residual-colors 1 \
  "${extra_args[@]}"

echo "=== Plotting log-std-residual norm (use_actor_diff_mean) ==="
python3 plots/plot_residual_norms_multi_tasks.py \
  --identifiers "${all_ids_diff_mean[@]}" \
  --task-titles "${task_titles[@]}" \
  --runs-per-task "${runs_per_task}" \
  --data-dir "${data_dir}" \
  --metric residual_log_std_norm \
  --ymin 0.0 \
  --ymax 1.0 \
  --ema-halflife "${ema_halflife}" \
  --output "plots/plots_multi_tasks/residual_norm_logstd.svg" \
  --labels "${labels_str}" \
  --suptitle "${suptitle}" \
  --errorbar ci \
  --max-steps "${max_steps}" \
  --residual-colors 1 \
  "${extra_args[@]}"

# NOTE: --ymin/--ymax above are just starting points based on this repo's
# default residual_bound / residual_mean_bound / residual_log_std_high/low
# (see examples/scripts/run_libero_pi05_criticgpt_aractor_diff*.sh) — adjust
# to taste once you see the actual inferred norms.
