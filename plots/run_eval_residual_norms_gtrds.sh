#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_residual_norms

# ---------------------------------------------------------------------------
# For every non-empty run name below, restores each checkpoint saved under
# $OUTPUT_DIR/<run_name>, runs on-policy rollouts through pi0, and writes a
# long-format CSV of residual-prediction norms to
# plots/data/residual_norms/<run_name>.csv (see plots/eval_residual_norms.py).
#
# Run names are grouped into task blocks exactly like
# plots/run_wandb_plots_autoresidual_mean_mean.sh, so the resulting CSVs can
# be fed straight into plots/run_plot_residual_norms_multi_tasks.sh.
# ---------------------------------------------------------------------------

module load devel/miniforge
conda deactivate
conda activate dsrl_pi0

export OUTPUT_DIR=/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero

export PYTHONPATH="${PYTHONPATH}:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/pfs/data6/home/ka/ka_anthropomatik/ka_eu3660/projects/dsrl_pi0/LIBERO

# --- Eval settings ---
pi0_checkpoint="pi05_libero"
libero_suite="libero_90"
num_rollouts=10
max_timesteps=100
seed=0
force=0      # set to 1 to re-evaluate runs whose output CSV already exists
abs_mean=1   # set to 1 to compute mean of absolute residuals instead of L2 norms

task1_ids=(
  "dsrl_pi05_libero_90_task29_2026_08_16_17_41_03_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  "dsrl_pi05_libero_90_task29_2026_08_21_23_56_23_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec2reps"
  "dsrl_pi05_libero_90_task29_2026_08_21_04_24_00_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps_0.1bound"
  "dsrl_pi05_libero_90_task29_2026_08_20_13_29_55_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec2reps_0.1bound"
)
task1_title="Task 29"
task1_task_id=29
task1_sft="0.50"


# --- Task 2 ---
task2_ids=(
  "dsrl_pi05_libero_90_task38_2026_08_16_17_19_53_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  "dsrl_pi05_libero_90_task38_2026_08_19_00_37_07_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec_2reps"
  "dsrl_pi05_libero_90_task38_2026_08_20_14_53_41_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps_0.1bound"
  "dsrl_pi05_libero_90_task38_2026_08_21_04_24_00_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec2reps_0.1bound"
)
task2_title="Task 38"
task2_sft="0.61"
task2_task_id=38

task3_ids=(
  "dsrl_pi05_libero_90_task59_2026_08_23_09_00_36_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  "dsrl_pi05_libero_90_task59_2026_08_23_09_00_36_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec_2reps"
  "dsrl_pi05_libero_90_task59_2026_08_20_14_53_41_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps_0.1bound"
  "dsrl_pi05_libero_90_task59_2026_08_24_03_15_03_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec_2reps_bounded0.2"
)
task3_title="Task 59"
task3_sft="0.34"
task3_task_id=59

task4_ids=(
  "dsrl_pi05_libero_90_task64_2026_08_19_00_09_20_0000--s-0_criticgpt_aractor_diff_7dims_5vec_2reps"
  "dsrl_pi05_libero_90_task64_2026_08_19_00_37_08_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec_2reps"
  "dsrl_pi05_libero_90_task64_2026_08_22_00_25_00_0000--s-0_criticgpt_aractor_diff_7dims_5vec2reps_0.1bound"
  "dsrl_pi05_libero_90_task64_2026_08_20_13_25_02_0000--s-0_criticgpt_aractor_diff_mean_7dims_5vec2reps_0.1bound"
  ""
)
task4_title="Task 64"
task4_sft="0.42"
task4_task_id=64

task_var_names=(task1_ids task2_ids task3_ids task4_ids)
task_ids=("${task1_task_id}" "${task2_task_id}" "${task3_task_id}" "${task4_task_id}")

if [[ "${abs_mean}" -eq 1 ]]; then
  output_dir="plots/data/residual_abs_means"
  abs_mean_flag="--abs_mean"
else
  output_dir="plots/data/residual_norms"
  abs_mean_flag=""
fi
mkdir -p "${output_dir}"

for block_idx in "${!task_var_names[@]}"; do
  declare -n _run_ids="${task_var_names[$block_idx]}"
  task_id="${task_ids[$block_idx]}"

  for run_id in "${_run_ids[@]}"; do
    [[ -z "${run_id}" ]] && continue

    run_dir="${OUTPUT_DIR}/${run_id}"
    output_csv="${output_dir}/${run_id}.csv"

    if [[ -f "${output_csv}" && "${force}" -ne 1 ]]; then
      echo "Skipping ${run_id} (${output_csv} already exists; set force=1 to re-evaluate)"
      continue
    fi

    echo "=== Evaluating ${run_id} (task_id=${task_id}) ==="
    python3 plots/eval_residual_norms.py \
      --run_dir "${run_dir}" \
      --task_id "${task_id}" \
      --libero_suite "${libero_suite}" \
      --pi0_checkpoint "${pi0_checkpoint}" \
      --num_rollouts "${num_rollouts}" \
      --max_timesteps "${max_timesteps}" \
      --seed "${seed}" \
      --output "${output_csv}" \
      ${abs_mean_flag}
  done
  unset -n _run_ids
done
