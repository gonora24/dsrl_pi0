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

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

export OUTPUT_DIR=/hkfs/work/workspace/scratch/eu3660-dsrl/DSRL_pi0_Libero/

# --- Eval settings ---
pi0_checkpoint="pi05_libero"
libero_suite="libero_90"
num_rollouts=10
max_timesteps=100
seed=0
force=0   # set to 1 to re-evaluate runs whose output CSV already exists

# --- Task 1: Task 28 ---
task1_ids=(
  "dsrl_pi05_libero_90_task28_2026_08_05_09_10_55_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  "dsrl_pi05_libero_90_task28_2026_07_29_17_18_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  "dsrl_pi05_libero_90_task28_2026_08_07_22_46_43_0000--s-0_criticgpt_aractor_diff_mean"
)
task1_task_id=28

# --- Task 2: Task 43 ---
task2_ids=(
  "dsrl_pi05_libero_90_task43_2026_08_05_09_15_05_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  "dsrl_pi05_libero_90_task43_2026_07_29_17_18_34_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  "dsrl_pi05_libero_90_task43_2026_08_07_22_46_43_0000--s-0_criticgpt_aractor_diff_mean"
)
task2_task_id=43

# --- Task 3: Task 59 ---
task3_ids=(
  "dsrl_pi05_libero_90_task59_2026_07_12_09_53_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  "dsrl_pi05_libero_90_task59_2026_07_29_17_18_42_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  "dsrl_pi05_libero_90_task59_2026_08_08_16_10_19_0000--s-0_criticgpt_aractor_diff_mean"
)
task3_task_id=59

# --- Task 4: Task 60 ---
task4_ids=(
  "dsrl_pi05_libero_90_task60_2026_08_04_16_38_56_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual"
  "dsrl_pi05_libero_90_task60_2026_07_29_17_18_38_0000--s-0_criticgpt_diffaractor_10replan_clipaction_tanhresidual_freeze100k"
  "dsrl_pi05_libero_90_task60_2026_08_08_16_11_50_0000--s-0_criticgpt_aractor_diff_mean"
)
task4_task_id=60

task_var_names=(task1_ids task2_ids task3_ids task4_ids)
task_ids=("${task1_task_id}" "${task2_task_id}" "${task3_task_id}" "${task4_task_id}")

mkdir -p plots/data/residual_norms

for block_idx in "${!task_var_names[@]}"; do
  declare -n _run_ids="${task_var_names[$block_idx]}"
  task_id="${task_ids[$block_idx]}"

  for run_id in "${_run_ids[@]}"; do
    [[ -z "${run_id}" ]] && continue

    run_dir="${OUTPUT_DIR}/${run_id}"
    output_csv="plots/data/residual_norms/${run_id}.csv"

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
      --output "${output_csv}"
  done
  unset -n _run_ids
done
