#!/bin/bash
set -euo pipefail

# --- W&B settings ---
proj_name="DSRL_NA_Libero"
entity="noragorhan-karlsruhe-institute-of-technology"  # leave empty to use WANDB_TEAM from jaxrl2/utils/wandb_config.py

# --- Runs to compare (W&B run ids / experiment ids, or local offline run dirs) ---
identifiers=(
  "dsrl_pi05_libero_90_task59_2026_08_23_08_03_43_0000--s-0_baseline_7dims_10replan"
  "dsrl_na_libero_90_task59_2026_08_12_19_44_15_0000--s-0_singleactorcritic_offline"
  "dsrl_na_libero_90_task59_2026_08_15_02_39_46_0000--s-0_singleactorcritic_offline"
  "dsrl_na_libero_90_task59_2026_08_17_08_47_19_0000--s-0_singleactorcritic_offline"
  "dsrl_na_libero_90_task59_2026_08_25_03_31_04_0000--s-0_singleactorcritic_offline"
)
sft_baselines="0.34"

# --- Optional legend labels (comma-separated run_id=Label pairs; leave empty for auto labels) ---
labels=(
  "${identifiers[0]}=DSRL-SAC (Baseline)"
  "${identifiers[1]}=RDS-NA"
  "${identifiers[2]}=RDS-NA"
  "${identifiers[3]}=RDS-NA"
  "${identifiers[4]}=RDS-NA"
)

# --- Plot settings ---
metric="evaluation/success_rate"
x_axis="_step"
title="RDS-NA on \$\pi_{0.5}\$ LIBERO-90 Task 59"
output="plots/pi05_libero90_task59_rds_na.svg"
show_plot=0   # set to 1 to display interactively
ymin=0.0
ymax=1.0
ema_halflife=1000 # training steps; set to 0 to disable smoothing
clip_to_shortest_run=0  # set to 0 to plot until the longest run ends
max_steps=180000

# --- Environment (adjust if needed) ---
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

if command -v module >/dev/null 2>&1; then
  module load devel/miniforge 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
  conda deactivate 2>/dev/null || true
  conda activate dsrl_pi0 2>/dev/null || true
fi

export PYTHONPATH="${PYTHONPATH:-}:${script_dir}"

cmd=(python3 "${script_dir}/wandb_plots.py" --project "${proj_name}")

if [[ -n "${entity}" ]]; then
  cmd+=(--entity "${entity}")
fi

cmd+=(--identifiers "${identifiers[@]}")
cmd+=(--metric "${metric}")
cmd+=(--x-axis "${x_axis}")
cmd+=(--ymin "${ymin}")
cmd+=(--ymax "${ymax}")
cmd+=(--ema-halflife "${ema_halflife}")
if [[ -n "${sft_baselines}" ]]; then
  cmd+=(--sft-baselines "${sft_baselines}")
fi
if [[ "${clip_to_shortest_run}" == "1" ]]; then
  cmd+=(--clip-to-shortest-run)
fi
if [[ -n "${max_steps}" ]]; then
  cmd+=(--max-steps "${max_steps}")
fi
if [[ -n "${labels[*]}" ]]; then
  labels_csv="$(IFS=,; echo "${labels[*]}")"
  cmd+=(--labels "${labels_csv}")
fi

if [[ -n "${title}" ]]; then
  cmd+=(--title "${title}")
fi

if [[ -n "${output}" ]]; then
  cmd+=(--output "${output}")
fi

if [[ "${show_plot}" == "1" ]]; then
  cmd+=(--show)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
