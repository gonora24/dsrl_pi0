#!/bin/bash
set -euo pipefail

# --- W&B settings ---
proj_name="DSRL_pi0_Libero"
entity="noragorhan-karlsruhe-institute-of-technology"  # leave empty to use WANDB_TEAM from jaxrl2/utils/wandb_config.py

# --- Runs to compare (W&B run ids / experiment ids, or local offline run dirs) ---
identifiers=(
  "dsrl_pi0_libero_2026_06_12_00_40_04_0000--s-0_libero_90_task29"
  "dsrl_pi0_libero_90_task29_2026_06_15_09_32_45_0000--s-0_chunkrewardcriticactor_mlp_50replan"
  "dsrl_pi0_libero_90_task29_2026_06_18_13_40_08_0000--s-0_chunkrewardcriticactor_criticgpt_1numqs_20replan"
)

# --- Optional legend labels (comma-separated run_id=Label pairs; leave empty for auto labels) ---
labels=(
  "${identifiers[0]}=Baseline"
  "${identifiers[1]}=Chunked MLP Critic + Actor"
  "${identifiers[2]}=Chunked Critic Transformer + Actor MLP"
)

# --- Plot settings ---
metric="evaluation/success_rate"
x_axis="_step"
title="LIBERO-90 Task 29"
output="plots/pi0_libero90_task29_success.png"
show_plot=0   # set to 1 to display interactively
ymin=0.0
ymax=1.0
ema_halflife=25000  # training steps; set to 0 to disable smoothing
clip_to_shortest_run=1  # set to 0 to plot until the longest run ends

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

if [[ "${clip_to_shortest_run}" == "1" ]]; then
  cmd+=(--clip-to-shortest-run)
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
