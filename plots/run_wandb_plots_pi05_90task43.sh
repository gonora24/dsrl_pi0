#!/bin/bash
set -euo pipefail

# --- W&B settings ---
proj_name="DSRL_pi05_Libero"
entity="noras-masterarbeit"  # leave empty to use WANDB_TEAM from jaxrl2/utils/wandb_config.py

# --- Runs to compare (W&B run ids / experiment ids, or local offline run dirs) ---
identifiers=(
  "dsrl_pi05_libero_90_task43_2026_06_28_05_42_04_0000--s-0_baseline_10numqs_10replan"
  "dsrl_pi05_libero_90_task43_2026_06_29_01_06_06_0000--s-0_chunkrewardcriticactor_mlp_10numqs_10replan_criticdim512_256_128"
  "dsrl_pi05_libero_90_task43_2026_06_29_08_55_16_0000--s-0_chunkrewardcriticactor_criticgpt_2numqs_10replan"
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
title="LIBERO-90 Task 43"
output="plots/pi05_libero90_task43_success.png"
show_plot=0   # set to 1 to display interactively
ymin=0.0
ymax=1.0
ema_halflife=50000  # training steps; set to 0 to disable smoothing
clip_to_shortest_run=0  # set to 0 to plot until the longest run ends

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
