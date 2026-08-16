#!/bin/bash
# set -euo pipefail

# Combined multi-task heatmap of the per-latent-dimension influence metric
# (one column per task, gapped between columns). Pure post-hoc plotting over
# already-saved *_metrics.json files (as produced by jaxrl2/tests/jacobian.py)
# — no GPU / policy loading needed.

export LD_LIBRARY_PATH=

export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00139/eu3660/dsrl_pi0/dsrl_pi0/"
export PYTHONPATH=$PYTHONPATH:/home/hk-project-pai00139/eu3660/dsrl_pi0/LIBERO

# --- Which metrics JSON files to combine into a single plot (one column per task) ---
METRICS_JSON="plots/plots/jacobians/*chunk_metrics.json"

# --- Output settings ---
OUT_DIR="plots/plots/jacobians"
FILENAME="influence_heatmap_all_tasks"
CMAP="viridis"
LOG_SCALE=1
GAP=0.4
TITLE="Influence Metric $\pi_{0.5}$ LIBERO-90"

python3 plots/plot_influence_heatmap.py \
    --metrics_json ${METRICS_JSON} \
    --out_dir "${OUT_DIR}" \
    --filename "${FILENAME}" \
    --cmap "${CMAP}" \
    --log_scale ${LOG_SCALE} \
    --gap ${GAP} \
    --title "${TITLE}"
