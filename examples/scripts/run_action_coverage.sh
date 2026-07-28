#!/usr/bin/env bash
#SBATCH --job-name=action_coverage
# run_action_coverage.sh
#
# End-to-end runner for action-space coverage analysis.
# Steps:
#   1. Collect actions (rollout + open-loop modes) for the given checkpoints.
#   2. Plot PCA coverage, per-dim std, and saturation curves.
#
# Usage:
#   bash examples/run_action_coverage.sh [OPTIONS]
#
# Examples:
#   # Quick test with 2 rollouts and 50 noise seeds
#   bash examples/run_action_coverage.sh --num_rollouts 2 --num_noise_seeds 50
#
#   # Compare two checkpoints with default settings
#   bash examples/run_action_coverage.sh \
#       --checkpoints "pi05_libero pi05_base" \
#       --num_rollouts 20 \
#       --num_noise_seeds 200
#
#   # Skip collection, just re-plot an existing run
#   bash examples/run_action_coverage.sh \
#       --data_dir logs/action_coverage/20240101_120000 \
#       --plot_only

set -euo pipefail

module load devel/miniforge 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate dsrl_pi0

device_id=0
export CUDA_VISIBLE_DEVICES=$device_id
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

export OUTPUT_DIR="${OUTPUT_DIR:-/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/XVLA_Libero_Tests}"
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/LIBERO"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CHECKPOINTS="pi05_libero"       # Space-separated list of checkpoint names
MODE="openloop"                     # rollout | openloop | both
NUM_ROLLOUTS=20
NUM_NOISE_SEEDS=200
QUERY_FREQ=5
LIBERO_SUITE="libero_90"
TASK_ID=59
SEED=0
OUTPUT_DIR="${OUTPUT_DIR:-./logs/action_coverage}"
DATA_DIR=""                     # Set to skip collection and re-plot only
PLOT_ONLY=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    grep "^#" "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints)    CHECKPOINTS="$2";      shift 2 ;;
        --mode)           MODE="$2";             shift 2 ;;
        --num_rollouts)   NUM_ROLLOUTS="$2";     shift 2 ;;
        --num_noise_seeds) NUM_NOISE_SEEDS="$2"; shift 2 ;;
        --query_freq)     QUERY_FREQ="$2";       shift 2 ;;
        --libero_suite)   LIBERO_SUITE="$2";     shift 2 ;;
        --task_id)        TASK_ID="$2";          shift 2 ;;
        --seed)           SEED="$2";             shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";       shift 2 ;;
        --data_dir)       DATA_DIR="$2";         shift 2 ;;
        --plot_only)      PLOT_ONLY=1;           shift   ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# XLA / JAX tuning
# ---------------------------------------------------------------------------
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_triton_gemm_any=True"
export TF_CPP_MIN_LOG_LEVEL=2

# ---------------------------------------------------------------------------
# Step 1 – Collect
# ---------------------------------------------------------------------------
if [[ $PLOT_ONLY -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo " Step 1: Collecting action distributions"
    echo "   checkpoints : ${CHECKPOINTS}"
    echo "   mode        : ${MODE}"
    echo "   num_rollouts: ${NUM_ROLLOUTS}"
    echo "   noise_seeds : ${NUM_NOISE_SEEDS}"
    echo "   suite/task  : ${LIBERO_SUITE} / task ${TASK_ID}"
    echo "   output_dir  : ${OUTPUT_DIR}"
    echo "============================================================"

    # shellcheck disable=SC2086  # intentional word-splitting of CHECKPOINTS
    python -m examples.collect_action_coverage \
        --checkpoints ${CHECKPOINTS} \
        --mode        "${MODE}" \
        --num_rollouts    "${NUM_ROLLOUTS}" \
        --num_noise_seeds "${NUM_NOISE_SEEDS}" \
        --query_freq      "${QUERY_FREQ}" \
        --libero_suite    "${LIBERO_SUITE}" \
        --task_id         "${TASK_ID}" \
        --seed            "${SEED}" \
        --output_dir      "${OUTPUT_DIR}"

    # Identify the most-recently created run directory to hand to the plotter
    DATA_DIR="$(ls -td "${OUTPUT_DIR}"/[0-9]* 2>/dev/null | head -n1)"
    if [[ -z "${DATA_DIR}" ]]; then
        echo "[ERROR] Could not locate output directory under ${OUTPUT_DIR}" >&2
        exit 1
    fi
    echo ""
    echo "Collection finished. Data directory: ${DATA_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 2 – Plot
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Step 2: Generating figures"
echo "   data_dir: ${DATA_DIR}"
echo "   mode    : ${MODE}"
echo "============================================================"

python -m examples.plot_action_coverage \
    --data_dir "${DATA_DIR}" \
    --mode     "${MODE}" \
    --libero_suite "${LIBERO_SUITE}" \
    --task_id "${TASK_ID}"

FIGS_DIR="${DATA_DIR}/figs"
echo ""
echo "============================================================"
echo " Done!"
echo " Figures saved to: ${FIGS_DIR}"
echo ""
echo " Files:"
# List generated figures if any
if [[ -d "${FIGS_DIR}" ]]; then
    ls "${FIGS_DIR}"/*.pdf "${FIGS_DIR}"/*.png 2>/dev/null | sed 's/^/   /'
fi
echo "============================================================"
