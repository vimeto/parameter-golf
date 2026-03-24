#!/bin/bash
# Packed multi-experiment launcher: runs 8 parallel 1-GPU experiments on one node.
#
# Spec file format (one experiment per line):
#   GPU_ID ENV_OVERRIDES
#
# Example spec file:
#   0 ITERATIONS=1000 NUM_LAYERS=6 MODEL_DIM=384 RUN_ID=small_6L
#   1 ITERATIONS=1000 NUM_LAYERS=9 MODEL_DIM=512 RUN_ID=base_9L
#   2 ITERATIONS=1000 NUM_LAYERS=12 MODEL_DIM=512 RUN_ID=deep_12L
#   3 ITERATIONS=1000 NUM_LAYERS=9 MODEL_DIM=640 RUN_ID=wide_9L
#   ...
#
# Lines starting with # are skipped. Empty lines are skipped.
#
# Usage:
#   sbatch scripts/slurm/lumi_packed.sh specs/sweep1.txt

#SBATCH --job-name=pgolf-packed
#SBATCH --account=project_462001163
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=logs/pgolf_packed_%j.out
#SBATCH --error=logs/pgolf_packed_%j.err

set -euo pipefail

# Workaround: compute nodes may not have project group in secondary groups.
# sg switches primary group so we can access /scratch/project_462001163/.
if [ "${PGOLF_SG_DONE:-}" != "1" ]; then
    export PGOLF_SG_DONE=1
    exec sg project_462001163 -c "bash $0 $*"
fi

SPEC_FILE="${1:?spec file is required}"
export CODE_DIR="${HOME}/parameter-golf"
cd "${CODE_DIR}"
mkdir -p logs

if [ ! -f "${SPEC_FILE}" ]; then
    echo "Error: spec file not found: ${SPEC_FILE}"
    exit 1
fi

# =============================================================================
# Load modules and environment
# =============================================================================

module load LUMI/25.03
set +e  # setup_env may have non-fatal mkdir errors
source "${CODE_DIR}/scripts/slurm/setup_env.sh"
set -e

PYTHON_PATH="/opt/miniconda3/envs/pytorch/bin/python"

# =============================================================================
# Pre-copy dataset to node-local NVMe (shared by all experiments)
# =============================================================================

SCRATCH_DATA="/scratch/project_462001163/${USER}/pgolf_cache/datasets/fineweb10B_sp1024"
SCRATCH_TOKENIZER="/scratch/project_462001163/${USER}/pgolf_cache/tokenizers"
LOCAL_DATA="/tmp/pgolf_data_${SLURM_JOB_ID}"

if [ -d "${SCRATCH_DATA}" ]; then
    echo "Copying dataset to node-local NVMe: ${LOCAL_DATA}"
    mkdir -p "${LOCAL_DATA}"
    cp -r "${SCRATCH_DATA}/"* "${LOCAL_DATA}/"
    export DATA_PATH="${LOCAL_DATA}"
    echo "Dataset copied ($(du -sh "${LOCAL_DATA}" | cut -f1))"
else
    echo "Warning: scratch dataset not found, using default DATA_PATH"
fi

if [ -d "${SCRATCH_TOKENIZER}" ] && [ -f "${SCRATCH_TOKENIZER}/fineweb_1024_bpe.model" ]; then
    cp "${SCRATCH_TOKENIZER}/fineweb_1024_bpe.model" "/tmp/fineweb_1024_bpe_${SLURM_JOB_ID}.model"
    export TOKENIZER_PATH="/tmp/fineweb_1024_bpe_${SLURM_JOB_ID}.model"
fi

echo "==========================================="
echo "Packed Experiment Launcher"
echo "==========================================="
echo "Spec file: ${SPEC_FILE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# =============================================================================
# Launch experiments
# =============================================================================

ALL_PIDS=()
RUN_INDEX=0

while IFS= read -r line || [ -n "${line}" ]; do
    # Skip empty lines and comments
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue

    # Parse: GPU_ID followed by space-separated ENV_OVERRIDES
    read -r GPU_ID ENV_REST <<< "${line}"

    RUN_LOG="${CODE_DIR}/logs/pgolf_packed_${SLURM_JOB_ID}_run${RUN_INDEX}.log"

    echo "-------------------------------------------"
    echo "Run ${RUN_INDEX}: GPU=${GPU_ID} ${ENV_REST}"
    echo "  Log: ${RUN_LOG}"

    # Launch training with per-experiment env overrides
    (
        # Export each KEY=VALUE pair from the spec line
        for kv in ${ENV_REST}; do
            export "${kv}"
        done

        TRAIN_SCRIPT="${TRAIN_SCRIPT:-records/our_submission/train_gpt.py}"

        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        singularity exec "${SIFPYTORCH}" \
            ${PYTHON_PATH} -u "${TRAIN_SCRIPT}" \
            > "${RUN_LOG}" 2>&1
    ) &

    ALL_PIDS+=($!)
    echo "  PID: ${ALL_PIDS[-1]}"
    RUN_INDEX=$((RUN_INDEX + 1))
done < "${SPEC_FILE}"

echo ""
echo "==========================================="
echo "All ${RUN_INDEX} experiments launched"
echo "==========================================="
echo "PIDs: ${ALL_PIDS[*]}"

# =============================================================================
# Cleanup on exit
# =============================================================================

cleanup() {
    echo ""
    echo "Cleaning up..."
    for pid in "${ALL_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${ALL_PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    # Clean up NVMe data
    rm -rf "${LOCAL_DATA}" 2>/dev/null || true
    rm -f "/tmp/fineweb_1024_bpe_${SLURM_JOB_ID}.model" 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT TERM INT

# =============================================================================
# Wait for all experiments
# =============================================================================

WORST_EXIT=0
for i in $(seq 0 $((${#ALL_PIDS[@]} - 1))); do
    if wait "${ALL_PIDS[$i]}"; then
        echo "Run ${i} (PID ${ALL_PIDS[$i]}) completed successfully"
    else
        EXIT_CODE=$?
        echo "Run ${i} (PID ${ALL_PIDS[$i]}) exited with code ${EXIT_CODE}"
        [ "${EXIT_CODE}" -gt "${WORST_EXIT}" ] && WORST_EXIT="${EXIT_CODE}"
    fi
done

echo ""
echo "==========================================="
echo "All runs complete. Worst exit code: ${WORST_EXIT}"
echo "==========================================="

exit ${WORST_EXIT}
