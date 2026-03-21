#!/bin/bash
# Sequential multi-experiment launcher: runs experiments one after another on 8 GPUs.
#
# Data is loaded once to NVMe, then each experiment runs for its full time budget
# with training + validation + serialization, before the next one starts.
#
# Spec file format (one experiment per line):
#   MAX_WALLCLOCK_SECONDS ENV_OVERRIDES
#
# Example spec file:
#   3000 RUN_ID=baseline ITERATIONS=20000 TRAIN_BATCH_TOKENS=524288
#   3000 RUN_ID=wider_mlp ITERATIONS=20000 TRAIN_BATCH_TOKENS=524288 MLP_MULT=3
#
# Lines starting with # are skipped. Empty lines are skipped.
#
# Usage:
#   sbatch scripts/slurm/lumi_sequential.sh specs/sequential1.txt

#SBATCH --job-name=pgolf-seq
#SBATCH --account=project_462001163
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=05:00:00
#SBATCH --output=logs/pgolf_seq_%j.out
#SBATCH --error=logs/pgolf_seq_%j.err

set -euo pipefail

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
source "${CODE_DIR}/scripts/slurm/setup_env.sh"

PYTHON_PATH="/opt/miniconda3/envs/pytorch/bin/python"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-records/our_submission/train_gpt.py}"
NUM_GPUS="${NUM_GPUS:-8}"

# =============================================================================
# Pre-copy dataset to node-local NVMe (once for all experiments)
# =============================================================================

SCRATCH_DATA="/scratch/project_462001163/${USER}/pgolf_cache/datasets/fineweb10B_sp1024"
SCRATCH_TOKENIZER="/scratch/project_462001163/${USER}/pgolf_cache/tokenizers"
LOCAL_DATA="/tmp/pgolf_data_${SLURM_JOB_ID}"

if [ -d "${SCRATCH_DATA}" ]; then
    echo "Copying dataset to node-local NVMe: ${LOCAL_DATA}"
    COPY_START=$(date +%s)
    mkdir -p "${LOCAL_DATA}"
    cp -r "${SCRATCH_DATA}/"* "${LOCAL_DATA}/"
    COPY_ELAPSED=$(( $(date +%s) - COPY_START ))
    echo "Dataset copied ($(du -sh "${LOCAL_DATA}" | cut -f1)) in ${COPY_ELAPSED}s"
else
    echo "ERROR: scratch dataset not found at ${SCRATCH_DATA}"
    echo "Run lumi_setup_data.sh first"
    exit 1
fi

if [ -d "${SCRATCH_TOKENIZER}" ] && [ -f "${SCRATCH_TOKENIZER}/fineweb_1024_bpe.model" ]; then
    cp "${SCRATCH_TOKENIZER}/fineweb_1024_bpe.model" "/tmp/fineweb_1024_bpe_${SLURM_JOB_ID}.model"
    export TOKENIZER_PATH="/tmp/fineweb_1024_bpe_${SLURM_JOB_ID}.model"
fi

export DATA_PATH="${LOCAL_DATA}"

echo "==========================================="
echo "Sequential Experiment Launcher"
echo "==========================================="
echo "Spec file: ${SPEC_FILE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${NUM_GPUS}"
echo "DATA_PATH: ${DATA_PATH}"
echo ""

# =============================================================================
# Run experiments sequentially
# =============================================================================

RUN_INDEX=0
RESULTS_FILE="${CODE_DIR}/logs/pgolf_seq_${SLURM_JOB_ID}_results.txt"
echo "run_index	run_id	val_bpb	val_bpb_postquant	artifact_bytes	steps	wallclock_s" > "${RESULTS_FILE}"

while IFS= read -r line || [ -n "${line}" ]; do
    # Skip empty lines and comments
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue

    # Parse: MAX_WALLCLOCK_SECONDS followed by space-separated ENV_OVERRIDES
    read -r RUN_WALLCLOCK ENV_REST <<< "${line}"

    RUN_LOG="${CODE_DIR}/logs/pgolf_seq_${SLURM_JOB_ID}_run${RUN_INDEX}.log"

    echo "==========================================="
    echo "Run ${RUN_INDEX}: wallclock=${RUN_WALLCLOCK}s ${ENV_REST}"
    echo "  Log: ${RUN_LOG}"
    echo "  Started: $(date)"
    echo "==========================================="

    RUN_START=$(date +%s)

    # Run training with per-experiment env overrides
    (
        # Export each KEY=VALUE pair from the spec line
        for kv in ${ENV_REST}; do
            export "${kv}"
        done
        export MAX_WALLCLOCK_SECONDS="${RUN_WALLCLOCK}"

        if [ "${NUM_GPUS}" -eq 1 ]; then
            HIP_VISIBLE_DEVICES=0 \
            singularity exec "${SIFPYTORCH}" \
                ${PYTHON_PATH} -u "${TRAIN_SCRIPT}"
        else
            singularity exec "${SIFPYTORCH}" \
                ${PYTHON_PATH} -m torch.distributed.run \
                --nproc_per_node="${NUM_GPUS}" \
                --standalone \
                "${TRAIN_SCRIPT}"
        fi
    ) > "${RUN_LOG}" 2>&1

    RUN_EXIT=$?
    RUN_ELAPSED=$(( $(date +%s) - RUN_START ))

    if [ ${RUN_EXIT} -ne 0 ]; then
        echo "  FAILED (exit code ${RUN_EXIT}) after ${RUN_ELAPSED}s"
        echo "  Last 10 lines of log:"
        tail -n 10 "${RUN_LOG}" 2>/dev/null || true
        echo ""
    else
        echo "  Completed in ${RUN_ELAPSED}s"
        # Extract key metrics from log
        RUN_ID_EXTRACTED=$(grep -o 'RUN_ID=[^ ]*' <<< "${ENV_REST}" | cut -d= -f2 || echo "run${RUN_INDEX}")
        VAL_BPB=$(grep 'final_int8_zlib_roundtrip_exact' "${RUN_LOG}" 2>/dev/null | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2 || echo "N/A")
        ARTIFACT=$(grep 'Total submission size int8+zlib' "${RUN_LOG}" 2>/dev/null | grep -o '[0-9]* bytes' | head -1 | cut -d' ' -f1 || echo "N/A")
        STEPS=$(grep -o 'step:[0-9]*/[0-9]*' "${RUN_LOG}" 2>/dev/null | tail -1 | cut -d: -f2 | cut -d/ -f1 || echo "N/A")

        echo "  val_bpb=${VAL_BPB} artifact=${ARTIFACT} steps=${STEPS}"
        echo "${RUN_INDEX}	${RUN_ID_EXTRACTED}	${VAL_BPB}	${VAL_BPB}	${ARTIFACT}	${STEPS}	${RUN_ELAPSED}" >> "${RESULTS_FILE}"
    fi

    echo ""
    RUN_INDEX=$((RUN_INDEX + 1))
done < "${SPEC_FILE}"

# =============================================================================
# Summary
# =============================================================================

echo "==========================================="
echo "All ${RUN_INDEX} experiments complete"
echo "==========================================="
echo ""
echo "Results summary:"
cat "${RESULTS_FILE}"

# =============================================================================
# Cleanup
# =============================================================================

rm -rf "${LOCAL_DATA}" 2>/dev/null || true
rm -f "/tmp/fineweb_1024_bpe_${SLURM_JOB_ID}.model" 2>/dev/null || true
echo "Cleanup complete"
