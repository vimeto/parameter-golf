#!/bin/bash
# Parameter Golf training launcher for LUMI-G
#
# Usage:
#   NUM_GPUS=1 sbatch scripts/slurm/lumi_train.sh              # single GPU
#   NUM_GPUS=8 sbatch scripts/slurm/lumi_train.sh              # 8 GPU DDP
#   NUM_GPUS=1 ITERATIONS=500 sbatch scripts/slurm/lumi_train.sh  # with overrides
#
# All train_gpt.py env vars (ITERATIONS, MODEL_DIM, NUM_LAYERS, etc.) are
# passed through to the training script.

#SBATCH --job-name=pgolf
#SBATCH --account=project_462001163
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pgolf_%j.out
#SBATCH --error=logs/pgolf_%j.err

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"
export CODE_DIR="${HOME}/parameter-golf"
cd "${CODE_DIR}"
mkdir -p logs

# =============================================================================
# Load modules and environment
# =============================================================================

module load LUMI/25.03
set +e  # setup_env may have non-fatal mkdir errors
source "${CODE_DIR}/scripts/slurm/setup_env.sh"
set -e

PYTHON_PATH="/opt/miniconda3/envs/pytorch/bin/python"

# =============================================================================
# Pre-copy dataset from scratch to node-local NVMe for fast I/O
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
    echo "Warning: scratch dataset not found at ${SCRATCH_DATA}, using default DATA_PATH"
fi

# Copy tokenizer to local NVMe too
if [ -d "${SCRATCH_TOKENIZER}" ] && [ -f "${SCRATCH_TOKENIZER}/fineweb_1024_bpe.model" ]; then
    cp "${SCRATCH_TOKENIZER}/fineweb_1024_bpe.model" "/tmp/fineweb_1024_bpe_${SLURM_JOB_ID}.model"
    export TOKENIZER_PATH="/tmp/fineweb_1024_bpe_${SLURM_JOB_ID}.model"
fi

# =============================================================================
# Launch training
# =============================================================================

echo "==========================================="
echo "Parameter Golf Training"
echo "==========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${NUM_GPUS}"
echo "DATA_PATH: ${DATA_PATH:-<default>}"
echo "TOKENIZER_PATH: ${TOKENIZER_PATH:-<default>}"
echo ""

TRAIN_SCRIPT="${TRAIN_SCRIPT:-records/our_submission/train_gpt.py}"

if [ "${NUM_GPUS}" -eq 1 ]; then
    # Single GPU training
    HIP_VISIBLE_DEVICES=0 \
    srun --ntasks=1 --export=ALL \
        singularity exec "${SIFPYTORCH}" \
        ${PYTHON_PATH} -u "${TRAIN_SCRIPT}"
else
    # Multi-GPU DDP training via torchrun
    srun --ntasks=1 --export=ALL \
        singularity exec "${SIFPYTORCH}" \
        ${PYTHON_PATH} -m torch.distributed.run \
        --nproc_per_node="${NUM_GPUS}" \
        --standalone \
        "${TRAIN_SCRIPT}"
fi

echo "Training complete"
