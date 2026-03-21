#!/bin/bash
# One-time data preparation for Parameter Golf on LUMI
#
# Downloads FineWeb dataset (sp1024, all shards) directly to scratch.
# Run once before training.
#
# Usage:
#   sbatch scripts/slurm/lumi_setup_data.sh

#SBATCH --job-name=pgolf-setup
#SBATCH --account=project_462001163
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pgolf_setup_%j.out
#SBATCH --error=logs/pgolf_setup_%j.err

set -euo pipefail

export CODE_DIR="${HOME}/parameter-golf"
cd "${CODE_DIR}"
mkdir -p logs

# =============================================================================
# Load modules and environment
# =============================================================================

module load LUMI/25.03
source "${CODE_DIR}/scripts/slurm/setup_env.sh"

PYTHON_PATH="/opt/miniconda3/envs/pytorch/bin/python"
SCRATCH_DATA="/scratch/project_462001163/${USER}/pgolf_cache"

echo "==========================================="
echo "Parameter Golf - Data Setup"
echo "==========================================="

# =============================================================================
# Install pip dependencies to projappl
# =============================================================================

echo ""
echo "Installing pip dependencies..."
singularity exec "${SIFPYTORCH}" \
    ${PYTHON_PATH} -m pip install --user --no-warn-script-location \
    sentencepiece numpy tqdm huggingface-hub datasets

echo "Pip packages installed to ${PYTHONUSERBASE}"

# =============================================================================
# Download FineWeb dataset directly to scratch
# =============================================================================
# The download script writes to ./data/datasets/ and ./data/tokenizers/
# relative to CWD. We create a workspace on scratch and run from there.

DOWNLOAD_DIR="${SCRATCH_DATA}/download_workspace"
mkdir -p "${DOWNLOAD_DIR}/data"

# Copy the download script to scratch workspace
cp "${CODE_DIR}/data/cached_challenge_fineweb.py" "${DOWNLOAD_DIR}/data/"
cp "${CODE_DIR}/data/tokenizer_specs.json" "${DOWNLOAD_DIR}/data/"

echo ""
echo "Downloading FineWeb dataset (sp1024, all 80 train shards) to scratch..."
cd "${DOWNLOAD_DIR}"

singularity exec "${SIFPYTORCH}" \
    ${PYTHON_PATH} -u data/cached_challenge_fineweb.py \
    --variant sp1024 \
    --train-shards 80

echo "Dataset download complete"
echo ""
ls -lh "${DOWNLOAD_DIR}/data/datasets/fineweb10B_sp1024/" | head -5
echo "..."
ls "${DOWNLOAD_DIR}/data/datasets/fineweb10B_sp1024/" | wc -l
echo "files total"
echo ""
ls -lh "${DOWNLOAD_DIR}/data/tokenizers/"

# Move to final location
mkdir -p "${SCRATCH_DATA}/datasets" "${SCRATCH_DATA}/tokenizers"
rsync -a "${DOWNLOAD_DIR}/data/datasets/fineweb10B_sp1024/" "${SCRATCH_DATA}/datasets/fineweb10B_sp1024/"
rsync -a "${DOWNLOAD_DIR}/data/tokenizers/" "${SCRATCH_DATA}/tokenizers/"

# =============================================================================
# Create directory structure
# =============================================================================

cd "${CODE_DIR}"
mkdir -p logs experiments/runs

echo ""
echo "==========================================="
echo "Setup complete!"
echo "==========================================="
echo "  Dataset: ${SCRATCH_DATA}/datasets/fineweb10B_sp1024/"
echo "  Tokenizer: ${SCRATCH_DATA}/tokenizers/"
echo "  Pip packages: ${PYTHONUSERBASE}"
echo ""
echo "Run training with:"
echo "  DATA_PATH=${SCRATCH_DATA}/datasets/fineweb10B_sp1024 sbatch scripts/slurm/lumi_train.sh"
