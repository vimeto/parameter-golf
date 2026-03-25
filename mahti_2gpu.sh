#!/bin/bash
#SBATCH --job-name=pgolf-2gpu
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mahti_%j.out
#SBATCH --error=logs/mahti_%j.err

source /appl/profile/zz-csc-env.sh
module load pytorch/2.9

cd /scratch/project_2013932/vtoivone/pgolf
mkdir -p logs

# TRAIN_SCRIPT must be set via --export
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gpt.py}"
NUM_GPUS="${NUM_GPUS:-2}"
echo "Script: ${TRAIN_SCRIPT}"
echo "RUN_ID: ${RUN_ID:-default}"
echo "GPUs: ${NUM_GPUS}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Workaround: GCC fails with "cannot read spec file './specs'" when our specs/
# directory is in CWD (GCC interprets 'specs' dir as a spec file).
# Run from /tmp, pass absolute paths for data and script.
PGOLF_DIR="$(pwd)"
export DATA_PATH="${DATA_PATH:-${PGOLF_DIR}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${PGOLF_DIR}/data/tokenizers/fineweb_1024_bpe.model}"
WORK_DIR="/tmp/${USER}/pgolf_run_${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}/logs"
cd "${WORK_DIR}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${PGOLF_DIR}/${TRAIN_SCRIPT}"
