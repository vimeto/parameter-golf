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

# Workaround 1: GCC fails with "cannot read spec file './specs'" when our specs/
# directory is in CWD. Temporarily rename it.
mv specs _specs_tmp 2>/dev/null || true

# Workaround 2: /tmp on compute nodes can be full. Use scratch for inductor cache.
export TORCHINDUCTOR_CACHE_DIR="/scratch/project_2013932/${USER}/inductor_cache_${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${TRAIN_SCRIPT}"
EXIT_CODE=$?

# Restore specs dir
mv _specs_tmp specs 2>/dev/null || true
exit ${EXIT_CODE}
