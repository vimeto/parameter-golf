#!/bin/bash
# Setup environment for Parameter Golf training on LUMI-G
# Usage: source scripts/slurm/setup_env.sh
#
# Directory layout:
# - Code:     ~/parameter-golf           (home - persistent, small)
# - Caches:   /scratch/project_462001163/$USER/pgolf_cache (scratch - large, auto-purged)
# - Pip pkgs: /projappl/project_462001163/$USER/python_user (projappl - persistent)

# Note: no set -euo pipefail here — this file is sourced, caller controls error handling

# =============================================================================
# Configuration
# =============================================================================

ACCOUNT="${SLURM_JOB_ACCOUNT:-${SBATCH_ACCOUNT:-project_462001163}}"
export CODE_DIR="${HOME}/parameter-golf"

SCRATCH_BASE="/scratch/${ACCOUNT}/${USER}/pgolf_cache"
PROJAPPL_BASE="/projappl/${ACCOUNT}/${USER}"

echo "Setting up environment:"
echo "  CODE_DIR=${CODE_DIR}"
echo "  SCRATCH_BASE=${SCRATCH_BASE}"
echo "  PROJAPPL_BASE=${PROJAPPL_BASE}"

# =============================================================================
# Create directory structure
# =============================================================================

mkdir -p "${SCRATCH_BASE}"/{datasets,tokenizers,logs} 2>/dev/null || true
mkdir -p "${PROJAPPL_BASE}"/{python_user,pip_cache} 2>/dev/null || true
mkdir -p "${CODE_DIR}/logs" 2>/dev/null || true

# =============================================================================
# Python paths (on projappl - persistent across jobs)
# =============================================================================

export PIP_CACHE_DIR="${PROJAPPL_BASE}/pip_cache"
export PYTHONUSERBASE="${PROJAPPL_BASE}/python_user"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python3.12/site-packages:${PYTHONPATH:-}"

# Unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1

# =============================================================================
# ROCm / PyTorch settings
# =============================================================================

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# =============================================================================
# MIOpen cache on /tmp (avoid Lustre locking issues)
# =============================================================================

if [ -n "${SLURM_JOB_ID:-}" ]; then
    export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-${SLURM_NODEID:-0}"
    export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
    mkdir -p "${MIOPEN_USER_DB_PATH}" 2>/dev/null || true
fi

# =============================================================================
# HuggingFace cache (on scratch)
# =============================================================================

export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HUB_CACHE}" 2>/dev/null || true

# =============================================================================
# Container and bind paths
# =============================================================================

export SIFPYTORCH="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl,/tmp,${HOME}"

echo "Environment configured:"
echo "  PYTHONUSERBASE=${PYTHONUSERBASE}"
echo "  HF_HOME=${HF_HOME}"
echo "  MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-<not in slurm>}"
