#!/usr/bin/env bash
# Mahti SSH bridge - run commands on Mahti from your local machine.
# Assumes ~/.ssh/config has a "mahti" host configured.
set -euo pipefail

REMOTE_DIR="/scratch/project_2013932/vtoivone/pgolf"
LOG_DIR="${REMOTE_DIR}/logs"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [args...]

Commands:
  submit <env_overrides...>   Submit a 1-GPU SLURM job on Mahti
                              e.g.: submit TRAIN_SCRIPT=specs/batch47/run1.py RUN_ID=b47_r1
  status                      Show running/pending jobs (squeue)
  logs <job_id>               Print stdout log for a job
  errors <job_id>             Print stderr log for a job
  tail <job_id> [n]           Tail stdout log (default 50 lines)
  cancel <job_id>             Cancel a job (scancel)
  sync                        Git push locally, then git pull on Mahti
  results <job_id>            Grep key metrics from a job log
  help                        Show this help message
EOF
}

case "${1:-help}" in
    submit)
        shift
        if [[ $# -lt 1 ]]; then
            echo "Usage: $(basename "$0") submit KEY=VALUE [KEY=VALUE ...]" >&2
            echo "  Required: TRAIN_SCRIPT=... RUN_ID=..." >&2
            exit 1
        fi
        # Build --export string from positional args
        EXPORT_VARS="ALL"
        TIME="01:00:00"
        for arg in "$@"; do
            if [[ "${arg}" == TIME=* ]]; then
                TIME="${arg#TIME=}"
            else
                EXPORT_VARS="${EXPORT_VARS},${arg}"
            fi
        done
        ssh mahti "cd ${REMOTE_DIR} && sbatch --export=${EXPORT_VARS} --time=${TIME} mahti_1gpu.sh"
        ;;
    status)
        ssh mahti "squeue -u \$USER --format='%.10i %.15j %.8T %.10M %.6D %.4C %R'"
        ;;
    logs)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") logs <job_id>" >&2; exit 1; fi
        ssh mahti "cat ${LOG_DIR}/mahti_${1}.out 2>/dev/null"
        ;;
    errors)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") errors <job_id>" >&2; exit 1; fi
        ssh mahti "cat ${LOG_DIR}/mahti_${1}.err 2>/dev/null"
        ;;
    tail)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") tail <job_id> [n]" >&2; exit 1; fi
        N="${2:-50}"
        ssh mahti "tail -n ${N} ${LOG_DIR}/mahti_${1}.out 2>/dev/null"
        ;;
    cancel)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") cancel <job_id>" >&2; exit 1; fi
        ssh mahti "scancel ${1}"
        ;;
    sync)
        echo "Pushing local changes..."
        git push 2>&1 || true
        echo "Pulling on Mahti..."
        ssh mahti "cd ${REMOTE_DIR} && git pull"
        ;;
    results)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") results <job_id>" >&2; exit 1; fi
        ssh mahti "grep -E 'roundtrip|sliding|ttt_bpb|artifact_bytes|post_quant|val_bpb|standard_postquant' ${LOG_DIR}/mahti_${1}.out 2>/dev/null | tail -20"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1" >&2
        usage
        exit 1
        ;;
esac
