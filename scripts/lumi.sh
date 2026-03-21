#!/usr/bin/env bash
# LUMI SSH bridge - run commands on LUMI from your local machine.
# Assumes ~/.ssh/config has a "lumi" host configured.
set -euo pipefail

REMOTE_DIR="~/parameter-golf"
LOG_DIR="${REMOTE_DIR}/logs"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [args...]

Commands:
  submit <script> [args...]     Submit a SLURM job (sbatch) on LUMI
  status                        Show running/pending jobs (squeue)
  logs <job_id>                 Print stdout log for a job
  errors <job_id>               Print stderr log for a job
  tail <job_id> [n]             Tail stdout log (default 50 lines)
  tail-err <job_id> [n]         Tail stderr log (default 50 lines)
  cancel <job_id>               Cancel a job (scancel)
  sync                          Git push locally, then git pull on LUMI
  results                       Print experiments/results.tsv from LUMI
  info                          Show partition availability (standard-g, dev-g)
  sacct <job_id>                Show accounting info for a job
  packed-results <job_id>       Compact results from a packed exploration batch (8 runs)
  seq-results <job_id>          Results from a sequential validation batch (4 runs)
  download <remote> <local>     SCP a file from LUMI to local
  help                          Show this help message
EOF
}

case "${1:-help}" in
    submit)
        shift
        if [[ $# -lt 1 ]]; then
            echo "Usage: $(basename "$0") submit <script> [args...]" >&2
            exit 1
        fi
        ssh lumi "cd ${REMOTE_DIR} && sbatch $*"
        ;;
    status)
        ssh lumi "squeue -u \$USER --format='%.10i %.15j %.8T %.10M %.6D %.4C %R'"
        ;;
    logs)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") logs <job_id>" >&2; exit 1; fi
        ssh lumi "cat ${LOG_DIR}/*${1}*.out 2>/dev/null"
        ;;
    errors)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") errors <job_id>" >&2; exit 1; fi
        ssh lumi "cat ${LOG_DIR}/*${1}*.err 2>/dev/null"
        ;;
    tail)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") tail <job_id> [n]" >&2; exit 1; fi
        job_id="$1"
        n="${2:-50}"
        ssh lumi "tail -n ${n} ${LOG_DIR}/*${job_id}*.out 2>/dev/null"
        ;;
    tail-err)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") tail-err <job_id> [n]" >&2; exit 1; fi
        job_id="$1"
        n="${2:-50}"
        ssh lumi "tail -n ${n} ${LOG_DIR}/*${job_id}*.err 2>/dev/null"
        ;;
    cancel)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") cancel <job_id>" >&2; exit 1; fi
        ssh lumi "scancel $1"
        echo "Cancelled job $1"
        ;;
    sync)
        echo "Pushing local changes..."
        git push origin main
        echo "Pulling on LUMI..."
        ssh lumi "cd ${REMOTE_DIR} && git fetch origin main && git reset --hard origin/main"
        ;;
    results)
        ssh lumi "cat ${REMOTE_DIR}/experiments/results.tsv 2>/dev/null"
        ;;
    info)
        ssh lumi "sinfo -p standard-g,dev-g -o '%P %a %l %D %T %F'"
        ;;
    sacct)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") sacct <job_id>" >&2; exit 1; fi
        ssh lumi "sacct -j $1 --format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed,Timelimit"
        ;;
    packed-results)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") packed-results <job_id>" >&2; exit 1; fi
        ssh lumi "for i in 0 1 2 3 4 5 6 7; do
          log=~/parameter-golf/logs/pgolf_packed_${1}_run\${i}.log
          if [ ! -f \$log ]; then
            echo \"run\${i} ? PENDING ? ? ?\"
            continue
          fi
          run_id=\$(grep 'run_id:' \$log 2>/dev/null | head -1 | sed 's/.*run_id://')
          bpb=\$(grep 'final_int8_zlib_roundtrip_exact' \$log 2>/dev/null | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
          artifact=\$(grep 'Total submission size int8+zlib' \$log 2>/dev/null | grep -o '[0-9]* bytes' | head -1 | cut -d' ' -f1)
          steps=\$(grep -o 'step:[0-9]*/[0-9]*' \$log 2>/dev/null | tail -1 | cut -d: -f2 | cut -d/ -f1)
          peak_mem=\$(grep 'peak memory' \$log 2>/dev/null | grep -o '[0-9]* MiB' | head -1)
          echo \"run\${i} \${run_id:-?} \${bpb:-FAILED} \${artifact:-?} \${steps:-?} \${peak_mem:-?}\"
        done" | column -t
        ;;
    seq-results)
        shift
        if [[ $# -lt 1 ]]; then echo "Usage: $(basename "$0") seq-results <job_id>" >&2; exit 1; fi
        ssh lumi "for i in 0 1 2 3; do
          log=~/parameter-golf/logs/pgolf_seq_${1}_run\${i}.log
          if [ ! -f \$log ]; then
            echo \"run\${i} ? PENDING ? ? ?\"
            continue
          fi
          run_id=\$(grep 'run_id:' \$log 2>/dev/null | head -1 | sed 's/.*run_id://')
          bpb=\$(grep 'final_int8_zlib_roundtrip_exact' \$log 2>/dev/null | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
          artifact=\$(grep 'Total submission size int8+zlib' \$log 2>/dev/null | grep -o '[0-9]* bytes' | head -1 | cut -d' ' -f1)
          steps=\$(grep -o 'step:[0-9]*/[0-9]*' \$log 2>/dev/null | tail -1 | cut -d: -f2 | cut -d/ -f1)
          peak_mem=\$(grep 'peak memory' \$log 2>/dev/null | grep -o '[0-9]* MiB' | head -1)
          echo \"run\${i} \${run_id:-?} \${bpb:-FAILED} \${artifact:-?} \${steps:-?} \${peak_mem:-?}\"
        done" | column -t
        ;;
    download)
        shift
        if [[ $# -lt 2 ]]; then echo "Usage: $(basename "$0") download <remote_path> <local_path>" >&2; exit 1; fi
        scp "lumi:$1" "$2"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1" >&2
        usage >&2
        exit 1
        ;;
esac
