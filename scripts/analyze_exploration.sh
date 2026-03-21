#!/usr/bin/env bash
# Analyze results from a packed exploration batch.
# Usage: ./scripts/analyze_exploration.sh <packed_job_id>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <packed_job_id>" >&2
    exit 1
fi

JOB_ID="$1"

echo "=== Exploration Batch ${JOB_ID} ==="
echo ""

# Get raw results
RAW=$("${SCRIPT_DIR}/lumi.sh" packed-results "${JOB_ID}" 2>&1)

if [[ -z "$RAW" ]]; then
    echo "No results found for job ${JOB_ID}" >&2
    exit 1
fi

# Print ranked results (sorted by bpb, column 3)
echo "--- Ranked by BPB (best first) ---"
echo "$RAW" | head -1  # header if any
echo "$RAW" | grep -v PENDING | grep -v FAILED | sort -k3 -n
echo ""

# Show failed/pending
FAILED=$(echo "$RAW" | grep FAILED || true)
PENDING=$(echo "$RAW" | grep PENDING || true)

if [[ -n "$FAILED" ]]; then
    echo "--- FAILED runs ---"
    echo "$FAILED"
    echo ""
fi

if [[ -n "$PENDING" ]]; then
    echo "--- PENDING runs ---"
    echo "$PENDING"
    echo ""
fi

# Check for crashes (tracebacks in stderr)
echo "--- Crash check ---"
CRASHES=$(ssh lumi "for i in 0 1 2 3 4 5 6 7; do
  err=~/parameter-golf/logs/pgolf_packed_${JOB_ID}_run\${i}.err
  if [ -f \$err ] && grep -q 'Traceback' \$err 2>/dev/null; then
    echo \"run\${i}: CRASH\"
  fi
done" 2>/dev/null || true)

if [[ -n "$CRASHES" ]]; then
    echo "$CRASHES"
else
    echo "No crashes detected."
fi
echo ""

# Check for OOM
echo "--- OOM check ---"
OOMS=$(ssh lumi "for i in 0 1 2 3 4 5 6 7; do
  err=~/parameter-golf/logs/pgolf_packed_${JOB_ID}_run\${i}.err
  log=~/parameter-golf/logs/pgolf_packed_${JOB_ID}_run\${i}.log
  if { [ -f \$err ] && grep -qi 'out of memory' \$err 2>/dev/null; } || \
     { [ -f \$log ] && grep -qi 'out of memory' \$log 2>/dev/null; }; then
    echo \"run\${i}: OOM\"
  fi
done" 2>/dev/null || true)

if [[ -n "$OOMS" ]]; then
    echo "$OOMS"
else
    echo "No OOM detected."
fi
echo ""

# Check for oversized artifacts
echo "--- Artifact size check (16MB limit) ---"
OVERSIZED=$(echo "$RAW" | grep -v PENDING | grep -v FAILED | awk '$4 != "?" && $4+0 > 16000000 {print $1 ": OVER_16MB (" $4 " bytes)"}' || true)

if [[ -n "$OVERSIZED" ]]; then
    echo "$OVERSIZED"
else
    echo "All artifacts within 16MB limit."
fi
