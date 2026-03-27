#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-$(git branch --show-current)}"
REMOTE_DIR="${REMOTE_DIR:-~/parameter-golf}"

echo "Pushing local branch ${BRANCH}..."
git push -u origin "${BRANCH}"

echo "Syncing branch ${BRANCH} on LUMI..."
ssh lumi "cd ${REMOTE_DIR} && git fetch origin ${BRANCH} && (git checkout ${BRANCH} || git checkout -b ${BRANCH} origin/${BRANCH}) && git reset --hard origin/${BRANCH}"

echo "LUMI sync complete for ${BRANCH}"
