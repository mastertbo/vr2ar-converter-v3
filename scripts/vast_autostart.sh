#!/usr/bin/env bash
set -euo pipefail

# Determine repository root (directory containing this script is /scripts)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Allow manual override of worker count; otherwise detect via nvidia-smi.
if [[ -n "${VR2AR_WORKER_COUNT:-}" ]]; then
  WORKER_COUNT="$VR2AR_WORKER_COUNT"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    WORKER_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
  else
    WORKER_COUNT=1
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not available on PATH; install Docker and retry." >&2
  exit 1
fi

# Start the dispatcher/UI container.
echo "Starting dispatcher (vr2ar service) in detached mode..."
docker compose up -d

# Start worker containers under the worker profile, scaling to the requested count.
if [[ "$WORKER_COUNT" -gt 0 ]]; then
  echo "Launching $WORKER_COUNT worker container(s) using profile 'worker'..."
  docker compose up -d --profile worker --scale vr2ar-worker="$WORKER_COUNT" vr2ar-worker
else
  echo "VR2AR_WORKER_COUNT is 0; skipping worker startup."
fi

echo "All services launched. Workers derive VR2AR_WORKER_ID from their hostname unless explicitly set."
