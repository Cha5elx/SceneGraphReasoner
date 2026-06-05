#!/usr/bin/env bash
set -u

THRESHOLD_MB=29000
CHECK_INTERVAL_SECONDS=60
GPU_IDS=(1 0)
RUN_SCRIPT="scripts/run.sh"
LOG_FILE="output.log"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found." >&2
  exit 1
fi

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "$RUN_SCRIPT not found." >&2
  exit 1
fi

while true; do
  for GPU_ID in "${GPU_IDS[@]}"; do
    FREE_MB=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')

    if ! [[ "$FREE_MB" =~ ^[0-9]+$ ]]; then
      echo "$(date '+%F %T') GPU${GPU_ID}: unable to read free memory, skip." >&2
      continue
    fi

    if [ "$FREE_MB" -ge "$THRESHOLD_MB" ]; then
      echo "$(date '+%F %T') GPU${GPU_ID}: ${FREE_MB} MB free >= ${THRESHOLD_MB} MB. Starting training."
      CUDA_VISIBLE_DEVICES="$GPU_ID" setsid bash "$RUN_SCRIPT" > /dev/null 2> "$LOG_FILE" &
      exit 0
    fi

    echo "$(date '+%F %T') GPU${GPU_ID}: ${FREE_MB} MB free < ${THRESHOLD_MB} MB."
  done

  sleep "$CHECK_INTERVAL_SECONDS"
done
