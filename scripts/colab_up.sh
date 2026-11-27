#!/usr/bin/env bash
set -euo pipefail

SESSION="${2:-gpu}"
REQUESTED_GPU="${1:-}"

if colab status -s "$SESSION" 2>&1 | grep -q "^\[$SESSION\]"; then
  echo "[colab_up] Reusing active session '$SESSION'."
else
  if [ -n "$REQUESTED_GPU" ]; then
    GPUS=("$REQUESTED_GPU")
  else
    GPUS=(H100 A100 G4 L4 T4)
  fi

  PROVISIONED=0
  for gpu in "${GPUS[@]}"; do
    echo "[colab_up] Trying GPU: $gpu (session '$SESSION')..."
    if COLAB_CLI_ASSIGN_RETRIES=1 COLAB_CLI_RETRY_BACKOFF=5 colab new -s "$SESSION" --gpu "$gpu"; then
      PROVISIONED=1
      break
    fi
    echo "[colab_up] $gpu unavailable, trying next tier..."
  done

  if [ "$PROVISIONED" -ne 1 ]; then
    echo "[colab_up] Failed to provision any GPU (${GPUS[*]})." >&2
    exit 1
  fi
fi

colab status -s "$SESSION"
echo '!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' | colab exec -s "$SESSION"

# Check if Google Drive is already mounted
if echo "import os; print('DRIVE_MOUNTED_YES' if os.path.isdir('/content/drive/MyDrive') else 'DRIVE_MOUNTED_NO')" | colab exec -s "$SESSION" 2>/dev/null | grep -q "DRIVE_MOUNTED_YES"; then
  echo "[colab_up] Google Drive already mounted at /content/drive."
elif [ -t 0 ]; then
  echo "[colab_up] Mounting Google Drive at /content/drive..."
  colab drivemount -s "$SESSION"
else
  echo "[colab_up] Next step (run in your terminal to mount Google Drive):"
  echo "  colab drivemount -s $SESSION"
fi
