#!/bin/bash

# ─────────────────────────────────────────────
#  GPU LAUNCHER FOR PYTORCH SCRIPTS
# ─────────────────────────────────────────────

chmod +x "$0"

# ── CONFIG ────────────────────────────────────
GPU_IDS="0,1"             
NUM_GPUS=2
LOG_DIR="runs/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_$TIMESTAMP.log"

# if the caller hasn't set GPU_ID, pick the first one from GPU_IDS
if [ -z "$GPU_IDS" ]; then
    IFS=',' read -r -a _gpus <<< "$GPU_IDS"
    GPU_IDS=${_gpus[0]}
    echo "[launcher] GPU_ID was unset, defaulting to $GPU_IDS (from GPU_IDS=$GPU_IDS)"
fi

# ── COMMAND ──────────────────────────────
COMMAND="python train.py --datasets_dir 'datasets' --model unet --loss idea"

# ── VALIDATION ────────────────────────────────
if [ -z "$COMMAND" ]; then
    echo "Please set COMMAND in the script."
    exit 1
fi

# Check nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "[WARNING] nvidia-smi not found. Is CUDA installed?"
fi

# ── GPU INFO ──────────────────────────────────
echo "============================================"
echo "  GPU LAUNCHER"
echo "============================================"
echo "  Time     : $TIMESTAMP"
echo "  GPU ID   : $GPU_IDS"
echo "  Command  : $COMMAND"
echo "  Log file : $LOG_FILE"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    && echo "--------------------------------------------"

# ── RUN ───────────────────────────────────────
mkdir -p "$LOG_DIR"

{
    echo "════════════════════════════════════════════"
    echo "Command: $COMMAND"
    echo "────────────────────────────────────────────"
} >> "$LOG_FILE"

nohup bash -c "
CUDA_VISIBLE_DEVICES=$GPU_IDS \
PYTHONUNBUFFERED=1 \
    $COMMAND
" >>"$LOG_FILE" 2>&1 &
PID=$!

echo "Started background job with PID $PID" >> "$LOG_FILE"
echo "Launched background job with PID $PID"
# we don't wait for the process; exit immediately with success
EXIT_CODE=0

# ── SUMMARY ───────────────────────────────────
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ DONE — Exit code: $EXIT_CODE"
else
    echo "  ❌ FAILED — Exit code: $EXIT_CODE"
fi
echo "  Log saved to: $LOG_FILE"
echo "============================================"

exit $EXIT_CODE