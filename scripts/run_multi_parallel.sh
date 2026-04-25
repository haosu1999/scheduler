#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${1:-configs/multi_gpu_tasks.json}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] 没有找到 python 或 python3，请先安装 Python。"
  exit 1
fi

JSON_TOOL="src/json_query.py"

TRAIN_SCRIPT="$($PYTHON_BIN "$JSON_TOOL" global_field "$CONFIG_FILE" train_script)"
LOG_DIR="$($PYTHON_BIN "$JSON_TOOL" global_field "$CONFIG_FILE" log_dir)"
TASK_COUNT="$($PYTHON_BIN "$JSON_TOOL" task_count "$CONFIG_FILE")"
GPU_COUNT="$($PYTHON_BIN "$JSON_TOOL" gpu_count "$CONFIG_FILE")"

if [ "$GPU_COUNT" -le 0 ]; then
  echo "[ERROR] multi_gpu_tasks.json 中 global.gpu_ids 不能为空。"
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "多显卡并行执行"
echo "配置文件：$CONFIG_FILE"
echo "训练脚本：$TRAIN_SCRIPT"
echo "日志目录：$LOG_DIR"
echo "任务数量：$TASK_COUNT"
echo "GPU 数量：$GPU_COUNT"
echo "说明：不同 GPU worker 同时运行；同一 GPU worker 内部按顺序执行。"
echo "============================================================"

run_worker() {
  local WORKER_INDEX="$1"
  local GPU_ID="$2"
  local TOTAL_WORKERS="$3"

  echo "[WORKER-$WORKER_INDEX] 启动，绑定 GPU：$GPU_ID"

  local ENABLED_ORDER=0

  for ((i=0; i<TASK_COUNT; i++)); do
    local ENABLED
    ENABLED="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" enabled)"

    if [ "$ENABLED" = "false" ]; then
      continue
    fi

    # 轮询分配任务：第 0、2、4... 个启用任务给 worker 0；第 1、3、5... 个启用任务给 worker 1，以此类推。
    local ASSIGNED_WORKER=$((ENABLED_ORDER % TOTAL_WORKERS))
    ENABLED_ORDER=$((ENABLED_ORDER + 1))

    if [ "$ASSIGNED_WORKER" -ne "$WORKER_INDEX" ]; then
      continue
    fi

    local NAME TASK_CONFIG EPOCHS BATCH_SIZE LEARNING_RATE OUTPUT_DIR LOG_FILE
    NAME="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" name)"
    TASK_CONFIG="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" config)"
    EPOCHS="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" epochs)"
    BATCH_SIZE="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" batch_size)"
    LEARNING_RATE="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" learning_rate)"
    OUTPUT_DIR="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" output_dir)"
    LOG_FILE="$LOG_DIR/gpu${GPU_ID}_${NAME}.log"

    mkdir -p "$OUTPUT_DIR"

    echo
    echo "------------------------------------------------------------"
    echo "[WORKER-$WORKER_INDEX][START] 任务：$NAME"
    echo "[WORKER-$WORKER_INDEX][START] GPU：$GPU_ID"
    echo "[WORKER-$WORKER_INDEX][START] 日志：$LOG_FILE"
    echo "------------------------------------------------------------"

    if ! CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$TRAIN_SCRIPT" \
        --task-name "$NAME" \
        --config "$TASK_CONFIG" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1; then
      echo "[WORKER-$WORKER_INDEX][ERROR] 任务失败：$NAME"
      echo "[WORKER-$WORKER_INDEX][ERROR] 请查看日志：$LOG_FILE"
      return 1
    fi

    echo "[WORKER-$WORKER_INDEX][DONE] 任务完成：$NAME"
  done

  echo "[WORKER-$WORKER_INDEX] 所有分配任务执行完成。"
  return 0
}

PIDS=()

for ((w=0; w<GPU_COUNT; w++)); do
  GPU_ID="$($PYTHON_BIN "$JSON_TOOL" gpu_id "$CONFIG_FILE" "$w")"
  run_worker "$w" "$GPU_ID" "$GPU_COUNT" &
  PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "============================================================"
  echo "[ERROR] 至少有一个 worker 执行失败，请检查日志目录：$LOG_DIR"
  echo "============================================================"
  exit 1
fi

echo
echo "============================================================"
echo "多显卡并行任务全部完成。"
echo "日志目录：$LOG_DIR"
echo "============================================================"
