#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${1:-configs/single_gpu_tasks.json}"

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
GPU_ID="$($PYTHON_BIN "$JSON_TOOL" global_field "$CONFIG_FILE" gpu_id)"
LOG_DIR="$($PYTHON_BIN "$JSON_TOOL" global_field "$CONFIG_FILE" log_dir)"
TASK_COUNT="$($PYTHON_BIN "$JSON_TOOL" task_count "$CONFIG_FILE")"

if [ -z "$GPU_ID" ]; then
  GPU_ID="0"
fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "单显卡串行执行"
echo "配置文件：$CONFIG_FILE"
echo "训练脚本：$TRAIN_SCRIPT"
echo "使用 GPU：$GPU_ID"
echo "日志目录：$LOG_DIR"
echo "任务数量：$TASK_COUNT"
echo "说明：当前任务未结束前，下一个任务不会启动。"
echo "============================================================"

for ((i=0; i<TASK_COUNT; i++)); do
  ENABLED="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" enabled)"

  if [ "$ENABLED" = "false" ]; then
    NAME="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" name)"
    echo "[SKIP] 跳过未启用任务：$NAME"
    continue
  fi

  NAME="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" name)"
  TASK_CONFIG="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" config)"
  EPOCHS="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" epochs)"
  BATCH_SIZE="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" batch_size)"
  LEARNING_RATE="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" learning_rate)"
  OUTPUT_DIR="$($PYTHON_BIN "$JSON_TOOL" task_field "$CONFIG_FILE" "$i" output_dir)"
  LOG_FILE="$LOG_DIR/${NAME}.log"

  mkdir -p "$OUTPUT_DIR"

  echo
  echo "------------------------------------------------------------"
  echo "[START] 任务序号：$i"
  echo "[START] 任务名称：$NAME"
  echo "[START] GPU：$GPU_ID"
  echo "[START] 日志：$LOG_FILE"
  echo "------------------------------------------------------------"

  if ! CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$TRAIN_SCRIPT" \
      --task-name "$NAME" \
      --config "$TASK_CONFIG" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --output-dir "$OUTPUT_DIR" \
      > "$LOG_FILE" 2>&1; then
    echo "[ERROR] 任务失败：$NAME"
    echo "[ERROR] 请查看日志：$LOG_FILE"
    exit 1
  fi

  echo "[DONE] 任务完成：$NAME"
done

echo
echo "============================================================"
echo "单显卡串行任务全部完成。"
echo "日志目录：$LOG_DIR"
echo "============================================================"
