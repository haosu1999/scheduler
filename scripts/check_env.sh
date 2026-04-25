#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] 没有找到 python 或 python3，请先安装 Python。"
  exit 1
fi

echo "============================================================"
echo "环境检查"
echo "项目根目录：$ROOT_DIR"
echo "Python 命令：$PYTHON_BIN"
$PYTHON_BIN --version
echo "Shell：$SHELL"
echo "当前目录：$(pwd)"
echo "============================================================"

echo "检查关键文件："
for path in \
  "run_by_python.py" \
  "configs/single_gpu_tasks.json" \
  "configs/multi_gpu_tasks.json" \
  "scripts/run_single_serial.sh" \
  "scripts/run_multi_parallel.sh" \
  "src/train_demo.py" \
  "src/json_query.py"
do
  if [ -e "$path" ]; then
    echo "  [OK] $path"
  else
    echo "  [MISSING] $path"
  fi
done

echo "============================================================"
echo "环境检查完成。"
