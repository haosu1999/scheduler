#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json_query.py

这是给 Shell 脚本使用的 JSON 读取工具。
为了避免依赖 jq，本项目用 Python 读取 JSON，再把字段值返回给 Shell 脚本。

用法示例：
python src/json_query.py global_field configs/single_gpu_tasks.json train_script
python src/json_query.py task_count configs/single_gpu_tasks.json
python src/json_query.py task_field configs/single_gpu_tasks.json 0 name
python src/json_query.py gpu_count configs/multi_gpu_tasks.json
python src/json_query.py gpu_id configs/multi_gpu_tasks.json 0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 配置文件不存在：{json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read fields from JSON config for shell scripts.")
    parser.add_argument("command", choices=[
        "global_field",
        "task_count",
        "task_field",
        "gpu_count",
        "gpu_id"
    ])
    parser.add_argument("config_path")
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()

    config = load_json(args.config_path)

    if args.command == "global_field":
        if len(args.args) != 1:
            raise ValueError("global_field 需要字段名，例如 train_script")
        field = args.args[0]
        print(to_shell_value(config.get("global", {}).get(field, "")))
        return 0

    if args.command == "task_count":
        print(len(config.get("tasks", [])))
        return 0

    if args.command == "task_field":
        if len(args.args) != 2:
            raise ValueError("task_field 需要任务下标和字段名，例如 0 name")
        index = int(args.args[0])
        field = args.args[1]
        tasks = config.get("tasks", [])
        if index < 0 or index >= len(tasks):
            raise IndexError(f"任务下标越界：{index}")
        print(to_shell_value(tasks[index].get(field, "")))
        return 0

    if args.command == "gpu_count":
        gpu_ids = config.get("global", {}).get("gpu_ids", [])
        print(len(gpu_ids))
        return 0

    if args.command == "gpu_id":
        if len(args.args) != 1:
            raise ValueError("gpu_id 需要 GPU 下标，例如 0")
        index = int(args.args[0])
        gpu_ids = config.get("global", {}).get("gpu_ids", [])
        if index < 0 or index >= len(gpu_ids):
            raise IndexError(f"GPU 下标越界：{index}")
        print(to_shell_value(gpu_ids[index]))
        return 0

    raise ValueError(f"未知命令：{args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[json_query.py ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
