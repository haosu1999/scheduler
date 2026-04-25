#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_by_python.py

本文件是统一启动入口，体现“Python 调用 Shell 脚本，然后 Shell 读取 JSON 配置”的结构。

执行示例：
python run_by_python.py --mode single
python run_by_python.py --mode multi
python run_by_python.py --mode single --config configs/single_gpu_tasks.json
python run_by_python.py --mode multi --config configs/multi_gpu_tasks.json

说明：
- Python：负责选择运行模式、检查配置文件、调用 Shell 脚本；
- Shell：负责具体任务调度；
- JSON：负责保存任务参数；
- train_demo.py：负责模拟训练任务。
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def find_bash() -> str:
    """
    查找 bash。
    Windows 下优先查找 Git Bash，避免默认调用未安装发行版的 WSL bash。
    """
    env_bash = os.environ.get("SCHEDULER_BASH")
    if env_bash and Path(env_bash).exists():
        return env_bash

    system_name = platform.system().lower()

    if system_name == "windows":
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
        ]
        for item in candidates:
            if Path(item).exists():
                return item

        bash_in_path = shutil.which("bash")
        if bash_in_path and "windows\\system32" not in bash_in_path.lower():
            return bash_in_path

        raise RuntimeError(
            "没有找到可用的 bash。\n"
            "建议安装 Git for Windows，然后重新打开 VSCode。\n"
            "安装后本程序会自动查找 C:\\Program Files\\Git\\bin\\bash.exe。"
        )

    bash_in_path = shutil.which("bash")
    if bash_in_path:
        return bash_in_path

    raise RuntimeError("没有找到 bash，请在 Linux/macOS/WSL/Git Bash 环境下运行。")


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def show_config_summary(config_path: Path) -> None:
    config = load_config(config_path)
    tasks = config.get("tasks", [])
    enabled_tasks = [task for task in tasks if task.get("enabled", True)]

    print("=" * 70)
    print("配置文件检查")
    print(f"配置路径：{config_path.as_posix()}")
    print(f"任务总数：{len(tasks)}")
    print(f"启用任务数：{len(enabled_tasks)}")

    global_cfg = config.get("global", {})
    if "gpu_id" in global_cfg:
        print(f"单显卡 GPU：{global_cfg.get('gpu_id')}")
    if "gpu_ids" in global_cfg:
        print(f"多显卡列表：{global_cfg.get('gpu_ids')}")

    print("启用任务：")
    for task in enabled_tasks:
        print(f"  - {task.get('name')} | epochs={task.get('epochs')} | batch={task.get('batch_size')}")
    print("=" * 70)


def run_shell(mode: str, config_path: Path) -> int:
    bash_path = find_bash()

    if mode == "single":
        script_path = Path("scripts/run_single_serial.sh")
    elif mode == "multi":
        script_path = Path("scripts/run_multi_parallel.sh")
    else:
        raise ValueError(f"未知模式：{mode}")

    abs_script_path = PROJECT_ROOT / script_path
    abs_config_path = PROJECT_ROOT / config_path

    if not abs_script_path.exists():
        raise FileNotFoundError(f"Shell 脚本不存在：{abs_script_path}")

    show_config_summary(abs_config_path)

    # 传给 bash 的参数用相对路径，避免 Windows 路径和 Git Bash 路径格式冲突。
    command = [
        bash_path,
        script_path.as_posix(),
        config_path.as_posix()
    ]

    print("Python 即将调用 Shell 脚本：")
    print(" ".join(command))
    print("=" * 70)

    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Python 调用 Shell 脚本执行 JSON 中配置的任务。")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        required=True,
        help="single 表示单显卡串行；multi 表示多显卡并行"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON 配置文件路径。不填写时自动使用默认配置。"
    )
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    elif args.mode == "single":
        config_path = Path("configs/single_gpu_tasks.json")
    else:
        config_path = Path("configs/multi_gpu_tasks.json")

    try:
        code = run_shell(args.mode, config_path)
        if code != 0:
            print(f"Shell 脚本执行失败，返回码：{code}", file=sys.stderr)
            return code
        print("全部执行完成。")
        return 0
    except Exception as exc:
        print(f"[run_by_python.py ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
