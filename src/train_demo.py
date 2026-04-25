#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_demo.py

这是一个“模拟训练程序”，用于演示 Shell 脚本如何调度多个任务。
真实项目中，你可以把它替换成自己的 train.py，例如 YOLO、PyTorch、TensorFlow 等训练入口。

它会：
1. 读取命令行参数；
2. 读取单个任务的 config JSON；
3. 模拟若干 epoch 的训练；
4. 在 output_dir 中保存 metrics.json 和 best_model.txt。
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path


def read_json(path: str) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo training script.")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "CPU/未指定GPU")
    exp_config = read_json(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"任务名称：{args.task_name}")
    print(f"使用 GPU：{gpu}")
    print(f"任务配置：{args.config}")
    print(f"实验类别：{exp_config.get('class_name', 'unknown')}")
    print(f"模型名称：{exp_config.get('model', 'unknown')}")
    print(f"训练轮数：{args.epochs}")
    print(f"Batch Size：{args.batch_size}")
    print(f"Learning Rate：{args.learning_rate}")
    print(f"输出目录：{args.output_dir}")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    seed = sum(ord(c) for c in args.task_name)
    random.seed(seed)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # 模拟训练过程，真实项目里这里会执行模型训练。
        time.sleep(0.35)
        loss = max(0.01, 1.0 / epoch + random.random() * 0.05)
        acc = min(0.99, 0.65 + epoch * 0.05 + random.random() * 0.03)
        best_acc = max(best_acc, acc)

        print(
            f"[{args.task_name}] "
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"loss={loss:.4f} | acc={acc:.4f} | gpu={gpu}",
            flush=True
        )

    metrics = {
        "task_name": args.task_name,
        "gpu": gpu,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "best_accuracy": round(best_acc, 4),
        "finish_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": exp_config,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with (output_dir / "best_model.txt").open("w", encoding="utf-8") as f:
        f.write(f"这是 {args.task_name} 的模拟模型文件。\n")
        f.write(f"实际项目中，这里可以替换成 best.pt、last.pt 等模型权重文件。\n")

    print(f"任务完成：{args.task_name}")
    print(f"结果已保存到：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
