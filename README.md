# Python 调用 Shell + JSON 配置的单/多显卡任务调度示例

本项目用于演示：

1. **Python 调用 Shell 脚本**；
2. **配置信息统一写入 JSON 文件**；
3. **单显卡：多个任务按顺序串行执行**；
4. **多显卡：多个 GPU worker 并行执行，每张 GPU 内部仍然串行执行自己的任务**。

项目中的 `src/train_demo.py` 是模拟训练程序，不依赖真实 GPU。真实项目中可以替换成自己的训练脚本，例如 `train.py`、`yolo train`、PyTorch 训练入口等。

---

## 一、项目结构

```text
python_shell_json_gpu_scheduler/
├── run_by_python.py                  # Python 统一启动入口：负责调用 Shell 脚本
├── configs/
│   ├── single_gpu_tasks.json         # 单显卡串行任务配置
│   ├── multi_gpu_tasks.json          # 多显卡并行任务配置
│   ├── exp_fight.json                # 单个任务配置：打架检测
│   ├── exp_fall.json                 # 单个任务配置：跌倒检测
│   ├── exp_run.json                  # 单个任务配置：奔跑检测
│   └── exp_loitering.json            # 单个任务配置：徘徊检测
├── scripts/
│   ├── check_env.sh                  # 环境检查脚本
│   ├── run_single_serial.sh          # 单显卡串行执行脚本
│   └── run_multi_parallel.sh         # 多显卡并行执行脚本
├── src/
│   ├── json_query.py                 # Shell 读取 JSON 的辅助工具
│   └── train_demo.py                 # 模拟训练程序
├── logs/                             # 运行日志目录
├── outputs/                          # 训练输出目录
└── docs/
    └── lesson_plan.md                # 详细教案和讲稿
```

---

## 二、Windows 用户准备工作

`.sh` 是 Shell 脚本，Windows 的 CMD / PowerShell 不能直接运行。推荐安装 **Git for Windows**，安装后会自带 Git Bash。

安装 Git Bash 后，可以直接使用：

```bash
python run_by_python.py --mode single
python run_by_python.py --mode multi
```

`run_by_python.py` 会自动查找常见路径中的 Git Bash，例如：

```text
C:\Program Files\Git\bin\bash.exe
```

---

## 三、推荐执行方式：通过 Python 调用 Shell

在 VSCode 中打开本项目文件夹，然后打开终端，确保当前位置是项目根目录。

### 1. 检查环境

```bash
python run_by_python.py --mode single --config configs/single_gpu_tasks.json
```

如果第一次只是想检查项目文件是否完整，也可以在 Git Bash 中运行：

```bash
./scripts/check_env.sh
```

### 2. 单显卡串行执行

```bash
python run_by_python.py --mode single
```

等价于 Python 内部调用：

```bash
bash scripts/run_single_serial.sh configs/single_gpu_tasks.json
```

执行逻辑：

```text
GPU 0：任务1 → 任务2 → 任务3 → 任务4
```

同一时间只有一个任务运行，适合单显卡环境。

### 3. 多显卡并行执行

```bash
python run_by_python.py --mode multi
```

等价于 Python 内部调用：

```bash
bash scripts/run_multi_parallel.sh configs/multi_gpu_tasks.json
```

执行逻辑：

```text
GPU 0：任务1 → 任务3
GPU 1：任务2 → 任务4
```

GPU 0 和 GPU 1 同时运行；但每张 GPU 内部仍然顺序执行自己的任务。

### 4. 手动指定配置文件

```bash
python run_by_python.py --mode single --config configs/single_gpu_tasks.json
python run_by_python.py --mode multi --config configs/multi_gpu_tasks.json
```

---

## 四、直接运行 Shell 脚本

如果你在 Git Bash、WSL、Linux 或 macOS 终端中，也可以直接执行：

```bash
chmod +x scripts/*.sh src/*.py
./scripts/check_env.sh
./scripts/run_single_serial.sh configs/single_gpu_tasks.json
./scripts/run_multi_parallel.sh configs/multi_gpu_tasks.json
```

---

## 五、查看日志和输出

单显卡日志：

```bash
ls logs/single_gpu
```

多显卡日志：

```bash
ls logs/multi_gpu
```

输出结果：

```bash
ls outputs/single_gpu
ls outputs/multi_gpu
```

每个任务完成后会在对应目录生成：

```text
metrics.json
best_model.txt
```

---

## 六、如何替换成自己的训练程序

假设你的真实训练命令是：

```bash
python train.py --config configs/xxx.yaml --epochs 100 --batch-size 16 --lr 0.001
```

可以修改 JSON 中的：

```json
"train_script": "src/train_demo.py"
```

改成：

```json
"train_script": "train.py"
```

然后在 `scripts/run_single_serial.sh` 和 `scripts/run_multi_parallel.sh` 中，将参数名调整成你自己的训练程序支持的参数即可。

例如本项目默认使用：

```bash
--task-name
--config
--epochs
--batch-size
--learning-rate
--output-dir
```

如果你的程序参数是 `--lr`，就把脚本里的：

```bash
--learning-rate "$LEARNING_RATE"
```

改成：

```bash
--lr "$LEARNING_RATE"
```

---

## 七、核心讲解一句话

> Python 是启动入口，Shell 是调度核心，JSON 是配置中心，训练程序是具体执行模块。
