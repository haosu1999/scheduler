# 详细教案：Python 调用 Shell 脚本，基于 JSON 配置实现单/多显卡任务调度

## 一、讲解主题

本节主要讲解实验任务的自动化调度方式。整体设计采用“Python 启动入口 + Shell 调度脚本 + JSON 配置文件”的结构。其中，Python 负责作为统一入口调用 Shell 脚本；Shell 脚本负责读取 JSON 配置并调度多个训练任务；JSON 文件负责保存任务参数；训练程序负责执行具体的模型训练或实验流程。

可以概括为：

```text
Python 启动入口
    ↓
调用 Shell 脚本
    ↓
Shell 脚本读取 JSON 配置文件
    ↓
根据 JSON 配置执行多个训练任务
    ↓
保存日志和实验结果
```

这部分重点不是模型结构本身，而是实验执行方式的工程化管理。它解决的是多组实验参数难管理、重复输入命令麻烦、单显卡容易显存冲突、多显卡利用率不高等问题。

---

## 二、教学目标

讲完这一部分后，需要让听众理解以下几个问题：

1. 为什么要把配置信息写入 JSON 文件；
2. 为什么要通过 Python 调用 Shell 脚本；
3. 单显卡环境下为什么采用串行执行；
4. 多显卡环境下为什么采用并行执行；
5. Shell 脚本如何读取配置、指定 GPU、执行任务和保存日志；
6. 这种方式对实验复现和结果管理有什么帮助。

---

## 三、整体架构讲解

### 1. 四个模块的职责

可以先讲清楚各部分分工：

```text
run_by_python.py
    统一启动入口，负责接收 mode 参数，并调用对应 Shell 脚本。

configs/*.json
    保存任务参数，例如任务名称、训练轮数、batch size、学习率、输出目录和 GPU 编号。

scripts/*.sh
    负责调度任务。单显卡脚本负责串行执行，多显卡脚本负责启动多个 worker 并行执行。

src/train_demo.py
    模拟训练程序。真实项目中可以替换成自己的 train.py。
```

汇报时可以这样说：

> 在实现上，我没有把所有参数直接写死在训练命令中，而是把参数统一放到 JSON 配置文件中。Python 文件作为统一启动入口，负责选择运行模式；Shell 脚本作为调度核心，负责读取 JSON 并组织任务执行；训练程序只关注单个任务的训练过程。这样可以让配置、调度和训练逻辑相互分离，后续修改实验参数时只需要调整 JSON 文件，不需要反复修改脚本代码。

---

## 四、为什么使用 JSON 配置文件

JSON 文件主要保存任务参数，例如：

```json
{
  "global": {
    "train_script": "src/train_demo.py",
    "gpu_id": "0",
    "log_dir": "logs/single_gpu"
  },
  "tasks": [
    {
      "name": "fight_detection_exp",
      "enabled": true,
      "config": "configs/exp_fight.json",
      "epochs": 4,
      "batch_size": 8,
      "learning_rate": 0.001,
      "output_dir": "outputs/single_gpu/fight_detection_exp"
    }
  ]
}
```

讲解重点：

- `global` 是全局配置，所有任务都要用；
- `tasks` 是任务列表，一个对象对应一个实验任务；
- `enabled` 可以控制任务是否启用；
- `output_dir` 可以让每个任务保存到独立目录，避免结果覆盖；
- 参数写在 JSON 中，便于后期维护、复现和批量修改。

可以这样讲：

> JSON 配置文件相当于实验任务清单。每个任务的训练轮数、batch size、学习率、配置文件路径和输出目录都写在 JSON 中。这样做的好处是参数和执行逻辑分开管理。当我要新增任务时，只需要在 tasks 数组中增加一个对象；当我要临时跳过某个任务时，只需要把 enabled 改成 false，而不需要改 Shell 脚本主体逻辑。

---

## 五、Python 调用 Shell 脚本的作用

本项目的入口是：

```bash
python run_by_python.py --mode single
python run_by_python.py --mode multi
```

`run_by_python.py` 的作用是：

1. 接收运行模式 `single` 或 `multi`；
2. 根据模式选择对应的 Shell 脚本；
3. 检查 JSON 配置文件；
4. 使用 `subprocess.run()` 调用 Shell 脚本；
5. 把 JSON 配置文件路径传递给 Shell。

可以这样讲：

> Python 脚本主要起到统一入口的作用。用户只需要通过 `--mode` 参数选择单显卡模式或多显卡模式，Python 就会自动选择对应的 Shell 脚本和 JSON 配置文件。实际的任务调度逻辑仍然放在 Shell 中完成，这样既保留了 Shell 脚本在任务调度方面的灵活性，又让整体启动方式更加统一。

核心代码思想：

```python
subprocess.run([
    bash_path,
    "scripts/run_single_serial.sh",
    "configs/single_gpu_tasks.json"
])
```

可以解释为：

> 这行代码的含义是，Python 通过 subprocess 模块启动 bash，并将 Shell 脚本路径和 JSON 配置文件路径作为参数传递给它。也就是说，Python 并不直接训练模型，而是负责调用 Shell，由 Shell 继续完成后续任务调度。

---

## 六、单显卡串行执行讲解

### 1. 为什么单显卡要串行

单显卡环境下，如果多个任务同时运行，很容易出现显存不足。例如两个训练任务同时加载模型和数据，可能会导致 CUDA out of memory。因此单显卡下更适合串行执行，也就是一个任务完成后再执行下一个任务。

讲解语句：

> 在单显卡环境下，我采用串行执行方式。因为只有一张 GPU，如果多个训练任务同时启动，就会同时占用显存，容易导致显存不足或任务崩溃。因此 Shell 脚本会按照 JSON 中任务列表的顺序依次执行任务，当前任务没有结束之前，下一个任务不会启动。

### 2. 单显卡执行流程

```text
读取 single_gpu_tasks.json
    ↓
读取 global.gpu_id
    ↓
遍历 tasks 列表
    ↓
取出任务参数
    ↓
CUDA_VISIBLE_DEVICES=0 执行训练
    ↓
保存日志
    ↓
执行下一个任务
```

### 3. 关键命令解释

Shell 脚本最终会执行类似命令：

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_demo.py \
  --task-name fight_detection_exp \
  --config configs/exp_fight.json \
  --epochs 4 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --output-dir outputs/single_gpu/fight_detection_exp
```

讲解重点：

- `CUDA_VISIBLE_DEVICES=0`：指定任务只使用第 0 张显卡；
- `python src/train_demo.py`：调用训练程序；
- 后面的参数来自 JSON 文件；
- `> log_file 2>&1`：把标准输出和错误输出都写入日志文件；
- Shell 命令默认阻塞执行，所以当前命令未结束时不会执行下一条任务命令。

### 4. 单显卡讲稿

> 单显卡脚本的核心是“固定 GPU + 顺序执行”。脚本首先读取 JSON 中的全局配置，包括训练脚本路径、GPU 编号和日志目录。然后遍历 tasks 数组，对每个启用的任务读取任务名称、配置文件路径、训练轮数、batch size、学习率和输出目录。每次执行任务时，脚本通过 `CUDA_VISIBLE_DEVICES=0` 指定使用第 0 张显卡，并调用训练程序。由于 Shell 命令默认是阻塞式执行的，所以当前训练任务没有结束之前，脚本不会进入下一次循环，从而保证同一时间只有一个任务占用 GPU。

---

## 七、多显卡并行执行讲解

### 1. 为什么多显卡要并行

如果服务器有多张显卡，仍然按单显卡方式一个任务一个任务跑，会造成其他显卡空闲。多显卡并行的目的就是让不同显卡同时执行不同任务，提高资源利用率。

讲解语句：

> 在多显卡环境下，如果仍然只用一张显卡顺序执行全部任务，会造成其他 GPU 资源浪费。因此我在多显卡脚本中根据 JSON 中的 gpu_ids 启动多个 worker，每个 worker 绑定一张显卡，不同 worker 之间并行执行。

### 2. 多显卡执行流程

```text
读取 multi_gpu_tasks.json
    ↓
读取 global.gpu_ids，例如 ["0", "1"]
    ↓
启动 worker 0，绑定 GPU 0
启动 worker 1，绑定 GPU 1
    ↓
按照任务顺序进行轮询分配
    ↓
GPU 0 执行任务 1 → 任务 3
GPU 1 执行任务 2 → 任务 4
    ↓
等待所有 worker 完成
```

### 3. 分配方式说明

对于两个 GPU 和四个任务：

```text
任务列表：
1. fight_detection_exp
2. fall_detection_exp
3. running_detection_exp
4. loitering_detection_exp
```

按照轮询方式分配后：

```text
GPU 0：fight_detection_exp → running_detection_exp
GPU 1：fall_detection_exp → loitering_detection_exp
```

可以画成时间轴：

```text
时间轴 →
GPU 0：fight_detection_exp       → running_detection_exp
GPU 1：fall_detection_exp        → loitering_detection_exp
```

解释重点：

- GPU 0 和 GPU 1 是并行的；
- 每张 GPU 内部仍然是串行的；
- 这样可以避免同一张 GPU 上多个任务同时抢显存；
- 同时又能充分利用多张 GPU。

### 4. 多显卡讲稿

> 多显卡脚本的核心是“多 GPU 并行 + 单 GPU 内部串行”。脚本首先读取 JSON 文件中的 `gpu_ids` 字段，例如 `["0", "1"]` 表示使用两张显卡。随后脚本会根据 GPU 数量启动对应数量的 worker，每个 worker 通过 `CUDA_VISIBLE_DEVICES=$GPU_ID` 绑定到指定显卡。任务分配时采用轮询方式，例如第一个任务分配给 GPU 0，第二个任务分配给 GPU 1，第三个任务再分配给 GPU 0，第四个任务分配给 GPU 1。这样不同 GPU 上的任务可以同时执行，而同一张 GPU 内部仍然按顺序执行自己的任务队列，既提升了 GPU 利用率，又避免了显存冲突。

---

## 八、日志和异常处理

### 1. 日志保存

每个任务都有独立日志，例如：

```text
logs/single_gpu/fight_detection_exp.log
logs/multi_gpu/gpu0_fight_detection_exp.log
```

讲解语句：

> 为了方便后续查看训练过程，我为每个任务保存单独日志。训练过程中的输出和错误信息都会重定向到对应日志文件中。这样即使任务运行时间较长，也可以在任务结束后查看具体训练过程和报错信息。

### 2. 异常处理

脚本中会判断任务是否执行成功。如果某个任务返回非 0 状态码，说明执行失败，脚本会输出错误并退出或通知 worker 失败。

讲解语句：

> Shell 脚本还会检查每个任务的返回状态。如果任务执行失败，脚本会记录错误并提示查看对应日志文件。这样可以避免任务失败后还继续盲目执行后续任务，减少无效实验结果。

---

## 九、课堂演示步骤

### 1. 打开项目

用 VSCode 打开项目文件夹。

### 2. 执行单显卡模式

```bash
python run_by_python.py --mode single
```

观察输出：

```text
Python 调用 Shell
Shell 读取 single_gpu_tasks.json
任务 1 开始
任务 1 完成
任务 2 开始
任务 2 完成
...
```

### 3. 查看日志

```bash
ls logs/single_gpu
cat logs/single_gpu/fight_detection_exp.log
```

### 4. 执行多显卡模式

```bash
python run_by_python.py --mode multi
```

观察输出：

```text
WORKER-0 启动，绑定 GPU：0
WORKER-1 启动，绑定 GPU：1
WORKER-0 执行 fight_detection_exp
WORKER-1 执行 fall_detection_exp
...
```

### 5. 查看输出结果

```bash
ls outputs/multi_gpu
cat outputs/multi_gpu/fight_detection_exp/metrics.json
```

---

## 十、PPT 讲解建议

建议做 5 页 PPT：

### 第 1 页：设计背景

标题：实验任务自动化调度设计

内容：

- 多个实验任务参数不同，手动输入命令容易出错；
- 单显卡同时跑多个任务容易显存冲突；
- 多显卡顺序执行会浪费资源；
- 因此采用 Python + Shell + JSON 的任务调度方式。

### 第 2 页：整体架构

画图：

```text
Python 启动入口
    ↓
Shell 调度脚本
    ↓
JSON 配置文件
    ↓
训练程序
```

注意讲清楚各模块职责。

### 第 3 页：单显卡串行执行

画图：

```text
GPU 0：任务1 → 任务2 → 任务3 → 任务4
```

重点讲：

- 固定 `CUDA_VISIBLE_DEVICES=0`；
- Shell 循环执行任务；
- 当前任务未结束，下一个任务不会启动。

### 第 4 页：多显卡并行执行

画图：

```text
GPU 0：任务1 → 任务3
GPU 1：任务2 → 任务4
```

重点讲：

- 根据 `gpu_ids` 启动 worker；
- 不同 GPU 并行；
- 同一 GPU 内部串行。

### 第 5 页：优势总结

内容：

- 参数集中管理；
- 方便复现实验；
- 避免显存冲突；
- 提高多显卡利用率；
- 日志独立保存，便于排错。

---

## 十一、答辩可直接使用的完整讲稿

> 在实验任务调度部分，我采用了 Python 调用 Shell 脚本，并通过 JSON 文件统一管理配置信息的方式。整体上可以分为四个部分：Python 启动入口、Shell 调度脚本、JSON 配置文件和具体训练程序。Python 文件主要用于接收运行模式，例如单显卡模式或多显卡模式，然后通过 subprocess 调用对应的 Shell 脚本；Shell 脚本负责读取 JSON 配置文件，并根据配置中的任务列表组织任务执行；JSON 文件中保存每个任务的名称、配置路径、训练轮数、batch size、学习率、输出目录和 GPU 编号；训练程序则负责执行具体的训练任务。  
>
> 在单显卡情况下，我采用串行执行方式。因为只有一张显卡，如果多个训练任务同时启动，很容易出现显存不足的问题。因此，Shell 脚本读取 JSON 中的任务列表后，会按照顺序依次执行每个任务。每个任务执行时，通过 `CUDA_VISIBLE_DEVICES=0` 指定使用第 0 张显卡。由于 Shell 命令默认是阻塞式执行的，所以当前任务没有结束之前，脚本不会继续启动下一个任务，从而保证同一时间只有一个任务占用 GPU。  
>
> 在多显卡情况下，我采用多 worker 并行执行方式。Shell 脚本会读取 JSON 中的 `gpu_ids` 字段，例如 `["0", "1"]` 表示使用两张显卡。脚本会根据显卡数量启动两个 worker，每个 worker 绑定一张 GPU。任务分配采用轮询方式，例如第一个任务分配给 GPU 0，第二个任务分配给 GPU 1，第三个任务再分配给 GPU 0，第四个任务分配给 GPU 1。这样 GPU 0 和 GPU 1 可以同时运行任务，提高多显卡利用率；同时，每张 GPU 内部仍然是一个任务完成后再执行下一个任务，避免同一张显卡上多个任务同时运行导致显存冲突。  
>
> 此外，脚本还为每个任务设置了独立的日志文件和输出目录。训练过程中的输出和错误信息都会保存到日志文件中，训练结果则保存到对应的输出目录。这样后续查看实验过程、定位错误和对比实验结果都比较方便。整体来看，这种方式实现了参数配置、任务调度和训练执行的分离，提高了实验执行的自动化程度，也有利于后续复现实验。

---

## 十二、论文中可以这样写

### 基于 Python、Shell 与 JSON 的任务调度实现

为了提高实验任务执行的自动化程度，本文采用 Python 调用 Shell 脚本并结合 JSON 配置文件的方式实现训练任务调度。JSON 文件用于统一保存实验任务参数，包括任务名称、配置文件路径、训练轮数、批量大小、学习率、输出目录和 GPU 编号等。Python 脚本作为统一启动入口，根据用户选择的运行模式调用对应的 Shell 脚本。Shell 脚本读取 JSON 配置后，按照任务列表组织训练任务执行。

在单显卡环境下，系统采用串行执行策略。Shell 脚本通过 `CUDA_VISIBLE_DEVICES=0` 固定使用一张显卡，并按照 JSON 中任务顺序依次执行多个训练任务。由于 Shell 命令默认具有阻塞特性，当前任务未结束时后续任务不会启动，因此可以避免多个任务同时占用同一张 GPU 导致显存不足。

在多显卡环境下，系统采用多 worker 并行执行策略。Shell 脚本根据 JSON 中的 `gpu_ids` 启动多个 worker，每个 worker 绑定一张 GPU。不同 worker 之间并行执行任务，而同一 worker 内部仍然按顺序执行分配到的任务队列。该策略在提高多显卡资源利用率的同时，也避免了单张 GPU 内部的显存竞争问题。每个任务的运行日志和输出结果均保存到独立目录中，便于后续实验复现、结果对比和错误排查。

---

## 十三、常见问题

### 1. 为什么不用 Python 直接完成所有调度？

可以回答：

> Python 当然也可以完成全部调度，但 Shell 在执行系统命令、设置环境变量、重定向日志和启动后台任务方面更直接。因此这里采用 Python 作为统一入口，Shell 负责底层任务调度，两者结合更加清晰。

### 2. 为什么不用多个任务同时跑在一张 GPU 上？

可以回答：

> 深度学习训练任务通常显存占用较高，多个任务同时运行在一张 GPU 上容易导致显存不足。因此单张 GPU 内部保持串行执行更稳妥。

### 3. 多显卡并行是不是所有任务同时启动？

可以回答：

> 不是所有任务都同时启动，而是每张 GPU 启动一个 worker。不同 worker 可以同时运行，但同一个 worker 内部仍然按照任务顺序执行。这样既能并行利用多张 GPU，又能避免单卡内部显存冲突。

### 4. JSON 文件有什么好处？

可以回答：

> JSON 可以把参数从脚本中分离出来，修改实验参数时不需要改调度脚本。新增任务、关闭任务、调整训练轮数和输出目录都可以直接修改 JSON 文件完成。

### 5. 如果一个任务失败怎么办？

可以回答：

> Shell 脚本会根据训练程序返回的状态码判断是否执行成功。如果任务失败，会输出错误提示，并提示查看对应日志文件。这样可以快速定位问题。
