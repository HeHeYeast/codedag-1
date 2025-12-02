# CodeDAG 实验指南

本文档描述如何使用 CodeDAG 对 Kaggle 比赛数据管道进行优化实验。

## 1. 概述

### 1.1 实验目标

对比 **基线训练** 与 **CodeDAG 优化后训练** 的性能差异，主要监控：

- CPU 利用率
- GPU 显存使用
- 训练 Loss
- 准确率
- Batch 处理时间

### 1.2 实验流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        实验流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │  1. 基线训练  │  直接运行原始训练代码，记录指标                │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  2. DAG 追踪  │  对数据管道进行追踪（如 DataLoader 迭代）     │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  3. 图优化    │  分区、粗化、生成迁移计划                     │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  4. 优化训练  │  应用迁移，运行训练，记录指标                 │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  5. 结果对比  │  生成对比报告                                 │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 环境准备

### 2.1 Docker 环境搭建

实验采用 Docker 容器化，为每个 Kaggle 竞赛创建隔离的运行环境。

#### 目录结构

```
experiments/
├── docker/                        # Docker 镜像文件
│   ├── Dockerfile.pytorch        # PyTorch 基础镜像
│   └── Dockerfile.tensorflow     # TensorFlow 基础镜像
├── setup_environment.sh          # 环境搭建脚本
├── setup_experiment.py           # 实验初始化脚本
├── monitoring_tools.py           # 指标监控工具
└── kaggle_[competition]/         # 具体竞赛目录
    ├── requirements.txt          # Python 额外依赖
    ├── experiment_config.json    # 实验配置
    ├── baseline.py              # 基准模式
    └── optimization.py          # 优化模式
```

#### 环境搭建流程

```bash
cd experiments

# 1. 检查环境
./setup_environment.sh check

# 2. 构建基础 Docker 镜像
./setup_environment.sh build

# 3. 创建新实验
python setup_experiment.py birdclef_2023 --framework pytorch

# 4. 设置实验环境
./setup_environment.sh setup kaggle_birdclef_2023

# 5. 运行实验
cd kaggle_birdclef_2023
../setup_environment.sh run baseline       # 基准模式
../setup_environment.sh run optimization   # 优化模式
```

### 2.2 项目结构

```
codedag_clean/
├── core/                    # 核心追踪模块
│   └── enhanced_tracer.py
├── optimizer/               # 图优化模块
│   └── optimizer_manager.py
├── migration/               # 迁移模块
│   └── api.py
├── experiments/             # 实验相关
│   └── monitoring_tools.py  # 指标监控工具（Timer, ResourceMonitor, MetricsLogger, ExperimentMonitor）
├── docs/
│   └── EXPERIMENT_GUIDE.md  # 本文档
└── results/                 # 实验结果输出
```

### 2.3 依赖安装（非 Docker 环境）

```bash
# 核心依赖
pip install torch torchvision torchaudio
pip install numpy

# 监控依赖
pip install psutil GPUtil

# 可视化（可选）
pip install matplotlib
```

## 3. 核心 API 使用

### 3.1 CodeDAG 三阶段流程

CodeDAG 的核心流程包含三个阶段，参考 `test_harness.py` 的实现：

```python
from core.enhanced_tracer import EnhancedTracer
from optimizer.optimizer_manager import OptimizerManager
from migration.api import PipelineMigrator

# ========== 阶段 1: 追踪 ==========
tracer = EnhancedTracer(max_depth=10, track_memory=True)

# 追踪数据管道函数（如 DataLoader 的迭代）
with tracer.tracing_context():
    result = target_func(*args, **kwargs)

# 获取 DAG
dag = tracer.dag_builder.dag
print(f"节点数: {len(dag.nodes)}, 边数: {len(dag.edges)}")

# ========== 阶段 2: 优化 ==========
optimizer = OptimizerManager(k=2, coarsen_max_depth=5, max_iterations=50)
opt_result = optimizer.optimize(dag, export_dir="./results")

# 获取分区结果
partitions = opt_result['optimized_partitions']
coarsened_dag = opt_result['coarsened_dag']

# 生成迁移计划
migration_plan = optimizer.get_migration_plan(
    partitions=partitions,
    dag=coarsened_dag,
    gpu_count=1
)

# ========== 阶段 3: 迁移 ==========
migrator = PipelineMigrator.from_plan(migration_plan, default_device="cuda:0")

try:
    # 应用迁移
    apply_stats = migrator.apply()
    print(f"Patched: {apply_stats.get('patched', 0)} 个函数")

    # 执行迁移后的函数
    migrated_result = target_func(*args, **kwargs)

finally:
    # 必须恢复环境
    migrator.restore()
```

### 3.2 追踪目标说明

**重要**：追踪目标应该是数据管道的核心函数，而不是单个 `__getitem__`：

| 框架 | 追踪目标 | 说明 |
|------|----------|------|
| PyTorch DataLoader | `iterator.__next__` | 获取下一个 batch 的完整流程 |
| TensorFlow Dataset | `iterator.get_next()` | tf.data 迭代 |
| 自定义管道 | 数据处理主函数 | 包含完整预处理逻辑的函数 |

**示例：追踪 PyTorch DataLoader**

```python
dataloader = DataLoader(dataset, batch_size=32)
iterator = iter(dataloader)

# 追踪获取一个 batch 的完整流程
with tracer.tracing_context():
    batch = next(iterator)
```

### 3.3 指标监控

使用 `monitoring_tools.py` 中的 `ExperimentMonitor` 进行监控：

```python
from experiments.monitoring_tools import ExperimentMonitor

# 创建监控器
monitor = ExperimentMonitor("BirdCLEF_2023", save_dir="./results")

# ===== 基线训练 =====
monitor.start_mode("baseline", config={"epochs": 2, "batch_size": 32})

for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        monitor.start_batch()

        # 训练逻辑
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 记录 batch
        monitor.end_batch()
        monitor.log_loss(epoch, batch_idx, loss.item())
        monitor.log_accuracy(epoch, batch_idx, accuracy)

monitor.stop_mode()

# ===== 优化训练 =====
monitor.start_mode("optimization", config={"epochs": 2, "batch_size": 32})
# ... 同样的训练循环，但应用了 migrator ...
monitor.stop_mode()

# ===== 保存结果 =====
monitor.save_json()      # 保存 JSON 数据
monitor.plot_metrics()   # 生成对比图表
```

### 3.4 监控组件说明

`monitoring_tools.py` 提供以下组件：

| 组件 | 功能 |
|------|------|
| `Timer` | 记录 batch 时间和训练总时间 |
| `ResourceMonitor` | 后台监控 CPU/GPU 使用率和内存（支持 Docker cgroup） |
| `MetricsLogger` | 记录 loss 和 accuracy |
| `ExperimentMonitor` | 统一管理器，整合上述组件 |

**ResourceMonitor 特性**：
- 自动检测 Docker 环境
- 支持 cgroup v1/v2 CPU 监控
- 支持多 GPU 监控
- 可配置采样间隔

## 4. 监控指标说明

### 4.1 时间指标

| 指标 | 说明 |
|------|------|
| `batch_times` | 每个 batch 的处理时间 `[(global_batch_idx, time)]` |
| `total_time` | 训练总时间 |

### 4.2 资源指标

| 指标 | 说明 |
|------|------|
| `cpu_usage` | CPU 使用率 `[(timestamp, percent)]` |
| `memory_usage` | 内存使用率 `[(timestamp, percent)]` |
| `gpu_usage` | GPU 计算利用率 `{gpu_id: [(timestamp, percent)]}` |
| `gpu_memory` | GPU 显存使用 `{gpu_id: [(timestamp, MB)]}` |

### 4.3 训练指标

| 指标 | 说明 |
|------|------|
| `losses` | 训练损失 `[(epoch, batch, loss)]` |
| `accuracies` | 准确率 `[(epoch, batch, accuracy)]` |

## 5. 配置选项

### 5.1 EnhancedTracer 配置

```python
tracer = EnhancedTracer(
    max_depth=10,        # 追踪最大深度
    track_memory=True,   # 是否追踪内存分配
)
```

### 5.2 OptimizerManager 配置

```python
optimizer = OptimizerManager(
    k=2,                  # 分区数量
    coarsen_max_depth=5,  # 粗化最大深度
    max_iterations=50,    # 优化最大迭代次数
)
```

### 5.3 ExperimentMonitor 配置

```python
monitor = ExperimentMonitor(
    experiment_name="my_exp",
    save_dir="./results",
)

# 开始模式时可配置资源监控间隔
monitor.start_mode("baseline", resource_interval=1.0)  # 1秒采样一次
```

## 6. 输出格式

### 6.1 结果 JSON 结构

```json
{
  "experiment_name": "BirdCLEF_2023",
  "timestamp": 1705312200,
  "datetime": "2024-01-15T10:30:00",
  "modes": {
    "baseline": {
      "config": {"epochs": 2, "batch_size": 32},
      "batch_times": [[0, 0.15], [1, 0.14], ...],
      "resource_usage": {
        "cpu_usage": [[1705312200, 45.2], ...],
        "gpu_usage": {"0": [[1705312200, 78.5], ...]},
        "gpu_memory": {"0": [[1705312200, 2048], ...]}
      },
      "metrics": {
        "losses": [[0, 0, 2.5], [0, 1, 2.3], ...],
        "accuracies": [[0, 0, 0.65], ...]
      },
      "summary": {
        "total_time": 120.5,
        "batch_time": {"mean": 0.145, "std": 0.02, "min": 0.12, "max": 0.18},
        "loss": {"initial": 2.5, "final": 0.8, "min": 0.8, "max": 2.5},
        "accuracy": {"initial": 0.65, "final": 0.92, "best": 0.92}
      }
    },
    "optimization": {
      ...
    }
  }
}
```

## 7. 实验示例

### 7.1 设计原则

**重要**：实验框架的目标是**包装原有训练代码**，而不是重写训练逻辑：

- 保持原有的 Dataset、Model、训练循环不变
- 只对数据管道进行追踪和优化
- 通过 `migrator.apply()` 透明地应用优化

```
┌─────────────────────────────────────────────────────────────┐
│                    原有 Kaggle 训练代码                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Dataset   │  │    Model    │  │   Training Loop     │  │
│  │ (保持不变)  │  │ (保持不变)  │  │    (保持不变)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    CodeDAG 实验包装                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. ExperimentMonitor: 监控指标                      │    │
│  │  2. EnhancedTracer: 追踪数据管道                     │    │
│  │  3. PipelineMigrator: 透明应用优化                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 BirdCLEF 2023 实验示例

```python
"""
BirdCLEF 2023 CodeDAG 优化实验

说明：
- 训练逻辑完全复用原有 baseline.py 的代码
- 只添加监控和数据管道优化的包装
"""

import sys
sys.path.insert(0, '/mnt/sda/gxy/codedag_clean')

import torch
from torch.utils.data import DataLoader

# CodeDAG 组件
from core.enhanced_tracer import EnhancedTracer
from optimizer.optimizer_manager import OptimizerManager
from migration.api import PipelineMigrator
from experiments.monitoring_tools import ExperimentMonitor

# ============================================================
# 导入原有 baseline.py 的所有组件（不做任何修改）
# ============================================================
from baseline import (
    BirdCLEFDataset,
    BirdCLEFModel,
    CONFIG,
    df_train,          # 原有数据
    # 如果 baseline.py 有独立的 train_epoch 函数，也可以导入
)


def train_epoch_with_monitor(model, dataloader, optimizer, criterion, epoch, monitor):
    """
    带监控的训练循环

    注意：这里的训练逻辑应该与原有 baseline.py 保持一致，
    只是添加了 monitor 的调用来记录指标。
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        monitor.start_batch()

        # ===== 以下是原有的训练逻辑，保持不变 =====
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # ===== 原有训练逻辑结束 =====

        # 记录指标
        monitor.end_batch()
        monitor.log_loss(epoch, batch_idx, loss.item())

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        accuracy = correct / total
        monitor.log_accuracy(epoch, batch_idx, accuracy)

        total_loss += loss.item()

    return total_loss / len(dataloader), correct / total


def run_experiment():
    """运行对比实验"""

    # ============================================================
    # 使用原有代码创建数据集和 DataLoader（不做修改）
    # ============================================================
    dataset = BirdCLEFDataset(df_train, CONFIG)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['train_batch_size'],
        shuffle=True,
        num_workers=CONFIG.get('num_workers', 0)
    )

    criterion = torch.nn.CrossEntropyLoss()
    monitor = ExperimentMonitor("BirdCLEF_2023", save_dir="./results")

    # ============================================================
    # 阶段 1: 基线训练（使用原有代码）
    # ============================================================
    print("=" * 60)
    print("阶段 1: 基线训练")
    print("=" * 60)

    model = BirdCLEFModel(CONFIG).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    monitor.start_mode("baseline", config=CONFIG)
    for epoch in range(CONFIG['epochs']):
        loss, acc = train_epoch_with_monitor(
            model, dataloader, optimizer, criterion, epoch, monitor
        )
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    monitor.stop_mode()

    # ============================================================
    # 阶段 2: CodeDAG 追踪与优化（只针对数据管道）
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段 2: CodeDAG 数据管道优化")
    print("=" * 60)

    # 追踪 DataLoader 的迭代过程
    tracer = EnhancedTracer(max_depth=10, track_memory=True)
    iterator = iter(dataloader)

    with tracer.tracing_context():
        _ = next(iterator)  # 追踪获取一个 batch 的完整流程

    dag = tracer.dag_builder.dag
    print(f"DAG 构建完成: {len(dag.nodes)} 节点, {len(dag.edges)} 边")

    # 图优化
    opt_mgr = OptimizerManager(k=2)
    opt_result = opt_mgr.optimize(dag)

    # 生成迁移计划
    migration_plan = opt_mgr.get_migration_plan(
        opt_result['optimized_partitions'],
        opt_result['coarsened_dag'],
        gpu_count=1
    )
    print(f"迁移计划: {len(migration_plan.get('context_device_map', {}))} 个上下文映射")

    # ============================================================
    # 阶段 3: 优化后训练（训练代码不变，只应用数据管道优化）
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段 3: 优化后训练")
    print("=" * 60)

    # 重新初始化模型（保证公平对比）
    model = BirdCLEFModel(CONFIG).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 创建迁移器
    migrator = PipelineMigrator.from_plan(migration_plan)

    monitor.start_mode("optimization", config=CONFIG)
    try:
        # 应用数据管道优化（透明地修改底层函数）
        migrator.apply()

        # 训练代码与基线完全相同
        for epoch in range(CONFIG['epochs']):
            loss, acc = train_epoch_with_monitor(
                model, dataloader, optimizer, criterion, epoch, monitor
            )
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    finally:
        # 必须恢复原始函数
        migrator.restore()
    monitor.stop_mode()

    # ============================================================
    # 阶段 4: 保存结果和对比
    # ============================================================
    print("\n" + "=" * 60)
    print("保存实验结果")
    print("=" * 60)

    monitor.save_json()
    monitor.plot_metrics()


if __name__ == "__main__":
    run_experiment()
```

### 7.3 关键点说明

1. **训练代码不变**：`train_epoch_with_monitor` 中的核心训练逻辑与原有 `baseline.py` 完全一致

2. **追踪目标**：追踪的是 `next(iterator)`，即 DataLoader 获取一个 batch 的完整流程，包括：
   - Dataset 的 `__getitem__` 调用
   - 数据预处理和增强
   - Collate 函数

3. **透明优化**：`migrator.apply()` 会自动修改底层函数，训练代码无需感知

4. **公平对比**：基线和优化训练使用相同的随机种子和初始化

## 8. 常见问题

### Q1: 追踪深度设置多少合适？

建议从 `max_depth=10` 开始。如果 DAG 节点过少，可适当增加；如果追踪时间过长，可减少。

### Q2: 如何确定分区数 k？

一般设置 `k=2`（CPU 分区 + GPU 分区）。如果有多 GPU 环境，可设置 `k = gpu_count + 1`。

### Q3: 迁移后性能没有提升？

可能原因：
1. 数据管道本身计算量不大，不是瓶颈
2. 追踪深度不够，关键操作未被捕获
3. 操作不支持 GPU 迁移

### Q4: 如何只监控不优化？

```python
monitor = ExperimentMonitor("baseline_only")
monitor.start_mode("baseline")
# ... 训练代码 ...
monitor.stop_mode()
monitor.save_json()
```

### Q5: 迁移后出现错误如何恢复？

使用 `try...finally` 确保恢复：

```python
migrator = PipelineMigrator.from_plan(plan)
try:
    migrator.apply()
    # 执行代码...
finally:
    if migrator.is_applied():
        migrator.restore()
```

## 9. 参考资料

- [CodeDAG API 参考](./API_REFERENCE.md)
- [迁移策略说明](./migration.md)
- [分区算法说明](./kway.md)
- [测试框架参考](../examples/integration_tests/test_harness.py)
