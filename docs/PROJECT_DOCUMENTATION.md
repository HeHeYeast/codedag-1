# CodeDAG 项目文档

## 项目概述

CodeDAG 是一个数据流图追踪、优化与迁移框架，用于分析 Python 代码的执行流程，构建有向无环图(DAG)，并支持将 CPU 上的数据处理管道自动迁移到 GPU 上执行。

### 核心功能

1. **代码追踪** - 自动追踪 Python 代码执行，构建数据流图
2. **图优化** - 图粗化、K-way 划分、迭代优化
3. **代码迁移** - 将 CPU 函数自动替换为 GPU 等价实现
4. **可视化** - 生成数据流图的 SVG 可视化

---

## 项目结构

```
codedag_clean/
├── core/                    # 核心追踪和DAG构建模块
│   ├── base_tracer.py       # 基础追踪器
│   ├── enhanced_tracer.py   # 增强追踪器
│   ├── dag_builder.py       # DAG构建器
│   ├── memory_profiler.py   # 内存分析器
│   ├── performance_monitor.py # 性能监控器
│   ├── function_filter.py   # 函数过滤器
│   └── exceptions.py        # 异常定义
├── utils/                   # 工具模块
│   ├── device_detector.py   # 设备检测器
│   ├── device_profiler.py   # 设备性能分析器
│   ├── unified_device_detector.py # 统一设备检测器
│   └── device_detector_v2.py # 设备检测器v2
├── optimizer/               # 优化器模块
│   ├── optimizer_manager.py # 优化管理器
│   ├── graph_coarsening.py  # 图粗化聚类器
│   ├── kway_partitioner.py  # K-way图划分器
│   └── iterative_optimizer.py # 迭代优化器
├── migration/               # 迁移模块
│   ├── api.py               # 迁移API
│   ├── core.py              # 核心迁移逻辑
│   ├── processors.py        # 数据处理器
│   ├── dispatchers.py       # 动态分发器
│   └── registry/            # 策略注册表
│       ├── base.py          # 注册表基类
│       ├── cv_specs.py      # OpenCV迁移策略
│       ├── numpy_specs.py   # NumPy迁移策略
│       ├── pytorch_specs.py # PyTorch迁移策略
│       └── audio_specs.py   # 音频处理迁移策略
├── visualization/           # 可视化模块
│   └── graphviz_visualizer.py # Graphviz可视化器
├── examples/                # 示例代码
│   ├── dataflow_tests/      # 数据流测试
│   └── optimizer_tests/     # 优化器测试
├── config.py                # 全局配置
└── __init__.py              # 包入口
```

---

## 模块详解

### 1. Core 模块 (`core/`)

核心模块负责代码执行追踪和数据流图构建。

#### 1.1 EnhancedTracer (增强追踪器)

**文件**: `core/enhanced_tracer.py`

主要类，负责追踪 Python 代码执行并构建数据流图。

```python
from core.enhanced_tracer import EnhancedTracer

# 创建追踪器
tracer = EnhancedTracer(
    max_depth=5,           # 最大追踪深度
    track_memory=True,     # 是否追踪内存
    filter_config=None     # 函数过滤配置
)

# 使用上下文管理器进行追踪
with tracer.tracing_context():
    result = my_function()

# 导出数据流图
dataflow = tracer.export_dataflow_graph()

# 导出可视化
tracer.export_visualization("output.svg")
```

**主要接口**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `max_depth`, `track_memory`, `filter_config` | - | 初始化追踪器 |
| `tracing_context()` | - | ContextManager | 返回追踪上下文管理器 |
| `export_dataflow_graph()` | - | `Dict` | 导出数据流图为字典 |
| `export_visualization(path)` | `path: str` | `bool` | 导出SVG可视化 |

#### 1.2 DAGBuilder (DAG构建器)

**文件**: `core/dag_builder.py`

负责构建和维护执行图。

```python
from core.dag_builder import DAGBuilder, ExecutionDAG, DAGNode

# DAGNode - 图节点
node = DAGNode(
    node_id=1,
    name="function_name",
    node_type="function_call"  # function_call | variable | operator
)
node.performance = {
    'execution_time': 0.001,
    'memory_usage': 1024
}

# ExecutionDAG - 执行图
dag = ExecutionDAG()
dag.nodes[node.node_id] = node
dag.edges.append((source_id, target_id, edge_type))
```

**DAGNode 属性**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `node_id` | `int` | 节点唯一标识 |
| `name` | `str` | 节点名称 |
| `node_type` | `str` | 节点类型 |
| `performance` | `Dict` | 性能数据 |
| `attributes` | `Dict` | 扩展属性 |

**ExecutionDAG 属性**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `nodes` | `Dict[int, DAGNode]` | 节点字典 |
| `edges` | `List[Tuple]` | 边列表 `(src, tgt, type)` |

#### 1.3 MemoryProfiler (内存分析器)

**文件**: `core/memory_profiler.py`

追踪内存分配和变量快照。

```python
from core.memory_profiler import MemoryProfiler

profiler = MemoryProfiler()
profiler.start()

# ... 执行代码 ...

snapshot = profiler.take_snapshot()
profiler.stop()
```

#### 1.4 PerformanceMonitor (性能监控器)

**文件**: `core/performance_monitor.py`

监控函数执行时间和资源使用。

```python
from core.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.measure():
    result = function()

stats = monitor.get_stats()
```

---

### 2. Utils 模块 (`utils/`)

设备检测和性能分析工具。

#### 2.1 UnifiedDeviceDetector (统一设备检测器)

**文件**: `utils/unified_device_detector.py`

检测可用的计算设备 (CPU/GPU)。

```python
from utils.unified_device_detector import UnifiedDeviceDetector

detector = UnifiedDeviceDetector()

# 检测所有可用设备
devices = detector.detect_all()

# 获取最佳设备
best_device = detector.get_best_device()
```

**主要接口**:

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `detect_all()` | `List[DeviceInfo]` | 检测所有设备 |
| `get_best_device()` | `DeviceInfo` | 获取最佳设备 |
| `get_cuda_devices()` | `List[DeviceInfo]` | 获取CUDA设备 |

#### 2.2 DeviceProfiler (设备性能分析器)

**文件**: `utils/device_profiler.py`

分析设备计算能力和内存带宽。

```python
from utils.device_profiler import DeviceProfiler

profiler = DeviceProfiler()

# 运行基准测试
benchmark = profiler.run_benchmark(device="cuda:0")

# 获取设备能力
capability = profiler.get_compute_capability()
```

---

### 3. Optimizer 模块 (`optimizer/`)

图优化和划分模块。

#### 3.1 OptimizerManager (优化管理器)

**文件**: `optimizer/optimizer_manager.py`

优化流程的主入口，协调图粗化、划分和迭代优化。

```python
from optimizer import OptimizerManager

# 创建优化管理器
optimizer = OptimizerManager(
    k=4,                    # 分区数量
    coarsen_max_depth=5,    # 粗化最大深度
    max_iterations=200      # 迭代优化最大轮数
)

# 执行优化
result = optimizer.optimize(dag, export_dir="output/")

# 获取迁移计划
migration_plan = optimizer.get_migration_plan(
    partitions=result['partitions'],
    dag=result['dag'],
    gpu_count=1
)
```

**主要接口**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `optimize(dag, export_dir)` | DAG, 导出目录 | `Dict` | 执行完整优化流程 |
| `get_migration_plan(...)` | 分区结果, DAG, GPU数量 | `Dict` | 生成迁移计划 |

**优化流程**:

```
原始DAG
   ↓
图粗化 (GraphCoarsening)
   ↓
K-way划分 (KWayPartitioner)
   ↓
迭代优化 (IterativeOptimizer)
   ↓
迁移计划
```

#### 3.2 GraphCoarsening (图粗化聚类器)

**文件**: `optimizer/graph_coarsening.py`

识别重复的子图模式并合并，减少图规模。

```python
from optimizer.graph_coarsening import GraphCoarsening

coarsener = GraphCoarsening(
    max_depth=5,            # 子图提取最大深度
    anchor_patterns=[]      # 锚点模式列表
)

coarsened_dag = coarsener.coarsen(dag)
stats = coarsener.get_stats()
```

**核心算法**:

1. **锚点识别** - 识别重复出现的函数调用
2. **子图提取** - BFS从锚点提取关联子图
3. **Hash计算** - 计算子图结构哈希
4. **同构合并** - 合并具有相同哈希的子图

#### 3.3 KWayPartitioner (K-way图划分器)

**文件**: `optimizer/kway_partitioner.py`

基于动态贪心策略的图划分算法。

```python
from optimizer.kway_partitioner import KWayPartitioner, Partition

partitioner = KWayPartitioner(
    k=4,                    # 分区数量
    lambda_weight=0.5,      # 瓶颈选择距离权重
    alpha=0.5,              # 性能权重
    beta=0.3,               # 距离惩罚权重
    gamma=0.2               # 负载惩罚权重
)

partitions = partitioner.partition(dag)
stats = partitioner.get_stats(partitions)
edge_cut = partitioner.compute_edge_cut(partitions, dag)
```

**Partition 数据结构**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `id` | `int` | 分区ID |
| `center` | `int` | 中心节点ID |
| `nodes` | `Set[int]` | 分区内节点集合 |
| `total_compute` | `float` | 总计算量 |
| `total_memory` | `float` | 总内存量 |

**划分流程**:

1. **预处理** - 计算图直径、节点权重
2. **瓶颈选择** - 选择K个关键节点作为分区中心
3. **初始化** - 创建初始分区
4. **贪心分配** - 迭代将节点分配到最优分区
5. **孤立处理** - 处理未分配的节点

#### 3.4 IterativeOptimizer (迭代优化器)

**文件**: `optimizer/iterative_optimizer.py`

基于贪心迭代的分区优化算法。

```python
from optimizer.iterative_optimizer import IterativeOptimizer

optimizer = IterativeOptimizer(
    alpha=0.4,              # 负载均衡权重
    beta=0.6,               # 关键路径权重
    max_iterations=200,     # 最大迭代次数
    no_improvement_threshold=20  # 无改善终止阈值
)

optimized_partitions = optimizer.optimize(partitions, dag)
stats = optimizer.get_stats()
```

**优化目标函数**:

```
J = α · CV(loads) + β · normalized_max_cp
```

- `CV(loads)`: 负载变异系数
- `normalized_max_cp`: 归一化最大关键路径

---

### 4. Migration 模块 (`migration/`)

代码迁移模块，将 CPU 函数自动替换为 GPU 实现。

#### 4.1 PipelineMigrator (管道迁移器)

**文件**: `migration/api.py`

迁移模块的主入口类。

```python
from migration import PipelineMigrator

# 方式1: 从JSON加载
migrator = PipelineMigrator.from_json("migration_plan.json")

# 方式2: 从优化器获取
migrator = PipelineMigrator.from_optimizer(
    optimizer,
    partitions=partitions,
    dag=dag,
    gpu_count=1
)

# 方式3: 自定义配置
from migration import create_custom_migrator
migrator = create_custom_migrator({
    "cv2.resize": "cuda:0",
    "cv2.cvtColor": "cuda:0"
})

# 应用迁移
migrator.apply()

# 执行代码 (函数会自动迁移到GPU)
result = pipeline()

# 恢复原始函数
migrator.restore()

# 或使用上下文管理器
with PipelineMigrator.from_json("plan.json") as migrator:
    result = pipeline()
# 自动恢复
```

**主要接口**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `from_json(path)` | JSON路径 | `PipelineMigrator` | 从JSON加载 |
| `from_plan(plan)` | 计划字典 | `PipelineMigrator` | 从字典加载 |
| `from_optimizer(...)` | 优化器等 | `PipelineMigrator` | 从优化器获取 |
| `apply()` | - | `Dict[str, int]` | 应用迁移Patch |
| `restore()` | - | `int` | 恢复原始函数 |

#### 4.2 OpSpec (操作规格)

**文件**: `migration/core.py`

定义函数迁移的规格说明。

```python
from migration.core import OpSpec, StrategyType

spec = OpSpec(
    source="cv2.resize",              # 源函数路径
    strategy=StrategyType.STANDARD_OP, # 迁移策略
    priority=1,                        # 优先级
    target_lib="kornia.geometry.transform",  # 目标库
    target_func="resize",              # 目标函数
    args_trans={0: "cv2_hwc_to_bchw"}, # 位置参数转换
    kwargs_trans={},                   # 关键字参数转换
    arg_renames={"dsize": "size"},     # 参数改名
    arg_value_maps={"dsize": "swap_hw"}, # 参数值映射
    injected_kwargs={},                # 注入参数
    structural_ops=[],                 # 结构性操作
    output_rule="keep_on_device"       # 输出处理规则
)
```

**策略类型 (StrategyType)**:

| 策略 | 说明 | 示例 |
|------|------|------|
| `MOVE_ONLY` | 仅搬运Tensor到GPU | PyTorch原生操作 |
| `STANDARD_OP` | 函数替换 | cv2.resize → kornia.resize |
| `FACTORY_OP` | 对象工厂模式 | librosa → torchaudio class |
| `PIPELINE_OP` | 管道容器 | Compose |
| `NON_MIGRATABLE` | 不可迁移 | I/O操作 |

#### 4.3 ProcessorRegistry (处理器注册表)

**文件**: `migration/processors.py`

提供各种数据转换处理器。

**输入处理器**:

| 名称 | 说明 |
|------|------|
| `ensure_tensor` | 通用Tensor转换 |
| `cv2_hwc_to_bchw` | OpenCV HWC → PyTorch BCHW |
| `pil_to_tensor` | PIL Image → Tensor |
| `audio_to_tensor` | 音频数据 → Tensor |

**值映射器**:

| 名称 | 说明 |
|------|------|
| `swap_hw` | 交换 (W,H) 为 (H,W) |
| `cv2_interp_mode` | cv2插值模式转换 |
| `cv2_color_code` | cv2颜色代码转换 |

**输出处理器**:

| 名称 | 说明 |
|------|------|
| `keep_on_device` | 保持在GPU |
| `to_numpy` | 转回NumPy |
| `bchw_to_hwc` | BCHW → HWC格式 |

#### 4.4 DispatcherRegistry (分发器注册表)

**文件**: `migration/dispatchers.py`

处理一对多的函数映射场景。

```python
from migration.dispatchers import DispatcherRegistry

# 注册自定义分发器
DispatcherRegistry.register("custom.func", my_dispatcher)

# 获取分发器
dispatcher = DispatcherRegistry.get("cv2.cvtColor")
```

**内置分发器**:

| 函数 | 说明 |
|------|------|
| `cv2.cvtColor` | 根据颜色代码选择kornia函数 |
| `cv2.threshold` | 根据阈值类型选择实现 |
| `PIL.Image.transpose` | 根据方法选择翻转/旋转 |
| `numpy.random.*` | NumPy随机函数GPU实现 |

---

### 5. Visualization 模块 (`visualization/`)

数据流图可视化。

#### 5.1 GraphvizDataflowVisualizer

**文件**: `visualization/graphviz_visualizer.py`

基于Graphviz的可视化器。

```python
from visualization import GraphvizDataflowVisualizer, create_dataflow_visualization

# 方式1: 从追踪器创建
success = create_dataflow_visualization(tracer, "output.svg")

# 方式2: 手动创建
visualizer = GraphvizDataflowVisualizer()
success = visualizer.generate_dataflow_svg(
    nodes=dag.nodes.values(),
    edges=dag.edges,
    output_path="output.svg",
    title="数据流图"
)

# 生成图例
visualizer.generate_legend_svg("legend.svg")
```

**节点样式**:

| 类型 | 形状 | 颜色 |
|------|------|------|
| `function` | 圆角矩形 | 浅绿色 |
| `variable` | 椭圆 | 浅蓝色 |
| `operator` | 菱形 | 橙色 |

**边类型**:

| 类型 | 颜色 | 说明 |
|------|------|------|
| `creates` | 绿色 | 创建变量 |
| `uses` | 蓝色 | 使用变量 |
| `produces` | 红色 | 产生输出 |
| `calls` | 紫色 | 函数调用 |

---

## 完整工作流程

### 典型使用流程

```python
import json
from core.enhanced_tracer import EnhancedTracer
from optimizer import OptimizerManager
from migration import PipelineMigrator

# ========== 阶段1: 代码追踪 ==========
tracer = EnhancedTracer(max_depth=8, track_memory=True)

with tracer.tracing_context():
    # 执行要分析的代码
    result = data_pipeline()

# 导出数据流图
dataflow = tracer.export_dataflow_graph()
tracer.export_visualization("dataflow.svg")

# ========== 阶段2: 图优化 ==========
dag = tracer.dag_builder.dag

optimizer = OptimizerManager(
    k=4,                    # 分区数
    coarsen_max_depth=5,
    max_iterations=100
)

# 执行优化
result = optimizer.optimize(dag, export_dir="output/")

# 生成迁移计划
migration_plan = optimizer.get_migration_plan(
    partitions=result['optimized_partitions'],
    dag=result['coarsened_dag'],
    gpu_count=1
)

# 保存迁移计划
with open("migration_plan.json", "w") as f:
    json.dump(migration_plan, f, indent=2)

# ========== 阶段3: 代码迁移 ==========
migrator = PipelineMigrator.from_plan(migration_plan)

with migrator:
    # 在此上下文中，指定的函数会自动迁移到GPU执行
    result = data_pipeline()
```

### PyTorch DataLoader 示例

```python
from torch.utils.data import Dataset, DataLoader
from core.enhanced_tracer import EnhancedTracer
from optimizer import OptimizerManager

class ImageDataset(Dataset):
    def __getitem__(self, idx):
        image = self.load_image(idx)
        image = self.resize(image)
        image = self.normalize(image)
        return image

# 追踪数据加载
tracer = EnhancedTracer(max_depth=8)
dataloader = DataLoader(ImageDataset(), batch_size=4)

with tracer.tracing_context():
    for batch in dataloader:
        break  # 只追踪一个batch

# 优化
dag = tracer.dag_builder.dag
optimizer = OptimizerManager(k=2)
result = optimizer.optimize(dag)
```

---

## API 参考快速索引

### Core 模块

| 类 | 文件 | 主要方法 |
|---|------|---------|
| `EnhancedTracer` | `core/enhanced_tracer.py` | `tracing_context()`, `export_dataflow_graph()` |
| `DAGBuilder` | `core/dag_builder.py` | `build()`, `add_node()`, `add_edge()` |
| `DAGNode` | `core/dag_builder.py` | 数据类 |
| `ExecutionDAG` | `core/dag_builder.py` | 数据类 |

### Optimizer 模块

| 类 | 文件 | 主要方法 |
|---|------|---------|
| `OptimizerManager` | `optimizer/optimizer_manager.py` | `optimize()`, `get_migration_plan()` |
| `GraphCoarsening` | `optimizer/graph_coarsening.py` | `coarsen()`, `get_stats()` |
| `KWayPartitioner` | `optimizer/kway_partitioner.py` | `partition()`, `get_stats()` |
| `IterativeOptimizer` | `optimizer/iterative_optimizer.py` | `optimize()`, `get_stats()` |

### Migration 模块

| 类 | 文件 | 主要方法 |
|---|------|---------|
| `PipelineMigrator` | `migration/api.py` | `apply()`, `restore()`, `from_json()` |
| `OpSpec` | `migration/core.py` | 数据类 |
| `ProcessorRegistry` | `migration/processors.py` | `get_input_processor()`, `get_output_processor()` |
| `DispatcherRegistry` | `migration/dispatchers.py` | `get()`, `register()` |

### Visualization 模块

| 类 | 文件 | 主要方法 |
|---|------|---------|
| `GraphvizDataflowVisualizer` | `visualization/graphviz_visualizer.py` | `generate_dataflow_svg()` |

---

## 依赖说明

### 必需依赖

- Python >= 3.8
- numpy

### 可选依赖

| 功能 | 依赖包 |
|------|--------|
| GPU支持 | torch, torchvision |
| 图像处理迁移 | kornia, opencv-python, Pillow |
| 音频处理迁移 | torchaudio, librosa |
| 可视化 | graphviz |

---

## 配置说明

**文件**: `config.py`

```python
from config import (
    PACKAGE_NAME,
    DEFAULT_MAX_DEPTH,
    DEFAULT_TRACK_MEMORY,
    LOG_LEVEL,
    LOG_FORMAT
)
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEFAULT_MAX_DEPTH` | 10 | 默认追踪深度 |
| `DEFAULT_TRACK_MEMORY` | True | 是否追踪内存 |
| `LOG_LEVEL` | "INFO" | 日志级别 |

---

## 更多信息

- 示例代码位于 `examples/` 目录
- 测试用例位于 `examples/dataflow_tests/` 和 `examples/optimizer_tests/`
