# CodeDAG API 参考手册

本文档详细描述每个模块的接口定义和使用方法。

---

## 1. Core 模块接口

### 1.1 EnhancedTracer

**位置**: `core/enhanced_tracer.py:24`

增强型代码追踪器，支持数据流图构建、内存追踪和性能监控。

#### 构造函数

```python
def __init__(
    self,
    max_depth: int = 10,
    track_memory: bool = True,
    filter_config: Optional[FilterConfig] = None
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_depth` | `int` | 10 | 函数调用追踪的最大深度 |
| `track_memory` | `bool` | True | 是否启用内存追踪 |
| `filter_config` | `FilterConfig` | None | 函数过滤配置 |

#### 方法

##### `tracing_context()`

```python
@contextmanager
def tracing_context(self) -> Generator[None, None, None]
```

返回追踪上下文管理器，在上下文内的代码执行会被追踪。

**示例**:
```python
tracer = EnhancedTracer()
with tracer.tracing_context():
    result = my_function()
```

##### `export_dataflow_graph()`

```python
def export_dataflow_graph(self) -> Dict[str, Any]
```

导出数据流图为字典格式。

**返回值**:
```python
{
    "nodes": [
        {
            "id": int,
            "name": str,
            "type": str,
            "performance": {...},
            "attributes": {...}
        }
    ],
    "edges": [
        {
            "source": int,
            "target": int,
            "type": str
        }
    ],
    "metadata": {...}
}
```

##### `export_visualization(path)`

```python
def export_visualization(self, output_path: str) -> bool
```

导出数据流图的 SVG 可视化。

| 参数 | 类型 | 说明 |
|------|------|------|
| `output_path` | `str` | SVG 文件输出路径 |

**返回值**: `bool` - 是否成功生成

---

### 1.2 DAGBuilder

**位置**: `core/dag_builder.py:15`

DAG 构建器，负责构建和维护执行图结构。

#### 构造函数

```python
def __init__(self)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `dag` | `ExecutionDAG` | 构建的 DAG 对象 |
| `node_counter` | `int` | 节点计数器 |

#### 方法

##### `add_function_node()`

```python
def add_function_node(
    self,
    name: str,
    context: str,
    performance: Dict = None
) -> int
```

添加函数调用节点。

**返回值**: `int` - 新节点的 ID

##### `add_variable_node()`

```python
def add_variable_node(
    self,
    name: str,
    version: int = 0,
    snapshot: Any = None
) -> int
```

添加变量节点。

##### `add_edge()`

```python
def add_edge(
    self,
    source_id: int,
    target_id: int,
    edge_type: str
) -> None
```

添加边。

| 参数 | 类型 | 说明 |
|------|------|------|
| `source_id` | `int` | 源节点 ID |
| `target_id` | `int` | 目标节点 ID |
| `edge_type` | `str` | 边类型: `creates`, `uses`, `produces`, `calls` |

---

### 1.3 DAGNode

**位置**: `core/dag_builder.py:8`

DAG 节点数据类。

```python
@dataclass
class DAGNode:
    node_id: int
    name: str
    node_type: str  # "function_call" | "variable" | "operator"
    performance: Dict = field(default_factory=dict)
    attributes: Dict = field(default_factory=dict)
```

#### 属性详解

| 属性 | 类型 | 说明 |
|------|------|------|
| `node_id` | `int` | 唯一标识符 |
| `name` | `str` | 节点名称 (函数名/变量名) |
| `node_type` | `str` | 节点类型 |
| `performance` | `Dict` | 性能数据 |
| `attributes` | `Dict` | 扩展属性 |

**performance 结构**:
```python
{
    "execution_time": float,      # 执行时间 (秒)
    "execution_time_ms": float,   # 执行时间 (毫秒)
    "memory_usage": int,          # 内存使用 (字节)
    "peak_memory_mb": float,      # 峰值内存 (MB)
    "call_count": int             # 调用次数 (聚类后)
}
```

**attributes 结构**:
```python
{
    "cluster_info": {             # 聚类信息 (图粗化后)
        "is_clustered": bool,
        "instance_count": int,
        "original_ids": List[int],
        "representative_id": int
    },
    "variable_snapshot": {...}    # 变量快照
}
```

---

### 1.4 ExecutionDAG

**位置**: `core/dag_builder.py:1`

执行图数据结构。

```python
class ExecutionDAG:
    nodes: Dict[int, DAGNode]
    edges: List[Tuple[int, int, str]]
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `nodes` | `Dict[int, DAGNode]` | 节点字典，key 为节点 ID |
| `edges` | `List[Tuple]` | 边列表，每条边为 `(source_id, target_id, edge_type)` |

---

### 1.5 MemoryProfiler

**位置**: `core/memory_profiler.py:10`

内存分析器。

#### 方法

```python
def start(self) -> None
def stop(self) -> None
def take_snapshot(self) -> MemorySnapshot
def get_current_usage(self) -> int  # 字节
def get_peak_usage(self) -> int     # 字节
```

---

### 1.6 PerformanceMonitor

**位置**: `core/performance_monitor.py:8`

性能监控器。

#### 方法

```python
@contextmanager
def measure(self) -> Generator[None, None, None]

def get_stats(self) -> Dict[str, Any]
```

**返回的统计信息**:
```python
{
    "execution_time": float,
    "memory_delta": int,
    "cpu_time": float
}
```

---

### 1.7 FunctionFilter

**位置**: `core/function_filter.py:5`

函数过滤器，控制哪些函数被追踪。

```python
class FunctionFilter:
    def __init__(self, config: FilterConfig = None)
    def should_trace(self, func_name: str, module: str) -> bool
```

**FilterConfig 结构**:
```python
{
    "include_patterns": List[str],  # 包含的模式
    "exclude_patterns": List[str],  # 排除的模式
    "include_modules": List[str],   # 包含的模块
    "exclude_modules": List[str]    # 排除的模块
}
```

---

## 2. Optimizer 模块接口

### 2.1 OptimizerManager

**位置**: `optimizer/optimizer_manager.py:20`

优化管理器，协调整个优化流程。

#### 构造函数

```python
def __init__(
    self,
    k: int = 4,
    coarsen_max_depth: int = 5,
    anchor_patterns: List[str] = None,
    lambda_weight: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    iter_alpha: float = 0.4,
    iter_beta: float = 0.6,
    max_iterations: int = 200,
    no_improvement_threshold: int = 20
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `k` | `int` | 4 | 分区数量 |
| `coarsen_max_depth` | `int` | 5 | 图粗化最大深度 |
| `anchor_patterns` | `List[str]` | None | 锚点模式 |
| `lambda_weight` | `float` | 0.5 | 瓶颈选择距离权重 |
| `alpha` | `float` | 0.5 | K-way 性能权重 |
| `beta` | `float` | 0.3 | K-way 距离惩罚 |
| `gamma` | `float` | 0.2 | K-way 负载惩罚 |
| `iter_alpha` | `float` | 0.4 | 迭代优化负载权重 |
| `iter_beta` | `float` | 0.6 | 迭代优化关键路径权重 |
| `max_iterations` | `int` | 200 | 最大迭代次数 |

#### 方法

##### `optimize()`

```python
def optimize(
    self,
    dag: ExecutionDAG,
    export_dir: Optional[str] = None
) -> Dict[str, Any]
```

执行完整优化流程。

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `dag` | `ExecutionDAG` | 输入的 DAG |
| `export_dir` | `str` | 导出目录 (可选) |

**返回值**:
```python
{
    "coarsened_dag": ExecutionDAG,          # 粗化后的 DAG
    "kway_partitions": List[Partition],     # K-way 分割结果
    "optimized_partitions": List[Partition],# 迭代优化后的分区列表
    "statistics": {                          # 各阶段统计信息
        "original_nodes": int,
        "original_edges": int,
        "coarsened_nodes": int,
        "coarsened_edges": int,
        "coarsening": Dict,                  # 粗化统计
        "partitioning": Dict,                # 划分统计
        "edge_cut": Dict,                    # 边切割统计
        "iteration": Dict                    # 迭代优化统计
    }
}
```

##### `get_migration_plan()`

```python
def get_migration_plan(
    self,
    partitions: List[Partition],
    dag: ExecutionDAG,
    gpu_count: int = 1
) -> Dict[str, Any]
```

生成迁移计划。

**返回值**:
```python
{
    "context_device_map": {
        "context_path": "cuda:0"
    },
    "gpu_count": int,
    "partition_device_map": {
        partition_id: "cuda:X"
    }
}
```

---

### 2.2 GraphCoarsening

**位置**: `optimizer/graph_coarsening.py:39`

图粗化聚类器。

#### 构造函数

```python
def __init__(
    self,
    max_depth: int = 5,
    anchor_patterns: List[str] = None
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_depth` | `int` | 5 | 子图提取最大深度 |
| `anchor_patterns` | `List[str]` | None | 自定义锚点模式 |

#### 方法

##### `coarsen()`

```python
def coarsen(self, dag: ExecutionDAG) -> ExecutionDAG
```

执行图粗化。

**返回值**: 粗化后的新 DAG

##### `get_stats()`

```python
def get_stats(self) -> Dict[str, Any]
```

**返回值**:
```python
{
    "pattern_count": int,          # 模式数量
    "total_instances": int,        # 总实例数
    "nodes_saved": int,            # 节省的节点数
    "pattern_details": [
        {
            "hash": str,
            "instance_count": int,
            "nodes_per_instance": int,
            "nodes_saved": int
        }
    ]
}
```

---

### 2.3 KWayPartitioner

**位置**: `optimizer/kway_partitioner.py:31`

K-way 图划分器。

#### 构造函数

```python
def __init__(
    self,
    k: int = 4,
    lambda_weight: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `k` | `int` | 4 | 分区数量 |
| `lambda_weight` | `float` | 0.5 | 瓶颈选择距离权重 |
| `alpha` | `float` | 0.5 | 性能权重 |
| `beta` | `float` | 0.3 | 距离惩罚权重 |
| `gamma` | `float` | 0.2 | 负载惩罚权重 |

#### 方法

##### `partition()`

```python
def partition(self, dag: ExecutionDAG) -> List[Partition]
```

执行 K-way 划分。

##### `get_stats()`

```python
def get_stats(self, partitions: List[Partition]) -> Dict
```

**返回值**:
```python
{
    "partition_count": int,
    "partition_sizes": List[int],
    "partition_loads": List[float],
    "avg_load": float,
    "max_load": float,
    "min_load": float,
    "load_balance": float,  # [0,1] 越接近1越均衡
    "total_nodes": int
}
```

##### `compute_edge_cut()`

```python
def compute_edge_cut(
    self,
    partitions: List[Partition],
    dag: ExecutionDAG
) -> Dict
```

**返回值**:
```python
{
    "total_edges": int,
    "cut_edges": int,
    "internal_edges": int,
    "cut_ratio": float
}
```

---

### 2.4 Partition

**位置**: `optimizer/kway_partitioner.py:20`

分区数据结构。

```python
class Partition:
    id: int                  # 分区 ID
    center: int              # 中心节点 ID
    nodes: Set[int]          # 分区内节点集合
    total_compute: float     # 总计算量
    total_memory: float      # 总内存量
```

---

### 2.5 IterativeOptimizer

**位置**: `optimizer/iterative_optimizer.py:23`

迭代优化器。

#### 构造函数

```python
def __init__(
    self,
    alpha: float = 0.4,
    beta: float = 0.6,
    max_iterations: int = 200,
    no_improvement_threshold: int = 20,
    recompute_cp_interval: int = 50,
    communication_overhead: float = 0.0001
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alpha` | `float` | 0.4 | 负载均衡权重 |
| `beta` | `float` | 0.6 | 关键路径权重 |
| `max_iterations` | `int` | 200 | 最大迭代次数 |
| `no_improvement_threshold` | `int` | 20 | 无改善终止阈值 |
| `recompute_cp_interval` | `int` | 50 | 关键路径重算间隔 |

#### 方法

##### `optimize()`

```python
def optimize(
    self,
    partitions: List[Partition],
    dag: ExecutionDAG
) -> List[Partition]
```

执行迭代优化。

##### `get_stats()`

```python
def get_stats(self) -> Dict
```

**返回值**:
```python
{
    "iterations": int,
    "moves_executed": int,
    "initial_objective": float,
    "final_objective": float,
    "improvement": float,
    "improvement_ratio": float  # 百分比
}
```

---

## 3. Migration 模块接口

### 3.1 PipelineMigrator

**位置**: `migration/api.py:64`

管道迁移器主类。

#### 构造函数

```python
def __init__(
    self,
    context_device_map: Dict[str, str],
    registry: Optional[DefaultStrategyRegistry] = None,
    processors: Optional[ProcessorRegistry] = None,
    default_device: str = "cuda:0",
    gpu_count: int = 1
)
```

#### 类方法

##### `from_json()`

```python
@classmethod
def from_json(
    cls,
    json_path: Union[str, Path],
    default_device: str = "cuda:0"
) -> "PipelineMigrator"
```

从 JSON 文件加载。

**JSON 格式**:
```json
{
    "context_device_map": {
        "main->process->cv2.resize": "cuda:0"
    },
    "gpu_count": 1
}
```

##### `from_plan()`

```python
@classmethod
def from_plan(
    cls,
    plan: Dict[str, Any],
    default_device: str = "cuda:0"
) -> "PipelineMigrator"
```

从字典加载。

##### `from_optimizer()`

```python
@classmethod
def from_optimizer(
    cls,
    optimizer: OptimizerProtocol,
    partitions: Any = None,
    dag: Any = None,
    gpu_count: int = 1,
    default_device: str = "cuda:0"
) -> "PipelineMigrator"
```

从优化器获取迁移计划。

#### 实例方法

##### `apply()`

```python
def apply(self) -> Dict[str, int]
```

应用迁移 Patch。

**返回值**:
```python
{
    "patched": int,   # 成功 Patch 的数量
    "failed": int,    # 失败的数量
    "skipped": int    # 跳过的数量
}
```

##### `restore()`

```python
def restore(self) -> int
```

恢复所有原始函数。

**返回值**: 恢复的函数数量

##### `is_applied()`

```python
def is_applied(self) -> bool
```

检查迁移是否已应用。

##### `get_stats()`

```python
def get_stats(self) -> Dict
```

**返回值**:
```python
{
    "is_applied": bool,
    "context_mappings": int,
    "registry_stats": Dict,
    "gpu_count": int,
    "default_device": str
}
```

#### 上下文管理器

```python
with PipelineMigrator.from_json("plan.json") as migrator:
    result = pipeline()
# 自动调用 restore()
```

---

### 3.2 OpSpec

**位置**: `migration/core.py:34`

操作规格数据类。

```python
@dataclass
class OpSpec:
    # 身份识别
    source: str                              # 源函数路径
    strategy: StrategyType = STANDARD_OP     # 迁移策略
    priority: int = 1                        # 优先级 0-3

    # 目标定位
    target_lib: str = ""                     # 目标库
    target_func: str = ""                    # 目标函数

    # 输入处理
    args_trans: Dict[int, str] = {}          # 位置参数转换
    kwargs_trans: Dict[str, str] = {}        # 关键字参数转换

    # 签名适配
    arg_renames: Dict[str, str] = {}         # 参数改名
    arg_value_maps: Dict[str, Union[str, Callable]] = {}  # 值映射
    injected_kwargs: Dict[str, Any] = {}     # 注入参数
    structural_ops: List[str] = []           # 结构性操作

    # 输出处理
    output_rule: str = "keep_on_device"

    # 运行时控制
    fallback_condition: Optional[Callable] = None
    notes: str = ""
```

---

### 3.3 StrategyType

**位置**: `migration/core.py:21`

迁移策略枚举。

```python
class StrategyType(Enum):
    MOVE_ONLY = "MoveOnly"           # 仅搬运 Tensor
    STANDARD_OP = "StandardOp"       # 函数替换
    FACTORY_OP = "FactoryOp"         # 对象工厂
    PIPELINE_OP = "PipelineOp"       # 管道容器
    NON_MIGRATABLE = "NonMigratable" # 不可迁移
```

---

### 3.4 ProcessorRegistry

**位置**: `migration/processors.py:644`

处理器注册表。

#### 类方法

```python
@classmethod
def get_input_processor(cls, name: str) -> Optional[Callable]

@classmethod
def get_value_mapper(cls, name: str) -> Optional[Callable]

@classmethod
def get_output_processor(cls, name: str) -> Optional[Callable]

@classmethod
def get_structural_op(cls, name: str) -> Optional[Callable]
```

#### 可用处理器

**输入处理器** (签名: `(x, device) -> x`):

| 名称 | 说明 |
|------|------|
| `ensure_tensor` | 通用 Tensor 转换 |
| `to_tensor_float` | 转为 float32 Tensor |
| `to_tensor_long` | 转为 int64 Tensor |
| `cv2_hwc_to_bchw` | OpenCV HWC → BCHW |
| `cv2_hwc_to_chw` | OpenCV HWC → CHW |
| `pil_to_tensor` | PIL → Tensor |
| `audio_to_tensor` | 音频 → Tensor |
| `pass` | 直接透传 |
| `to_device` | 仅搬运设备 |

**值映射器** (签名: `(value) -> value`):

| 名称 | 说明 |
|------|------|
| `swap_hw` | 交换 (W,H) → (H,W) |
| `ensure_tuple` | 确保 tuple 类型 |
| `cv2_flip_code` | cv2 flip 代码转换 |
| `cv2_interp_mode` | cv2 插值模式转换 |
| `cv2_color_code` | cv2 颜色代码转换 |
| `pil_interp_mode` | PIL 插值模式转换 |

**输出处理器** (签名: `(x) -> x`):

| 名称 | 说明 |
|------|------|
| `keep_on_device` | 保持在设备上 |
| `to_numpy` | 转回 NumPy |
| `bchw_to_hwc` | BCHW → HWC |
| `to_pil` | 转为 PIL Image |

---

### 3.5 DispatcherRegistry

**位置**: `migration/dispatchers.py:351`

动态分发器注册表。

```python
@classmethod
def get(cls, func_path: str) -> Optional[Callable]

@classmethod
def has_dispatcher(cls, func_path: str) -> bool

@classmethod
def register(cls, func_path: str, dispatcher: Callable) -> None

@classmethod
def all_dispatchers(cls) -> List[str]
```

**内置分发器**:

| 函数路径 | 说明 |
|---------|------|
| `cv2.cvtColor` | 颜色转换分发 |
| `cv2.threshold` | 阈值处理分发 |
| `PIL.Image.Image.transpose` | 翻转/旋转分发 |
| `numpy.random.choice` | 随机选择 |
| `numpy.random.normal` | 正态分布 |
| `numpy.random.uniform` | 均匀分布 |

---

### 3.6 DefaultStrategyRegistry

**位置**: `migration/registry/base.py:29`

默认策略注册表。

```python
def register(self, spec: OpSpec) -> None
def register_batch(self, specs: List[OpSpec]) -> None
def get(self, func_path: str) -> Optional[OpSpec]
def all_sources(self) -> List[str]
def get_by_priority(self, max_priority: int = 3) -> List[OpSpec]
def get_by_strategy(self, strategy: StrategyType) -> List[OpSpec]
def stats(self) -> Dict
```

---

### 3.7 便捷函数

**位置**: `migration/api.py`

```python
# 从 JSON 加载并应用
def migrate_from_json(
    json_path: Union[str, Path],
    default_device: str = "cuda:0"
) -> PipelineMigrator

# 创建自定义迁移器
def create_custom_migrator(
    function_device_map: Dict[str, str],
    additional_specs: Optional[List[OpSpec]] = None,
    default_device: str = "cuda:0"
) -> PipelineMigrator

# 列出支持的函数
def list_supported_functions() -> Dict[str, List[str]]
```

---

## 4. Visualization 模块接口

### 4.1 GraphvizDataflowVisualizer

**位置**: `visualization/graphviz_visualizer.py:23`

Graphviz 可视化器。

#### 构造函数

```python
def __init__(self)
```

#### 方法

##### `generate_dataflow_svg()`

```python
def generate_dataflow_svg(
    self,
    nodes: List[DAGNode],
    edges: List[Tuple],
    output_path: str = "dataflow.svg",
    title: str = "数据流图可视化"
) -> bool
```

生成数据流图 SVG。

##### `generate_legend_svg()`

```python
def generate_legend_svg(
    self,
    output_path: str = "dataflow_legend.svg"
) -> bool
```

生成图例 SVG。

---

### 4.2 便捷函数

```python
def create_dataflow_visualization(
    tracer: EnhancedTracer,
    output_path: str = "dataflow.svg"
) -> bool
```

从追踪器直接创建可视化。

---

## 5. Utils 模块接口

### 5.1 UnifiedDeviceDetector

**位置**: `utils/unified_device_detector.py`

```python
class UnifiedDeviceDetector:
    def detect_all(self) -> List[DeviceInfo]
    def get_best_device(self) -> DeviceInfo
    def get_cuda_devices(self) -> List[DeviceInfo]
    def is_cuda_available(self) -> bool
```

### 5.2 DeviceProfiler

**位置**: `utils/device_profiler.py`

```python
class DeviceProfiler:
    def run_benchmark(self, device: str) -> Dict
    def get_compute_capability(self) -> Tuple[int, int]
    def get_memory_info(self) -> Dict
```

---

## 6. 配置接口

**位置**: `config.py`

```python
PACKAGE_NAME: str = "codedag"
DEFAULT_MAX_DEPTH: int = 10
DEFAULT_TRACK_MEMORY: bool = True
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```
