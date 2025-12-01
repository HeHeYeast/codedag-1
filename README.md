# CodeDAG - 简洁版数据流图解析框架

## 🎯 项目概述

这是CodeDAG框架的简洁版本，专注于核心的数据流图解析和优化功能。去除了冗余的实验代码，保留了最核心的模块和功能。

## 📁 目录结构

```
codedag_clean/
├── core/                    # 核心追踪模块
│   ├── enhanced_tracer.py   # 增强追踪器
│   ├── base_tracer.py       # 基础追踪器
│   ├── dag_builder.py       # DAG构建器
│   ├── memory_profiler.py   # 内存分析器
│   └── performance_monitor.py # 性能监控器
│
├── migration/               # 设备迁移模块
│   ├── migration_manager.py # 迁移管理器
│   ├── device_context.py    # 设备上下文
│   └── migration_planner.py # 迁移规划器
│
├── optimizer/               # 优化模块
│   ├── dag_optimizer.py     # DAG优化器
│   ├── subgraph_partitioner.py # 子图分割器
│   └── optimization_strategy.py # 优化策略
│
├── utils/                   # 工具模块
│   ├── device_detector.py   # 设备检测器
│   └── device_profiler.py   # 设备性能分析器
│
├── examples/                # 测试示例
│   ├── corrected_test_runner.py # 主要测试运行器
│   ├── test_basic_arithmetic.py # 基础算术测试
│   ├── test_pytorch_dataset.py  # PyTorch数据集测试
│   └── test_numpy_arrays.py     # NumPy数组测试
│
├── tracer.py               # 主要追踪器 (MigrationEnabledTracer)
├── tracer_enhanced.py      # 增强追踪器 (EnhancedMigrationTracer)
├── visualization.py        # 可视化模块
├── config.py              # 配置文件
└── __init__.py            # 包初始化
```

## 🚀 核心功能

### 1. **数据流图解析**
- **EnhancedTracer**: 使用`start_tracing()`/`stop_tracing()`进行深度追踪
- **EnhancedMigrationTracer**: 使用`tracing_context()`进行上下文追踪
- **MigrationEnabledTracer**: 三阶段工作流程，支持迁移优化

### 2. **自动导出功能**
- **JSON导出**: `export_dataflow_graph(output_path)`
- **完整元数据**: 包含节点、边、设备信息、性能数据
- **设备检测**: 自动检测可用的CPU/GPU设备

### 3. **通用追踪能力**
- ✅ **基础算术**: a+b, 函数调用链
- ✅ **复杂计算**: NumPy大矩阵运算  
- ✅ **深度学习框架**: PyTorch Dataset/DataLoader
- ✅ **内存监控**: 内存分配和使用追踪
- ✅ **性能分析**: 执行时间和设备利用率

## 🧪 快速测试

### 运行所有测试
```bash
cd codedag_clean
python examples/corrected_test_runner.py
```

### 测试结果示例
```
CodeDAG 修正版测试运行器
==================================================
✓ EnhancedMigrationTracer 测试通过
✓ EnhancedTracer 测试通过  
✓ PyTorch Dataset 测试通过

测试完成: 3/3 通过
🎉 所有测试通过！CodeDAG核心功能正常
```

### 单独测试类别
```bash
# 基础算术测试
python examples/test_basic_arithmetic.py

# PyTorch数据集测试  
python examples/test_pytorch_dataset.py

# NumPy数组测试
python examples/test_numpy_arrays.py
```

## 📊 使用方式

### 1. **简单函数追踪**
```python
from tracer_enhanced import EnhancedMigrationTracer

tracer = EnhancedMigrationTracer(max_depth=8)

def my_function():
    return x + y

# 追踪执行
with tracer.tracing_context():
    result = my_function()

# 导出结果
tracer.export_dataflow_graph("my_results.json")
```

### 2. **PyTorch Dataset追踪**
```python
from tracer_enhanced import EnhancedMigrationTracer
from torch.utils.data import DataLoader

tracer = EnhancedMigrationTracer(max_depth=6)

# 你的Dataset类
class MyDataset(Dataset):
    def __getitem__(self, idx):
        # 这些函数会被自动追踪
        data = self.load_data(idx)
        processed = self.process_data(data)
        return self.to_tensor(processed)

# 追踪数据加载过程
with tracer.tracing_context():
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=4)
    
    for batch in dataloader:
        break  # 只处理一个batch
        
# 导出数据流图
tracer.export_dataflow_graph("dataset_dataflow.json")
```

### 3. **深度追踪模式**
```python
from core.enhanced_tracer import EnhancedTracer

tracer = EnhancedTracer(max_depth=10, track_memory=True)

tracer.start_tracing()
# 执行你的代码
tracer.stop_tracing()

# 查看结果
print(f"追踪节点数: {len(tracer.enhanced_nodes)}")
```

## 📋 导出的数据格式

### JSON导出示例
```json
{
  "timestamp": "2025-11-06T21:10:33.204263",
  "metadata": {
    "total_nodes": 9,
    "total_edges": 6,
    "gpu_operations": 0,
    "traced_operations": ["compute_function_0", "complex_function_1"],
    "target_device": "cpu",
    "trace_depth": 8,
    "available_devices": [
      {
        "device_id": "cpu",
        "device_type": "cpu", 
        "physical_cores": 18,
        "memory_gb": 125.48,
        "compute_power": 250.2
      }
    ]
  },
  "nodes": [...],
  "edges": [...],
  "performance_data": {...}
}
```

## ✅ 验证的功能

### 测试覆盖
- [x] **EnhancedMigrationTracer**: 9个节点，6条边
- [x] **EnhancedTracer**: 18个节点，完整函数追踪
- [x] **PyTorch Dataset**: 24个节点，16条边，包含Dataset操作
- [x] **自动导出**: JSON格式，完整元数据
- [x] **设备检测**: CPU/GPU自动识别

### 核心能力验证
1. ✅ **通用数据流解析** - 不限于Dataset，支持任意Python函数
2. ✅ **完整的追踪深度** - 解决了原来的"2节点问题"
3. ✅ **自动导出功能** - 内置JSON导出，无需手动实现
4. ✅ **性能数据收集** - 执行时间、内存使用、设备信息
5. ✅ **框架兼容性** - 支持PyTorch、NumPy等主流框架

## 🔧 配置选项

### 追踪器参数
- `max_depth`: 最大追踪深度 (建议6-10)
- `track_memory`: 是否追踪内存使用
- `track_gpu`: 是否启用GPU监控

### 导出选项
- `export_dataflow_graph(path)`: 导出完整数据流图
- `export_results(path)`: 导出结果摘要
- 自动创建输出目录
- JSON格式，包含完整元数据

## 🎯 主要改进

相比原始复杂版本:

1. **简化了架构** - 移除了冗余的examples和experiments
2. **统一了API** - 明确了不同tracer的使用方式  
3. **完善了导出** - 直接使用内置的导出功能
4. **验证了通用性** - 证明了对各种计算场景的支持
5. **提供了清晰的使用示例** - 快速上手指南

这个简洁版本专注于核心功能，提供了稳定、易用的数据流图解析能力。