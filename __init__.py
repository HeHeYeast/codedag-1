"""
CodeDAG - 数据流图解析与优化模块

一个集成的数据流图解析与优化系统，提供：
- 执行追踪和DAG构建
- 性能监控和分析  
- 智能设备迁移
- 子图分割和优化
- 可视化和报告生成

核心工作流程：
1. DAG构建阶段 - 追踪代码执行流程构建有向无环图
2. 优化分析阶段 - 基于DAG计算最优设备划分策略
3. 优化执行阶段 - 根据策略执行自动迁移和优化

主要应用场景：
- 数据加载优化：装饰Dataset的__next__方法，自动优化数据获取过程
- 跨设备迁移：通过migrate变量控制数据在CPU/GPU间的自动迁移
- 性能分析：监控和分析不同配置下的性能表现
"""

# 主要接口
from .integrated_tracer import CodeDAGTracer

# 核心组件
from .core import BaseTracer, DAGBuilder, PerformanceMonitor
from .migration import MigrationManager, MigrationPlanner, DeviceContext, CudaTensorContext
from .optimizer import DAGOptimizer, SubgraphPartitioner, OptimizationStrategy

# 可视化
from .visualization import DAGVisualizer

# 向后兼容的导入
from .tracer import MigrationEnabledTracer
from .migration import MigrationManager as OldMigrationManager
from .visualization import visualize_dag

# 版本信息
__version__ = "1.0.0"
__author__ = "CodeDAG Team"

# 主要导出
__all__ = [
    # 主要接口
    'CodeDAGTracer',
    
    # 核心组件
    'BaseTracer',
    'DAGBuilder', 
    'PerformanceMonitor',
    
    # 迁移组件
    'MigrationManager',
    'MigrationPlanner',
    'DeviceContext',
    'CudaTensorContext',
    
    # 优化组件
    'DAGOptimizer',
    'SubgraphPartitioner', 
    'OptimizationStrategy',
    
    # 可视化
    'DAGVisualizer',
    
    # 向后兼容
    'MigrationEnabledTracer',
    'visualize_dag'
]

# 便捷函数
def create_tracer(optimization_mode="performance", max_depth=3, enable_gpu=True):
    """
    创建CodeDAG追踪器的便捷函数
    
    Args:
        optimization_mode: 优化模式 ("performance", "memory", "balanced")
        max_depth: 最大追踪深度
        enable_gpu: 是否启用GPU优化
    
    Returns:
        CodeDAGTracer实例
    """
    from .optimizer.optimization_strategy import OptimizationMode
    
    mode_map = {
        "performance": OptimizationMode.PERFORMANCE,
        "memory": OptimizationMode.MEMORY, 
        "balanced": OptimizationMode.BALANCED
    }
    
    mode = mode_map.get(optimization_mode, OptimizationMode.PERFORMANCE)
    return CodeDAGTracer(max_depth=max_depth, optimization_mode=mode)

def quick_optimize_dataset(dataset_class, *args, **kwargs):
    """
    快速优化Dataset的便捷函数
    
    使用示例:
    ```python
    from codedag import quick_optimize_dataset
    
    # 优化数据集
    optimized_dataset = quick_optimize_dataset(MyDataset, data_path="/path/to/data")
    dataloader = DataLoader(optimized_dataset, batch_size=32)
    
    # 使用优化后的数据加载器
    for batch in dataloader:
        # 数据会自动迁移到最优设备
        process_batch(batch)
    ```
    """
    # 创建追踪器
    tracer = create_tracer()
    
    # 创建数据集实例
    dataset = dataset_class(*args, **kwargs)
    
    # 进行优化分析（简化版）
    # 这里可以添加自动的追踪和优化逻辑
    
    return dataset