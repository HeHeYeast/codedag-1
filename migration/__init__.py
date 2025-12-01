"""
迁移模块 (Migration Module)

数据管道 CPU -> GPU 迁移的核心模块

主要组件：
- PipelineMigrator: 主入口类，用于应用和管理迁移
- OpSpec: 操作规格定义
- StrategyType: 策略类型枚举
- ProcessorRegistry: 处理器注册表
- StrategyRegistry: 策略注册表

使用示例：

    # 离线模式 - 从 JSON 加载
    from migration import PipelineMigrator
    migrator = PipelineMigrator.from_json("migration_plan.json")
    migrator.apply()

    # 在线模式 - 从优化器获取
    from migration import PipelineMigrator
    plan = optimizer.get_migration_plan(partitions, dag)
    migrator = PipelineMigrator.from_plan(plan)
    migrator.apply()

    # 手动模式 - 自定义配置
    from migration import create_custom_migrator
    migrator = create_custom_migrator({
        "cv2.resize": "cuda:0",
        "cv2.cvtColor": "cuda:0",
    })
    migrator.apply()

    # 使用 context manager
    with PipelineMigrator.from_json("plan.json") as migrator:
        result = pipeline()
    # 自动恢复
"""

from .core import (
    # 核心类型
    OpSpec,
    StrategyType,
    # 上下文追踪
    SparseContextTracker,
    # 工具函数
    normalize_context_key,
    make_hashable,
    resolve_object_path,
    # 核心组件
    UniversalWrapper,
    PatchInjector,
)

from .processors import (
    ProcessorRegistry,
    InputProcessors,
    ValueMappers,
    OutputProcessors,
    StructuralOps,
)

from .dispatchers import (
    DispatcherRegistry,
    CV2Dispatchers,
    PILDispatchers,
    NumpyRandomDispatchers,
)

from .registry import (
    StrategyRegistry,
    DefaultStrategyRegistry,
    create_default_registry,
)

from .api import (
    PipelineMigrator,
    OptimizerProtocol,
    migrate_from_json,
    create_custom_migrator,
    list_supported_functions,
)

__all__ = [
    # 主入口
    'PipelineMigrator',
    'migrate_from_json',
    'create_custom_migrator',
    'list_supported_functions',

    # 核心类型
    'OpSpec',
    'StrategyType',
    'OptimizerProtocol',

    # 注册表
    'StrategyRegistry',
    'DefaultStrategyRegistry',
    'ProcessorRegistry',
    'DispatcherRegistry',
    'create_default_registry',

    # 处理器
    'InputProcessors',
    'ValueMappers',
    'OutputProcessors',
    'StructuralOps',

    # 分发器
    'CV2Dispatchers',
    'PILDispatchers',
    'NumpyRandomDispatchers',

    # 上下文追踪
    'SparseContextTracker',

    # 核心组件
    'UniversalWrapper',
    'PatchInjector',

    # 工具函数
    'normalize_context_key',
    'make_hashable',
    'resolve_object_path',
]
