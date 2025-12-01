"""
迁移模块对外接口 (Migration API)

提供简洁的 API 用于：
1. 离线模式：从 JSON 文件加载迁移计划
2. 在线模式：直接从优化器获取迁移计划
3. 手动模式：自定义迁移配置
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Protocol, runtime_checkable

from .core import (
    OpSpec,
    StrategyType,
    SparseContextTracker,
    PatchInjector,
    normalize_context_key,
)
from .processors import ProcessorRegistry
from .registry import create_default_registry, DefaultStrategyRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# 协议定义
# =============================================================================

@runtime_checkable
class OptimizerProtocol(Protocol):
    """
    优化器接口协议

    任何实现了 get_migration_plan 方法的对象都可以作为 optimizer 使用
    """

    def get_migration_plan(
        self,
        partitions: Any = None,
        dag: Any = None,
        gpu_count: int = 1,
    ) -> Dict[str, Any]:
        """
        获取迁移计划

        Args:
            partitions: 分区结果
            dag: DAG 结构
            gpu_count: GPU 数量

        Returns:
            迁移计划字典，必须包含 'context_device_map' 键
        """
        ...


# =============================================================================
# 主要类
# =============================================================================

class PipelineMigrator:
    """
    数据管道迁移器

    主要入口类，负责协调整个迁移流程：
    1. 加载迁移计划（context -> device 映射）
    2. 初始化策略注册表和处理器
    3. 注入 Patch
    4. 提供运行时上下文追踪

    使用方式：

    ```python
    # 离线模式 - 从 JSON 加载
    migrator = PipelineMigrator.from_json("migration_plan.json")
    migrator.apply()

    # 在线模式 - 从优化器获取
    from optimizer import OptimizerManager
    optimizer = OptimizerManager()
    plan = optimizer.get_migration_plan(partitions, dag)
    migrator = PipelineMigrator.from_plan(plan)
    migrator.apply()

    # 使用 context manager
    with PipelineMigrator.from_json("plan.json") as migrator:
        result = original_pipeline()
    # 自动恢复

    # 恢复原始函数
    migrator.restore()
    ```
    """

    def __init__(
        self,
        context_device_map: Dict[str, str],
        registry: Optional[DefaultStrategyRegistry] = None,
        processors: Optional[ProcessorRegistry] = None,
        default_device: str = "cuda:0",
        gpu_count: int = 1,
    ):
        """
        初始化迁移器

        Args:
            context_device_map: 上下文路径到设备的映射
            registry: 策略注册表（默认使用内置注册表）
            processors: 处理器注册表（默认使用内置处理器）
            default_device: 默认目标设备
            gpu_count: GPU 数量（用于多卡场景）
        """
        self.context_device_map = context_device_map
        self.registry = registry or create_default_registry()
        self.processors = processors or ProcessorRegistry
        self.default_device = default_device
        self.gpu_count = gpu_count

        # 运行时组件
        self.context_tracker = SparseContextTracker()
        self._injector: Optional[PatchInjector] = None
        self._is_applied = False

    @classmethod
    def from_json(
        cls,
        json_path: Union[str, Path],
        default_device: str = "cuda:0",
    ) -> "PipelineMigrator":
        """
        从 JSON 文件加载迁移计划

        JSON 格式：
        ```json
        {
            "context_device_map": {
                "main->process->cv2.resize": "cuda:0",
                ...
            },
            "gpu_count": 1
        }
        ```

        Args:
            json_path: JSON 文件路径
            default_device: 默认设备

        Returns:
            PipelineMigrator 实例
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Migration plan not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        context_device_map = data.get("context_device_map", {})
        gpu_count = data.get("gpu_count", 1)

        logger.info(f"从 {json_path} 加载迁移计划，共 {len(context_device_map)} 个映射")

        return cls(
            context_device_map=context_device_map,
            default_device=default_device,
            gpu_count=gpu_count,
        )

    @classmethod
    def from_plan(
        cls,
        plan: Dict[str, Any],
        default_device: str = "cuda:0",
    ) -> "PipelineMigrator":
        """
        从优化器返回的迁移计划创建

        Args:
            plan: 优化器返回的迁移计划 dict
            default_device: 默认设备

        Returns:
            PipelineMigrator 实例
        """
        context_device_map = plan.get("context_device_map", {})
        gpu_count = plan.get("gpu_count", 1)

        logger.info(f"从内存加载迁移计划，共 {len(context_device_map)} 个映射")

        return cls(
            context_device_map=context_device_map,
            default_device=default_device,
            gpu_count=gpu_count,
        )

    @classmethod
    def from_optimizer(
        cls,
        optimizer: OptimizerProtocol,
        partitions: Any = None,
        dag: Any = None,
        gpu_count: int = 1,
        default_device: str = "cuda:0",
    ) -> "PipelineMigrator":
        """
        直接从优化器获取迁移计划

        Args:
            optimizer: 实现 OptimizerProtocol 的对象
                       必须提供 get_migration_plan(partitions, dag, gpu_count) 方法
            partitions: 分区结果（可选）
            dag: DAG 结构（可选）
            gpu_count: GPU 数量
            default_device: 默认设备

        Returns:
            PipelineMigrator 实例

        Note:
            optimizer 必须实现 get_migration_plan 方法，返回包含
            'context_device_map' 键的字典。参见 OptimizerProtocol。
        """
        plan = optimizer.get_migration_plan(
            partitions=partitions,
            dag=dag,
            gpu_count=gpu_count,
        )
        return cls.from_plan(plan, default_device=default_device)

    def apply(self) -> Dict[str, int]:
        """
        应用迁移 Patch

        Returns:
            统计信息 {'patched': n, 'failed': m, 'skipped': k}
        """
        if self._is_applied:
            logger.warning("迁移已应用，跳过重复应用")
            return {'patched': 0, 'failed': 0, 'skipped': 0}

        self._injector = PatchInjector(
            context_device_map=self.context_device_map,
            registry=self.registry,
            processors=self.processors,
            context_tracker=self.context_tracker,
            default_device=self.default_device,
        )

        stats = self._injector.apply_all()
        self._is_applied = True

        logger.info(f"迁移 Patch 应用完成: {stats}")
        return stats

    def restore(self) -> int:
        """
        恢复所有原始函数

        Returns:
            恢复的函数数量
        """
        if not self._is_applied:
            logger.warning("迁移未应用，无需恢复")
            return 0

        if self._injector is None:
            return 0

        count = self._injector.restore_all()
        self._is_applied = False
        self.context_tracker.reset()

        logger.info(f"已恢复 {count} 个原始函数")
        return count

    def is_applied(self) -> bool:
        """检查迁移是否已应用"""
        return self._is_applied

    def get_stats(self) -> Dict:
        """获取当前状态统计"""
        return {
            'is_applied': self._is_applied,
            'context_mappings': len(self.context_device_map),
            'registry_stats': self.registry.stats(),
            'gpu_count': self.gpu_count,
            'default_device': self.default_device,
        }

    def __enter__(self):
        """支持 context manager 用法"""
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动恢复"""
        self.restore()
        return False


# =============================================================================
# 便捷函数
# =============================================================================

def migrate_from_json(
    json_path: Union[str, Path],
    default_device: str = "cuda:0",
) -> PipelineMigrator:
    """
    便捷函数：从 JSON 加载并应用迁移

    Args:
        json_path: JSON 文件路径
        default_device: 默认设备

    Returns:
        已应用的 PipelineMigrator 实例
    """
    migrator = PipelineMigrator.from_json(json_path, default_device)
    migrator.apply()
    return migrator


def create_custom_migrator(
    function_device_map: Dict[str, str],
    additional_specs: Optional[List[OpSpec]] = None,
    default_device: str = "cuda:0",
) -> PipelineMigrator:
    """
    创建自定义迁移器

    适用于不使用优化器，手动指定迁移的场景。
    支持简单的函数名映射，利用 Wrapper 的模糊匹配能力。

    Args:
        function_device_map: 函数路径到设备的映射
            支持完整路径: {"main->process->cv2.resize": "cuda:0"}
            也支持简单函数名: {"cv2.resize": "cuda:0"}
        additional_specs: 额外的 OpSpec 定义
        default_device: 默认设备

    Returns:
        PipelineMigrator 实例（未应用）

    Example:
        ```python
        migrator = create_custom_migrator({
            "cv2.resize": "cuda:0",
            "cv2.cvtColor": "cuda:0",
        })
        migrator.apply()
        ```
    """
    registry = create_default_registry()

    # 注册额外的策略
    if additional_specs:
        for spec in additional_specs:
            registry.register(spec)

    return PipelineMigrator(
        context_device_map=function_device_map,
        registry=registry,
        default_device=default_device,
    )


def list_supported_functions() -> Dict[str, List[str]]:
    """
    列出所有支持迁移的函数

    Returns:
        按策略类型分组的函数列表
    """
    registry = create_default_registry()

    # 显式映射 StrategyType 到友好的 key 名
    key_map = {
        StrategyType.MOVE_ONLY: 'move_only',
        StrategyType.STANDARD_OP: 'standard_op',
        StrategyType.FACTORY_OP: 'factory_op',
        StrategyType.PIPELINE_OP: 'pipeline_op',
        StrategyType.NON_MIGRATABLE: 'non_migratable',
    }

    result = {k: [] for k in key_map.values()}

    for strategy_type, key_name in key_map.items():
        specs = registry.get_by_strategy(strategy_type)
        # 排序以保持输出稳定
        result[key_name] = sorted([spec.source for spec in specs])

    return result
