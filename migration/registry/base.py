"""
策略注册表基类 (Strategy Registry Base)
"""

import logging
from typing import Dict, List, Optional

from ..core import OpSpec, StrategyType

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """策略注册表抽象基类"""

    def get(self, func_path: str) -> Optional[OpSpec]:
        """获取函数的迁移策略"""
        raise NotImplementedError

    def register(self, spec: OpSpec) -> None:
        """注册迁移策略"""
        raise NotImplementedError

    def all_sources(self) -> List[str]:
        """获取所有已注册的源函数路径"""
        raise NotImplementedError


class DefaultStrategyRegistry(StrategyRegistry):
    """
    默认策略注册表实现

    特性：
    - 支持精确匹配和前缀匹配
    - 优先级排序
    - 批量注册
    """

    def __init__(self):
        self._specs: Dict[str, OpSpec] = {}
        # 前缀匹配表 (用于 MoveOnly 策略)
        self._prefix_specs: Dict[str, OpSpec] = {}

    def register(self, spec: OpSpec) -> None:
        """
        注册迁移策略

        Args:
            spec: OpSpec 实例
        """
        source = spec.source

        # 检查是否是前缀模式 (以 * 结尾)
        if source.endswith('*'):
            prefix = source[:-1]  # 去掉 *
            self._prefix_specs[prefix] = spec
            logger.debug(f"注册前缀策略: {prefix}*")
        else:
            self._specs[source] = spec
            logger.debug(f"注册策略: {source}")

    def register_batch(self, specs: List[OpSpec]) -> None:
        """批量注册策略"""
        for spec in specs:
            self.register(spec)

    def get(self, func_path: str) -> Optional[OpSpec]:
        """
        获取函数的迁移策略

        查找顺序：
        1. 精确匹配
        2. 前缀匹配 (最长前缀优先)

        Args:
            func_path: 函数完整路径，如 "cv2.resize"

        Returns:
            OpSpec 或 None
        """
        # 1. 精确匹配
        if func_path in self._specs:
            return self._specs[func_path]

        # 2. 前缀匹配 (最长前缀优先)
        best_match = None
        best_prefix_len = 0

        for prefix, spec in self._prefix_specs.items():
            if func_path.startswith(prefix):
                if len(prefix) > best_prefix_len:
                    best_prefix_len = len(prefix)
                    best_match = spec

        return best_match

    def all_sources(self) -> List[str]:
        """获取所有已注册的源函数路径"""
        sources = list(self._specs.keys())
        # 前缀模式也加入 (带 * 后缀)
        sources.extend(f"{prefix}*" for prefix in self._prefix_specs.keys())
        return sources

    def get_by_priority(self, max_priority: int = 3) -> List[OpSpec]:
        """
        按优先级获取策略

        Args:
            max_priority: 最大优先级 (0=Critical, 1=High, 2=Medium, 3=Low)

        Returns:
            满足优先级要求的 OpSpec 列表
        """
        result = []
        for spec in self._specs.values():
            if spec.priority <= max_priority:
                result.append(spec)
        return sorted(result, key=lambda s: s.priority)

    def get_by_strategy(self, strategy: StrategyType) -> List[OpSpec]:
        """
        按策略类型获取

        Args:
            strategy: 策略类型

        Returns:
            该类型的所有 OpSpec
        """
        result = []
        for spec in self._specs.values():
            if spec.strategy == strategy:
                result.append(spec)
        for spec in self._prefix_specs.values():
            if spec.strategy == strategy:
                result.append(spec)
        return result

    def stats(self) -> Dict:
        """获取注册表统计信息"""
        by_strategy = {}
        by_priority = {0: 0, 1: 0, 2: 0, 3: 0}

        for spec in self._specs.values():
            strategy_name = spec.strategy.value
            by_strategy[strategy_name] = by_strategy.get(strategy_name, 0) + 1
            by_priority[spec.priority] = by_priority.get(spec.priority, 0) + 1

        return {
            'total': len(self._specs) + len(self._prefix_specs),
            'exact_match': len(self._specs),
            'prefix_match': len(self._prefix_specs),
            'by_strategy': by_strategy,
            'by_priority': by_priority,
        }
