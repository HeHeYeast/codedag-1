"""
策略注册表模块 (Strategy Registry Module)

提供 OpSpec 的注册、查找和管理功能
"""

from .base import StrategyRegistry, DefaultStrategyRegistry
from .cv_specs import register_cv_specs
from .numpy_specs import register_numpy_specs
from .audio_specs import register_audio_specs
from .pytorch_specs import register_pytorch_specs

__all__ = [
    'StrategyRegistry',
    'DefaultStrategyRegistry',
    'create_default_registry',
]


def create_default_registry() -> DefaultStrategyRegistry:
    """
    创建包含所有默认策略的注册表

    Returns:
        配置好的 DefaultStrategyRegistry 实例
    """
    registry = DefaultStrategyRegistry()

    # 注册各领域的策略
    register_pytorch_specs(registry)  # MoveOnly 策略
    register_cv_specs(registry)       # OpenCV -> Kornia
    register_numpy_specs(registry)    # NumPy -> PyTorch
    register_audio_specs(registry)    # Librosa -> Torchaudio

    return registry
