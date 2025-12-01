"""
PyTorch 原生函数策略 (MoveOnly)

这些函数原生支持 GPU Tensor，只需搬运输入即可

注意：
- 前缀匹配用于大部分 torch.*/torchvision.*/torchaudio.* 函数
- 工厂函数需要单独处理，注入 device 参数而不是转换输入
"""

from ..core import OpSpec, StrategyType
from .base import DefaultStrategyRegistry


def register_pytorch_specs(registry: DefaultStrategyRegistry) -> None:
    """注册 PyTorch/TorchVision/Torchaudio 原生函数策略"""

    # =====================================================
    # 1. 工厂函数 - 精确匹配，注入 device 而不是转换输入
    # =====================================================
    # 这些函数的第一个参数通常是 shape/value，不应该被转为 Tensor

    factory_funcs = [
        "torch.tensor",
        "torch.as_tensor",
        "torch.from_numpy",
        "torch.zeros",
        "torch.ones",
        "torch.empty",
        "torch.full",
        "torch.randn",
        "torch.rand",
        "torch.randint",
        "torch.arange",
        "torch.linspace",
        "torch.logspace",
        "torch.eye",
        "torch.zeros_like",
        "torch.ones_like",
        "torch.empty_like",
        "torch.full_like",
        "torch.randn_like",
        "torch.rand_like",
    ]

    for func in factory_funcs:
        # *_like 函数需要转换输入 Tensor
        if func.endswith("_like"):
            registry.register(OpSpec(
                source=func,
                strategy=StrategyType.MOVE_ONLY,
                priority=0,
                args_trans={0: "ensure_tensor"},  # 第一个参数是 Tensor
                output_rule="keep_on_device",
                notes="*_like 函数，输入 Tensor 决定输出设备"
            ))
        else:
            registry.register(OpSpec(
                source=func,
                strategy=StrategyType.STANDARD_OP,
                priority=0,
                args_trans={},  # 不转换输入（shape/value）
                injected_kwargs={"device": "TARGET_DEVICE"},
                output_rule="keep_on_device",
                notes="工厂函数，注入 device 参数"
            ))

    # =====================================================
    # 2. PyTorch 原生 - 前缀匹配
    # =====================================================
    # 注意：精确匹配优先于前缀匹配，所以上面的工厂函数不会被覆盖

    # torch.nn.functional.* (大部分需要转换输入)
    registry.register(OpSpec(
        source="torch.nn.functional.*",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "ensure_tensor"},
        notes="PyTorch nn.functional 原生支持"
    ))

    # torch.fft.* (FFT 操作)
    registry.register(OpSpec(
        source="torch.fft.*",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "ensure_tensor"},
        notes="PyTorch FFT 模块"
    ))

    # torch.linalg.* (线性代数)
    registry.register(OpSpec(
        source="torch.linalg.*",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "ensure_tensor"},
        notes="PyTorch 线性代数模块"
    ))

    # =====================================================
    # 3. TorchVision - 前缀匹配
    # =====================================================

    # torchvision.transforms.v2.*
    registry.register(OpSpec(
        source="torchvision.transforms.v2.*",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "ensure_tensor"},
        notes="TorchVision v2 transforms 支持 Tensor"
    ))

    # torchvision.transforms.functional.*
    registry.register(OpSpec(
        source="torchvision.transforms.functional.*",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "ensure_tensor"},
        notes="TorchVision functional API"
    ))

    # torchvision.transforms.v2.functional.*
    registry.register(OpSpec(
        source="torchvision.transforms.v2.functional.*",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "ensure_tensor"},
        notes="TorchVision v2 functional API"
    ))

    # =====================================================
    # 4. TorchVision Compose (特殊处理)
    # =====================================================

    registry.register(OpSpec(
        source="torchvision.transforms.Compose",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "ensure_tensor"},
        notes="前提：内部 transform 都支持 Tensor"
    ))

    registry.register(OpSpec(
        source="torchvision.transforms.v2.Compose",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "ensure_tensor"},
        notes="v2 Compose 支持 Tensor"
    ))

    # =====================================================
    # 5. Torchaudio - 前缀匹配
    # =====================================================

    # torchaudio.transforms.*
    registry.register(OpSpec(
        source="torchaudio.transforms.*",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "audio_to_tensor"},
        notes="Torchaudio transforms，实例需 .to(device)"
    ))

    # torchaudio.functional.*
    registry.register(OpSpec(
        source="torchaudio.functional.*",
        strategy=StrategyType.MOVE_ONLY,
        priority=0,
        args_trans={0: "audio_to_tensor"},
        notes="Torchaudio functional API"
    ))
