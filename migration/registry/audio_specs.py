"""
音频函数策略 (Audio Specs)

Librosa -> Torchaudio 映射
SciPy Signal -> PyTorch 映射
"""

from ..core import OpSpec, StrategyType
from .base import DefaultStrategyRegistry


def register_audio_specs(registry: DefaultStrategyRegistry) -> None:
    """注册音频相关函数策略"""

    # =====================================================
    # 1. 不可迁移的 I/O 函数
    # =====================================================

    io_funcs = [
        "librosa.load",
        "soundfile.read",
        "soundfile.write",
        "scipy.io.wavfile.read",
        "scipy.io.wavfile.write",
    ]

    for func in io_funcs:
        registry.register(OpSpec(
            source=func,
            strategy=StrategyType.NON_MIGRATABLE,
            priority=0,
            notes="音频 I/O 操作，不可迁移"
        ))

    # =====================================================
    # 2. Librosa -> Torchaudio (FactoryOp)
    # =====================================================

    # librosa.resample - P1 (最慢的音频 CPU 操作之一)
    registry.register(OpSpec(
        source="librosa.resample",
        strategy=StrategyType.FACTORY_OP,
        priority=1,
        target_lib="torchaudio.transforms",
        target_func="Resample",
        args_trans={0: "audio_to_tensor"},
        kwargs_trans={"y": "audio_to_tensor"},
        arg_renames={"orig_sr": "orig_freq", "target_sr": "new_freq"},
        output_rule="keep_on_device",
        notes="最慢的音频 CPU 操作之一"
    ))

    # librosa.feature.melspectrogram - P1
    registry.register(OpSpec(
        source="librosa.feature.melspectrogram",
        strategy=StrategyType.FACTORY_OP,
        priority=1,
        target_lib="torchaudio.transforms",
        target_func="MelSpectrogram",
        args_trans={0: "audio_to_tensor"},
        kwargs_trans={"y": "audio_to_tensor"},
        arg_renames={
            "sr": "sample_rate",
            "n_fft": "n_fft",
            "hop_length": "hop_length",
            "n_mels": "n_mels"
        },
        injected_kwargs={"pad_mode": "reflect"},  # 对齐 librosa 默认值
        output_rule="keep_on_device"
    ))

    # librosa.feature.mfcc - P2
    registry.register(OpSpec(
        source="librosa.feature.mfcc",
        strategy=StrategyType.FACTORY_OP,
        priority=2,
        target_lib="torchaudio.transforms",
        target_func="MFCC",
        args_trans={0: "audio_to_tensor"},
        kwargs_trans={"y": "audio_to_tensor"},
        arg_renames={"sr": "sample_rate", "n_mfcc": "n_mfcc"},
        output_rule="keep_on_device"
    ))

    # librosa.stft - P2 (使用 torch.stft)
    registry.register(OpSpec(
        source="librosa.stft",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="stft",
        args_trans={0: "audio_to_tensor_no_batch"},
        kwargs_trans={"y": "audio_to_tensor_no_batch"},
        arg_renames={
            "n_fft": "n_fft",
            "hop_length": "hop_length",
            "win_length": "win_length"
        },
        injected_kwargs={"return_complex": True},
        output_rule="keep_on_device"
    ))

    # librosa.istft - P2
    registry.register(OpSpec(
        source="librosa.istft",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="istft",
        args_trans={0: "ensure_tensor"},
        arg_renames={
            "hop_length": "hop_length",
            "win_length": "win_length",
            "n_fft": "n_fft"
        },
        output_rule="keep_on_device"
    ))

    # librosa.amplitude_to_db - P2
    registry.register(OpSpec(
        source="librosa.amplitude_to_db",
        strategy=StrategyType.FACTORY_OP,
        priority=2,
        target_lib="torchaudio.transforms",
        target_func="AmplitudeToDB",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # librosa.db_to_amplitude - P2
    registry.register(OpSpec(
        source="librosa.db_to_amplitude",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torchaudio.functional",
        target_func="DB_to_amplitude",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # librosa.effects.time_stretch - P3
    registry.register(OpSpec(
        source="librosa.effects.time_stretch",
        strategy=StrategyType.FACTORY_OP,
        priority=3,
        target_lib="torchaudio.transforms",
        target_func="TimeStretch",
        args_trans={0: "audio_to_tensor"},
        output_rule="keep_on_device"
    ))

    # librosa.effects.pitch_shift - P3
    registry.register(OpSpec(
        source="librosa.effects.pitch_shift",
        strategy=StrategyType.FACTORY_OP,
        priority=3,
        target_lib="torchaudio.transforms",
        target_func="PitchShift",
        args_trans={0: "audio_to_tensor"},
        kwargs_trans={"y": "audio_to_tensor"},
        arg_renames={"sr": "sample_rate", "n_steps": "n_steps"},
        output_rule="keep_on_device"
    ))

    # =====================================================
    # 3. SciPy Signal -> PyTorch (StandardOp)
    # =====================================================

    # scipy.fft.fft
    registry.register(OpSpec(
        source="scipy.fft.fft",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="torch.fft",
        target_func="fft",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # scipy.fft.ifft
    registry.register(OpSpec(
        source="scipy.fft.ifft",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="torch.fft",
        target_func="ifft",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # scipy.fft.rfft
    registry.register(OpSpec(
        source="scipy.fft.rfft",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="torch.fft",
        target_func="rfft",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # scipy.fft.irfft
    registry.register(OpSpec(
        source="scipy.fft.irfft",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="torch.fft",
        target_func="irfft",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # scipy.fft.fft2
    registry.register(OpSpec(
        source="scipy.fft.fft2",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="torch.fft",
        target_func="fft2",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # scipy.fft.ifft2
    registry.register(OpSpec(
        source="scipy.fft.ifft2",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="torch.fft",
        target_func="ifft2",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # scipy.signal.convolve
    registry.register(OpSpec(
        source="scipy.signal.convolve",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="torch.nn.functional",
        target_func="conv1d",
        args_trans={0: "audio_to_tensor", 1: "ensure_tensor"},
        output_rule="keep_on_device",
        notes="参数映射可能需要调整，mode 处理不同"
    ))
