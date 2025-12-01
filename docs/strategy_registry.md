# 迁移策略库完整规格说明 (Migration Strategy Registry Specification)

## 1. 概述

本文档定义了数据管道迁移模块的**策略库 (Strategy Registry)**，包括：
- 策略类型定义
- OpSpec 数据结构
- 处理器规则命名标准
- 完整的函数映射清单

---

## 2. 策略类型 (Strategy Types)

```python
class StrategyType(Enum):
    MOVE_ONLY = "MoveOnly"           # 仅搬运 Tensor (PyTorch 原生)
    STANDARD_OP = "StandardOp"       # 函数替换 (cv2 -> kornia, numpy -> torch)
    FACTORY_OP = "FactoryOp"         # 对象工厂 (librosa -> torchaudio class)
    PIPELINE_OP = "PipelineOp"       # 管道容器 (Compose)
    NON_MIGRATABLE = "NonMigratable" # 显式标记不可迁移
```

| 类型 | 说明 | 典型场景 |
|------|------|---------|
| **MoveOnly** | 只搬运输入到 GPU，原函数自动 dispatch | PyTorch/TorchVision/Torchaudio 原生 |
| **StandardOp** | 通用映射：Input → Processor → ArgMapper → Backend | cv2.resize, np.add |
| **FactoryOp** | 工厂模式：创建 GPU Transform 实例并执行 | librosa.melspectrogram |
| **PipelineOp** | 管道编译：将 CPU 容器转为 GPU Sequential | albumentations.Compose |
| **NonMigratable** | 显式标记不可迁移，避免误判 | cv2.imread, I/O 操作 |

---

## 3. OpSpec 数据结构

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable

@dataclass
class OpSpec:
    """迁移操作规格说明书"""

    # --- 1. 身份识别 ---
    source: str                    # 源函数全路径 "cv2.resize"
    strategy: StrategyType = StrategyType.STANDARD_OP
    priority: int = 1              # 0=Critical, 1=High, 2=Medium, 3=Low

    # --- 2. 目标定位 (Lazy Loading) ---
    target_lib: str = ""           # "kornia.geometry.transform"
    target_func: str = ""          # "resize" (FactoryOp 时为类名)

    # --- 3. 输入数据流处理 (Data Flow) ---
    # 位置参数处理规则: Index -> Rule Name
    args_trans: Dict[int, str] = field(default_factory=dict)
    # 关键字参数处理规则: Arg Name -> Rule Name
    kwargs_trans: Dict[str, str] = field(default_factory=dict)

    # --- 4. 参数签名适配 (Signature Adaptation) ---
    # 参数改名: Old Name -> New Name
    arg_renames: Dict[str, str] = field(default_factory=dict)
    # 参数值转换: Param Name -> Transform Function/Name
    arg_value_maps: Dict[str, Union[str, Callable]] = field(default_factory=dict)
    # 注入默认参数: 目标函数需要的额外参数
    injected_kwargs: Dict[str, Any] = field(default_factory=dict)
    # 结构性调整: 预定义的特殊动作列表
    structural_ops: List[str] = field(default_factory=list)

    # --- 5. 输出处理 (Output) ---
    output_rule: str = "keep_on_device"  # "to_numpy", "bchw_to_hwc"

    # --- 6. 运行时控制 ---
    fallback_condition: Optional[Callable] = None
    notes: str = ""
```

---

## 4. 处理器规则命名标准 (Processor Rules)

### 4.1 Input Processors (输入数据转换)

用于 `args_trans` 和 `kwargs_trans` 字段。

#### A. 通用 Tensor 转换

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `ensure_tensor` | PyTorch 原生, NumPy | 若是 Tensor: `to(device)`；若是 Numpy/List: `torch.as_tensor(x).to(device)` |
| `to_tensor_float` | Audio, SciPy | 同上，转 Tensor 后强制 `.float()` (float32) |
| `to_tensor_long` | 索引、标签 | 同上，转 Tensor 后强制 `.long()` (int64) |
| `list_to_tensor_stack` | np.stack, torch.cat | 输入是 `List[Array]`，转为 `List[Tensor]` 并搬运到 GPU |

#### B. 计算机视觉 (CV) 专用

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `cv2_hwc_to_bchw` | cv2.* 输入图像 | `from_numpy(HWC)` → `permute(2,0,1)` → `unsqueeze(0)` → `.to(device).float()` |
| `cv2_hwc_to_chw` | cv2.* 无 batch | `from_numpy(HWC)` → `permute(2,0,1)` → `.to(device).float()` |
| `pil_to_tensor` | PIL | 调用 `v2.functional.to_image` 或 `to_tensor` 转 GPU |
| `bbox_list_to_tensor` | Detection | 将 `[[x,y,w,h], ...]` 转为 Tensor |
| `mask_to_tensor` | Segmentation | 将 mask array 转为 bool/long Tensor |

#### C. 音频专用

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `audio_to_tensor` | librosa.* | `from_numpy` → `.float()` → `.to(device)` → `unsqueeze(0)` (添加 batch) |
| `audio_to_tensor_no_batch` | 已有 batch | `from_numpy` → `.float()` → `.to(device)` |

#### D. 直通与忽略

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `pass` | 非数据参数 | 不做任何处理，直接透传（如 kernel_size, flags） |
| `drop` | 不支持的参数 | 构建参数时跳过此参数（慎用） |
| `to_device` | 已是 Tensor | 只做 `.to(device)` |

---

### 4.2 Arg Value Mappers (参数值变换)

用于 `arg_value_maps` 字段。

#### A. 几何参数调整

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `swap_hw` | cv2.resize (dsize) | `(W, H)` → `(H, W)` |
| `xy_to_yx` | 坐标点变换 | `(x, y)` → `(y, x)` |
| `ensure_tuple` | size 参数 | `int 256` → `(256, 256)` |
| `ensure_list` | 多值参数 | 确保是 list 类型 |
| `to_float` | 数值参数 | 强制转 float |
| `to_int` | 数值参数 | 强制转 int |

#### B. OpenCV 常量/枚举映射

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `cv2_flip_code` | cv2.flip | `0` → `[-2]`, `1` → `[-1]`, `-1` → `[-2,-1]` |
| `cv2_border_type` | copyMakeBorder | `cv2.BORDER_REFLECT` → `"reflect"` |
| `cv2_interp_mode` | resize | `cv2.INTER_LINEAR` → `"bilinear"` |
| `cv2_color_code` | cvtColor | `cv2.COLOR_BGR2RGB` → 选择对应 kornia 函数 |

#### C. PIL 常量映射

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `pil_interp_mode` | Image.resize | `PIL.Image.BILINEAR` → `InterpolationMode.BILINEAR` |
| `pil_flip_method` | Image.transpose | `FLIP_LEFT_RIGHT` → 选择 hflip/vflip |

#### D. 音频参数映射

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `librosa_pad_mode` | melspectrogram | 确保 `pad_mode` 参数对齐 |

---

### 4.3 Output Processors (输出结果处理)

用于 `output_rule` 字段。

| Rule Name | 适用场景 | 逻辑描述 |
|-----------|---------|---------|
| `keep_on_device` | 默认推荐 | 结果保持在 GPU Tensor，供后续节点使用 |
| `to_numpy` | 必须回退 CPU | `.detach().cpu().numpy()` |
| `to_numpy_uint8` | 图像输出 | `.detach().cpu().numpy().astype(np.uint8)` |
| `bchw_to_hwc` | 对接 CPU CV 库 | `squeeze(0).permute(1,2,0).cpu().numpy()` |
| `bchw_to_hwc_uint8` | OpenCV 格式还原 | `squeeze(0).permute(1,2,0).cpu().numpy().astype(uint8)` |
| `chw_to_hwc` | 无 batch 图像 | `permute(1,2,0).cpu().numpy()` |
| `to_pil` | 对接 PIL | Tensor 转 PIL Image 对象 |
| `squeeze_batch` | 移除 batch 维度 | `.squeeze(0)` |

---

### 4.4 Structural Operations (结构性操作)

用于 `structural_ops` 字段。

| Op Name | 逻辑描述 |
|---------|---------|
| `swap_args_0_1` | 交换位置参数 0 和 1 |
| `swap_args_1_2` | 交换位置参数 1 和 2 |
| `args_to_kwargs` | 将所有位置参数转为关键字参数 |
| `flatten_list_arg_0` | 展开第一个参数（如果是列表） |

---

## 5. 完整策略清单

### 5.1 不可迁移 (NonMigratable)

显式标记，避免误判。

| 源函数 | 原因 | 替代方案 |
|--------|------|---------|
| `cv2.imread` | I/O 密集 | `torchvision.io.read_image` (需 NVJPG) |
| `cv2.imwrite` | I/O | - |
| `cv2.VideoCapture` | I/O | `torchvision.io.read_video` |
| `PIL.Image.open` | I/O | `torchvision.io.read_image` |
| `PIL.Image.save` | I/O | - |
| `librosa.load` | 文件 I/O | `torchaudio.load` (仍是 CPU) |
| `soundfile.read` | 文件 I/O | - |
| `pandas.read_csv` | I/O | - |
| `pandas.read_parquet` | I/O | - |

---

### 5.2 PyTorch 原生 (MoveOnly) - P0

| 源函数 | 策略 | 备注 |
|--------|------|------|
| `torch.*` 所有 Tensor 方法 | MoveOnly | 自动 dispatch |
| `torch.nn.functional.*` | MoveOnly | |
| `torchvision.transforms.v2.*` | MoveOnly | 需确保内部都支持 Tensor |
| `torchvision.transforms.Compose` | MoveOnly | 前提：内部 transform 支持 Tensor |
| `torchvision.transforms.functional.*` | MoveOnly | |
| `torchaudio.transforms.*` | MoveOnly | Transform 实例需 `.to(device)` |
| `torchaudio.functional.*` | MoveOnly | |

---

### 5.3 OpenCV → Kornia/PyTorch (StandardOp)

#### cv2.resize - P1

```python
OpSpec(
    source="cv2.resize",
    strategy=StrategyType.STANDARD_OP,
    priority=1,
    target_lib="kornia.geometry.transform",
    target_func="resize",
    args_trans={0: "cv2_hwc_to_bchw"},
    kwargs_trans={"src": "cv2_hwc_to_bchw"},
    arg_renames={"dsize": "size"},
    arg_value_maps={"dsize": "swap_hw"},
    output_rule="keep_on_device",
    notes="最常见的数据管道瓶颈"
)
```

#### cv2.cvtColor - P1

```python
# 需要根据 COLOR_* 常量动态选择目标函数
OpSpec(
    source="cv2.cvtColor",
    strategy=StrategyType.STANDARD_OP,
    priority=1,
    target_lib="kornia.color",
    target_func="",  # 动态选择
    args_trans={0: "cv2_hwc_to_bchw"},
    arg_value_maps={"code": "cv2_color_code"},
    fallback_condition=lambda code: code not in SUPPORTED_COLOR_CODES,
    notes="需要 color_code 映射表"
)
```

**cv2.cvtColor 子映射表**:

| COLOR_* 常量 | Kornia 函数 | 支持 |
|-------------|-------------|------|
| `COLOR_BGR2RGB` | `kornia.color.bgr_to_rgb` | ✓ |
| `COLOR_RGB2BGR` | `kornia.color.rgb_to_bgr` | ✓ |
| `COLOR_BGR2GRAY` | `kornia.color.bgr_to_grayscale` | ✓ |
| `COLOR_RGB2GRAY` | `kornia.color.rgb_to_grayscale` | ✓ |
| `COLOR_GRAY2BGR` | `kornia.color.grayscale_to_rgb` | ✓ |
| `COLOR_GRAY2RGB` | `kornia.color.grayscale_to_rgb` | ✓ |
| `COLOR_BGR2HSV` | `kornia.color.bgr_to_hsv` | ✓ |
| `COLOR_HSV2BGR` | `kornia.color.hsv_to_bgr` | ✓ |
| `COLOR_RGB2HSV` | `kornia.color.rgb_to_hsv` | ✓ |
| `COLOR_HSV2RGB` | `kornia.color.hsv_to_rgb` | ✓ |
| `COLOR_BGR2LAB` | `kornia.color.bgr_to_lab` | ✓ |
| `COLOR_LAB2BGR` | `kornia.color.lab_to_bgr` | ✓ |
| `COLOR_BGR2YUV` | `kornia.color.bgr_to_yuv` | ✓ |
| `COLOR_YUV2BGR` | `kornia.color.yuv_to_bgr` | ✓ |
| 其他 | - | ✗ Fallback CPU |

#### cv2.GaussianBlur - P2

```python
OpSpec(
    source="cv2.GaussianBlur",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="kornia.filters",
    target_func="gaussian_blur2d",
    args_trans={0: "cv2_hwc_to_bchw"},
    arg_renames={"ksize": "kernel_size", "sigmaX": "sigma"},
    arg_value_maps={"ksize": "ensure_tuple"},
    output_rule="keep_on_device"
)
```

#### cv2.flip - P2

```python
OpSpec(
    source="cv2.flip",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="flip",
    args_trans={0: "cv2_hwc_to_bchw"},
    arg_renames={"flipCode": "dims"},
    arg_value_maps={"flipCode": "cv2_flip_code"},
    output_rule="keep_on_device"
)
```

#### cv2.normalize - P2

```python
OpSpec(
    source="cv2.normalize",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="kornia.enhance",
    target_func="normalize_min_max",
    args_trans={0: "cv2_hwc_to_bchw"},
    output_rule="keep_on_device",
    notes="语义可能与 cv2 不完全一致"
)
```

#### cv2.warpAffine - P3

```python
OpSpec(
    source="cv2.warpAffine",
    strategy=StrategyType.STANDARD_OP,
    priority=3,
    target_lib="kornia.geometry.transform",
    target_func="warp_affine",
    args_trans={0: "cv2_hwc_to_bchw", 1: "ensure_tensor"},
    arg_renames={"M": "M", "dsize": "dsize"},
    arg_value_maps={"dsize": "swap_hw"},
    output_rule="keep_on_device"
)
```

#### cv2.threshold - P3

```python
OpSpec(
    source="cv2.threshold",
    strategy=StrategyType.STANDARD_OP,
    priority=3,
    target_lib="",  # 自定义实现
    target_func="cv2_threshold_impl",
    args_trans={0: "cv2_hwc_to_bchw"},
    output_rule="keep_on_device",
    notes="使用 torch.where 组合实现"
)
```

#### cv2.copyMakeBorder - P3

```python
OpSpec(
    source="cv2.copyMakeBorder",
    strategy=StrategyType.STANDARD_OP,
    priority=3,
    target_lib="torch.nn.functional",
    target_func="pad",
    args_trans={0: "cv2_hwc_to_bchw"},
    arg_value_maps={"borderType": "cv2_border_type"},
    structural_ops=["cv2_border_args_to_pad"],  # (top,bottom,left,right) -> (left,right,top,bottom)
    output_rule="keep_on_device"
)
```

---

### 5.4 PIL → TorchVision (StandardOp) - P2

#### Image.resize

```python
OpSpec(
    source="PIL.Image.Image.resize",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torchvision.transforms.v2.functional",
    target_func="resize",
    args_trans={0: "pil_to_tensor"},
    kwargs_trans={"size": "pass"},
    arg_value_maps={"resample": "pil_interp_mode"},
    output_rule="keep_on_device"
)
```

#### Image.crop

```python
OpSpec(
    source="PIL.Image.Image.crop",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torchvision.transforms.v2.functional",
    target_func="crop",
    args_trans={0: "pil_to_tensor"},
    structural_ops=["pil_box_to_crop_params"],  # (l,t,r,b) -> (t,l,h,w)
    output_rule="keep_on_device"
)
```

#### Image.rotate

```python
OpSpec(
    source="PIL.Image.Image.rotate",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torchvision.transforms.v2.functional",
    target_func="rotate",
    args_trans={0: "pil_to_tensor"},
    output_rule="keep_on_device"
)
```

#### Image.transpose

```python
OpSpec(
    source="PIL.Image.Image.transpose",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torchvision.transforms.v2.functional",
    target_func="",  # 根据 method 动态选择 hflip/vflip
    args_trans={0: "pil_to_tensor"},
    arg_value_maps={"method": "pil_flip_method"},
    output_rule="keep_on_device"
)
```

---

### 5.5 NumPy → PyTorch (StandardOp)

#### 基础转换 - P1

```python
# np.array / np.asarray
OpSpec(
    source="numpy.array",
    strategy=StrategyType.STANDARD_OP,
    priority=1,
    target_lib="torch",
    target_func="tensor",
    args_trans={0: "pass"},  # 原始数据直接传
    injected_kwargs={"device": "TARGET_DEVICE"},
    output_rule="keep_on_device"
)

OpSpec(
    source="numpy.asarray",
    strategy=StrategyType.STANDARD_OP,
    priority=1,
    target_lib="torch",
    target_func="as_tensor",
    args_trans={0: "pass"},
    injected_kwargs={"device": "TARGET_DEVICE"},
    output_rule="keep_on_device"
)
```

#### 数组创建 - P2

```python
# np.zeros
OpSpec(
    source="numpy.zeros",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="zeros",
    args_trans={0: "pass"},
    injected_kwargs={"device": "TARGET_DEVICE"},
    output_rule="keep_on_device"
)

# np.ones
OpSpec(
    source="numpy.ones",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="ones",
    args_trans={0: "pass"},
    injected_kwargs={"device": "TARGET_DEVICE"},
    output_rule="keep_on_device"
)

# np.zeros_like / np.ones_like
OpSpec(
    source="numpy.zeros_like",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="zeros_like",
    args_trans={0: "ensure_tensor"},
    output_rule="keep_on_device"
)
```

#### 形状操作 - P2

```python
# np.reshape
OpSpec(
    source="numpy.reshape",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="reshape",
    args_trans={0: "ensure_tensor"},
    output_rule="keep_on_device"
)

# np.transpose
OpSpec(
    source="numpy.transpose",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="permute",
    args_trans={0: "ensure_tensor"},
    arg_renames={"axes": "dims"},
    output_rule="keep_on_device"
)

# np.expand_dims
OpSpec(
    source="numpy.expand_dims",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="unsqueeze",
    args_trans={0: "ensure_tensor"},
    arg_renames={"axis": "dim"},
    output_rule="keep_on_device"
)

# np.squeeze
OpSpec(
    source="numpy.squeeze",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="squeeze",
    args_trans={0: "ensure_tensor"},
    arg_renames={"axis": "dim"},
    output_rule="keep_on_device"
)
```

#### 拼接操作 - P2

```python
# np.concatenate
OpSpec(
    source="numpy.concatenate",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="cat",
    args_trans={0: "list_to_tensor_stack"},
    arg_renames={"axis": "dim"},
    output_rule="keep_on_device"
)

# np.stack
OpSpec(
    source="numpy.stack",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="stack",
    args_trans={0: "list_to_tensor_stack"},
    arg_renames={"axis": "dim"},
    output_rule="keep_on_device"
)

# np.split
OpSpec(
    source="numpy.split",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="split",
    args_trans={0: "ensure_tensor"},
    output_rule="keep_on_device"
)
```

#### 数学运算 - P2

```python
# 二元运算: add, sub, mul, div, matmul, dot
for np_func, torch_func in [
    ("add", "add"), ("subtract", "sub"), ("multiply", "mul"),
    ("divide", "div"), ("matmul", "matmul"), ("dot", "matmul")
]:
    OpSpec(
        source=f"numpy.{np_func}",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func=torch_func,
        args_trans={0: "ensure_tensor", 1: "ensure_tensor"},
        output_rule="keep_on_device"
    )

# 一元运算: abs, sqrt, exp, log, sin, cos, tan
for func in ["abs", "sqrt", "exp", "log", "sin", "cos", "tan"]:
    OpSpec(
        source=f"numpy.{func}",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func=func,
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    )

# np.power
OpSpec(
    source="numpy.power",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="pow",
    args_trans={0: "ensure_tensor", 1: "ensure_tensor"},
    output_rule="keep_on_device"
)

# np.clip
OpSpec(
    source="numpy.clip",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="clamp",
    args_trans={0: "ensure_tensor"},
    arg_renames={"a_min": "min", "a_max": "max"},
    output_rule="keep_on_device"
)
```

#### 统计运算 - P2

```python
# mean, sum, max, min, std, var
for func in ["mean", "sum", "std", "var"]:
    OpSpec(
        source=f"numpy.{func}",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func=func,
        args_trans={0: "ensure_tensor"},
        arg_renames={"axis": "dim"},
        output_rule="keep_on_device"
    )

# np.max / np.min (返回值)
OpSpec(
    source="numpy.max",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="max",
    args_trans={0: "ensure_tensor"},
    arg_renames={"axis": "dim"},
    output_rule="keep_on_device"
)

# np.argmax / np.argmin
OpSpec(
    source="numpy.argmax",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="argmax",
    args_trans={0: "ensure_tensor"},
    arg_renames={"axis": "dim"},
    output_rule="keep_on_device"
)
```

#### 逻辑运算 - P2

```python
# np.where
OpSpec(
    source="numpy.where",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="where",
    args_trans={0: "ensure_tensor", 1: "ensure_tensor", 2: "ensure_tensor"},
    output_rule="keep_on_device"
)
```

#### 随机数 (含 Seed 注意) - P2

```python
# np.random.rand
OpSpec(
    source="numpy.random.rand",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="rand",
    injected_kwargs={"device": "TARGET_DEVICE"},
    output_rule="keep_on_device",
    notes="Seed 机制不同，需同步 torch.manual_seed"
)

# np.random.randn
OpSpec(
    source="numpy.random.randn",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="randn",
    injected_kwargs={"device": "TARGET_DEVICE"},
    output_rule="keep_on_device"
)

# np.random.randint
OpSpec(
    source="numpy.random.randint",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="randint",
    arg_renames={"low": "low", "high": "high", "size": "size"},
    injected_kwargs={"device": "TARGET_DEVICE"},
    output_rule="keep_on_device"
)
```

---

### 5.6 Audio: Librosa → Torchaudio (FactoryOp)

#### librosa.resample - P1

```python
OpSpec(
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
)
```

#### librosa.feature.melspectrogram - P1

```python
OpSpec(
    source="librosa.feature.melspectrogram",
    strategy=StrategyType.FACTORY_OP,
    priority=1,
    target_lib="torchaudio.transforms",
    target_func="MelSpectrogram",
    args_trans={0: "audio_to_tensor"},
    kwargs_trans={"y": "audio_to_tensor"},
    arg_renames={"sr": "sample_rate", "n_fft": "n_fft", "hop_length": "hop_length", "n_mels": "n_mels"},
    injected_kwargs={"pad_mode": "reflect"},  # 对齐 librosa 默认值
    output_rule="keep_on_device"
)
```

#### librosa.feature.mfcc - P2

```python
OpSpec(
    source="librosa.feature.mfcc",
    strategy=StrategyType.FACTORY_OP,
    priority=2,
    target_lib="torchaudio.transforms",
    target_func="MFCC",
    args_trans={0: "audio_to_tensor"},
    kwargs_trans={"y": "audio_to_tensor"},
    arg_renames={"sr": "sample_rate", "n_mfcc": "n_mfcc"},
    output_rule="keep_on_device"
)
```

#### librosa.stft - P2

```python
OpSpec(
    source="librosa.stft",
    strategy=StrategyType.STANDARD_OP,
    priority=2,
    target_lib="torch",
    target_func="stft",
    args_trans={0: "audio_to_tensor"},
    kwargs_trans={"y": "audio_to_tensor"},
    arg_renames={"n_fft": "n_fft", "hop_length": "hop_length", "win_length": "win_length"},
    injected_kwargs={"return_complex": True},
    output_rule="keep_on_device"
)
```

#### librosa.amplitude_to_db - P2

```python
OpSpec(
    source="librosa.amplitude_to_db",
    strategy=StrategyType.FACTORY_OP,
    priority=2,
    target_lib="torchaudio.transforms",
    target_func="AmplitudeToDB",
    args_trans={0: "ensure_tensor"},
    output_rule="keep_on_device"
)
```

#### librosa.effects.time_stretch - P3

```python
OpSpec(
    source="librosa.effects.time_stretch",
    strategy=StrategyType.FACTORY_OP,
    priority=3,
    target_lib="torchaudio.transforms",
    target_func="TimeStretch",
    args_trans={0: "audio_to_tensor"},
    output_rule="keep_on_device"
)
```

---

### 5.7 SciPy (StandardOp) - P3

```python
# scipy.fft.fft
OpSpec(
    source="scipy.fft.fft",
    strategy=StrategyType.STANDARD_OP,
    priority=3,
    target_lib="torch.fft",
    target_func="fft",
    args_trans={0: "ensure_tensor"},
    output_rule="keep_on_device"
)

# scipy.fft.ifft
OpSpec(
    source="scipy.fft.ifft",
    strategy=StrategyType.STANDARD_OP,
    priority=3,
    target_lib="torch.fft",
    target_func="ifft",
    args_trans={0: "ensure_tensor"},
    output_rule="keep_on_device"
)

# scipy.ndimage.rotate
OpSpec(
    source="scipy.ndimage.rotate",
    strategy=StrategyType.STANDARD_OP,
    priority=3,
    target_lib="kornia.geometry.transform",
    target_func="rotate",
    args_trans={0: "cv2_hwc_to_bchw"},
    output_rule="keep_on_device"
)
```

---

### 5.8 Pandas (StandardOp) - P3 (有限支持)

```python
# df.values / df.to_numpy()
OpSpec(
    source="pandas.DataFrame.values",
    strategy=StrategyType.STANDARD_OP,
    priority=3,
    target_lib="torch",
    target_func="tensor",
    args_trans={0: "pass"},  # 属性访问，特殊处理
    injected_kwargs={"device": "TARGET_DEVICE"},
    output_rule="keep_on_device",
    notes="实际是属性访问，需要特殊 hook"
)
```

**注**：Pandas 运算符重载 (`df['a'] + df['b']`) 暂不支持。

---

### 5.9 容器/管道 (PipelineOp) - P3

```python
# albumentations.Compose
OpSpec(
    source="albumentations.Compose",
    strategy=StrategyType.PIPELINE_OP,
    priority=3,
    target_lib="kornia.augmentation",
    target_func="AugmentationSequential",
    notes="需要编译转换内部 transforms 列表"
)
```

---

## 6. 实现阶段规划

### Phase 1 (MVP) - 立竿见影

| 类别 | 函数 |
|------|------|
| MoveOnly | PyTorch/TorchVision/Torchaudio 原生 |
| StandardOp | `cv2.resize`, `cv2.cvtColor` (常见 code) |
| StandardOp | `np.array`, `np.asarray` |
| FactoryOp | `librosa.resample`, `librosa.feature.melspectrogram` |
| NonMigratable | 显式标记 I/O 函数 |

### Phase 2 - 补全常用

| 类别 | 函数 |
|------|------|
| StandardOp | 剩余 NumPy 函数 (reshape, transpose, 数学运算等) |
| StandardOp | 剩余 cv2 函数 (flip, blur, normalize) |
| StandardOp | PIL 函数 (resize, crop, rotate) |
| StandardOp | NumPy Random (含 Seed 同步) |
| FactoryOp | 剩余 librosa 函数 |

### Phase 3 - 攻坚

| 类别 | 函数 |
|------|------|
| StandardOp | SciPy (fft, signal) |
| StandardOp | Pandas (有限支持) |
| PipelineOp | albumentations.Compose |

---

## 7. 示例配置

### 完整示例：cv2.resize

```python
OpSpec(
    # 身份
    source="cv2.resize",
    strategy=StrategyType.STANDARD_OP,
    priority=1,

    # 目标
    target_lib="kornia.geometry.transform",
    target_func="resize",

    # 输入转换
    args_trans={0: "cv2_hwc_to_bchw"},
    kwargs_trans={"src": "cv2_hwc_to_bchw"},

    # 参数适配
    arg_renames={"dsize": "size"},
    arg_value_maps={"dsize": "swap_hw"},
    injected_kwargs={},
    structural_ops=[],

    # 输出
    output_rule="keep_on_device",

    # 运行时
    fallback_condition=None,
    notes="最常见的数据管道瓶颈，需交换 (W,H) 为 (H,W)"
)
```

### 完整示例：librosa.feature.melspectrogram

```python
OpSpec(
    # 身份
    source="librosa.feature.melspectrogram",
    strategy=StrategyType.FACTORY_OP,
    priority=1,

    # 目标
    target_lib="torchaudio.transforms",
    target_func="MelSpectrogram",

    # 输入转换
    args_trans={0: "audio_to_tensor"},
    kwargs_trans={"y": "audio_to_tensor"},

    # 参数适配
    arg_renames={
        "sr": "sample_rate",
        "n_fft": "n_fft",
        "hop_length": "hop_length",
        "n_mels": "n_mels"
    },
    arg_value_maps={},
    injected_kwargs={"pad_mode": "reflect"},
    structural_ops=[],

    # 输出
    output_rule="keep_on_device",

    # 运行时
    fallback_condition=None,
    notes="Librosa 和 Torchaudio 默认参数有差异，需注入 pad_mode"
)
```

---

## 8. 扩展指南

### 添加新函数支持

1. 确定策略类型 (MoveOnly/StandardOp/FactoryOp/PipelineOp/NonMigratable)
2. 确定目标库和函数
3. 分析参数差异：
   - 数据格式差异 → `args_trans` / `kwargs_trans`
   - 参数命名差异 → `arg_renames`
   - 参数值差异 → `arg_value_maps`
   - 缺失参数 → `injected_kwargs`
   - 结构差异 → `structural_ops`
4. 确定输出处理规则
5. 测试并添加到 Registry

### 添加新处理器规则

在 `processors.py` 中添加新函数，并在规则命名表中注册。
