这是一个非常核心的梳理工作。为了构建一个健壮的**迁移策略库（Migration Registry）**，我们需要将常见的数据处理操作进行详尽的分类映射。

我将按照**源生态（Origin Ecosystem）**进行分类，并详细列出**操作类型**、**常用函数**、**GPU 替代方案**以及**关键的参数/数据适配策略**。

特别地，我将**容器类操作（如 `Compose`）**单独列为一类，因为它们的迁移策略通常涉及整个管道的优化。

---

### 1. 容器与控制流 (Containers & Pipelines) —— *新增/重点*

这部分决定了数据流的整体调度。如果能拦截容器，通常比拦截容器内的单个算子效率更高。

| 操作名称 | 源对象 (Origin) | 目标对象 (Target) | 迁移策略 (Migration Action) | 注意事项 |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch Compose** | `torchvision.transforms.Compose` | **原生支持** (只要内部 Transforms 是 v2) | 1. **Pre-Hook**: `input.to(device)`<br>2. **Execution**: 依次调用内部 transforms<br>3. **Post-Hook**: 保持在 GPU | **前提**：内部所有 transform 必须支持 Tensor 输入（建议升级到 `v2`）。如果混杂了 PIL 操作，会报错。 |
| **RandomApply** | `torchvision.transforms.RandomApply` | **原生支持** | 同上。PyTorch 的随机逻辑 (`torch.rand`) 支持 GPU。 | 确保随机种子同步（如果涉及多卡）。 |
| **Albumentations Compose** | `albumentations.Compose` | `kornia.augmentation.container.Sequential` | **替换实现**。将 A 库的 transforms 列表映射为 Kornia 的 augmentation 列表。 | 需要编写一个转换器，将 A 的参数配置转为 Kornia 的配置。 |
| **Sequential** | `torch.nn.Sequential` | **原生支持** | 输入转 GPU，直接执行 `forward`。 | 通常用于模型层，但在预处理中也常见。 |

---

### 2. PyTorch 生态 (Native Support)

PyTorch 自身的函数最容易迁移，核心在于**输入搬运**。

#### 2.1 Tensor 变换与计算 (Core)
*   **策略**: `MoveTensor(device)` -> `OriginalFunc`

| 操作类型 | 常用函数 | 行为描述 |
| :--- | :--- | :--- |
| **维度操作** | `permute`, `transpose`, `reshape`, `view`, `unsqueeze`, `squeeze`, `flatten` | 纯元数据操作，GPU 上极快。 |
| **拼接/分割** | `cat`, `stack`, `split`, `chunk` | 涉及内存拷贝，GPU 带宽优势明显。 |
| **数学运算** | `add`, `sub`, `mul`, `div`, `matmul`, `bmm`, `abs`, `pow`, `exp`, `log` | 算术密集型，GPU 加速核心。 |
| **归约统计** | `mean`, `sum`, `max`, `min`, `std`, `var` | 并行归约，GPU 优势巨大。 |
| **类型转换** | `float()`, `long()`, `byte()`, `type_as()` | 数据类型转换。 |

#### 2.2 视觉变换 (TorchVision v2 / Kornia)
*   **策略**: `MoveTensor(device)` -> `OriginalFunc` (v2) 或 `ReplaceWithKornia`

| 操作类型 | 常用函数 (`transforms.v2`) | 常用函数 (`kornia`) | 备注 |
| :--- | :--- | :--- | :--- |
| **尺寸调整** | `v2.Resize` | `kornia.geometry.resize` | v2 支持 Antialias。 |
| **裁剪** | `v2.CenterCrop`, `v2.RandomCrop` | `kornia.augmentation.CenterCrop` | 涉及索引操作。 |
| **翻转/旋转** | `v2.RandomHorizontalFlip`, `v2.RandomRotation` | `kornia.augmentation.RandomHorizontalFlip` | 几何变换。 |
| **色彩调整** | `v2.ColorJitter`, `v2.Normalize` | `kornia.enhance.Normalize` | 逐像素操作，GPU 极快。 |
| **类型转换** | `v2.ToDtype`, `v2.ToImage` | - | v2 引入的新 API，用于替代 ToTensor。 |

---

### 3. NumPy 生态 (CPU Array -> GPU Tensor)

核心在于**类型映射**和**函数名映射**。必须将 NumPy 数组转为 GPU Tensor。

*   **策略**: `NumpyToTensor(device)` -> `MappedTorchFunc`

| 操作类型 | NumPy 函数 | 映射到的 PyTorch 函数 | 参数差异/适配逻辑 |
| :--- | :--- | :--- | :--- |
| **数组创建** | `np.array`, `np.asarray` | `torch.tensor` | - |
| **全0/1** | `np.zeros`, `np.ones` | `torch.zeros`, `torch.ones` | 需注入 `device=...` 参数。 |
| **形状变换** | `np.reshape` | `torch.reshape` | - |
| **维度交换** | `np.transpose` | `torch.permute` | NumPy `axes` 参数对应 Torch `dims`。 |
| **维度扩展** | `np.expand_dims` | `torch.unsqueeze` | 参数名 `axis` -> `dim`。 |
| **拼接** | `np.concatenate` | `torch.cat` | 参数名 `axis` -> `dim`。 |
| **堆叠** | `np.stack` | `torch.stack` | - |
| **数学运算** | `np.add`, `np.sin`, ... | `torch.add`, `torch.sin`, ... | 基本一一对应。 |
| **矩阵乘法** | `np.dot`, `np.matmul` | `torch.matmul` | - |
| **逻辑运算** | `np.where` | `torch.where` | - |
| **最值索引** | `np.argmax`, `np.argmin` | `torch.argmax`, `torch.argmin` | - |

---

### 4. OpenCV 生态 (Image Processing)

最复杂的部分。核心在于 **HWC (uint8) <-> BCHW (float)** 的转换以及 **BGR <-> RGB**。

*   **策略**: `ImageHWC2CHW(device, normalize=False)` -> `KorniaImplementation`

| 操作类型 | OpenCV 函数 | Kornia / Torch 替代 | 关键适配逻辑 (Adapter Logic) |
| :--- | :--- | :--- | :--- |
| **读取** | `cv2.imread` | (不建议迁移) | IO 密集型，通常留在 CPU。若必须迁移，使用 `torchvision.io.read_image` (NVJPG)。 |
| **缩放** | `cv2.resize` | `kornia.geometry.resize` | **参数陷阱**: cv2 `dsize=(W,H)`, kornia `size=(H,W)`。需要交换参数顺序。 |
| **色彩转换** | `cv2.cvtColor` | `kornia.color.bgr_to_rgb` (等) | 需解析 `cv2.COLOR_BGR2RGB` 等常量，映射到 Kornia 对应函数。 |
| **翻转** | `cv2.flip` | `torch.flip` | cv2 `flipCode` (0,1,-1) 需映射到 torch `dims`。 |
| **旋转** | `cv2.warpAffine` (矩阵) | `kornia.geometry.warp_affine` | 需要将 CPU 上的变换矩阵也转为 Tensor 并 `.to(device)`。 |
| **高斯模糊** | `cv2.GaussianBlur` | `kornia.filters.gaussian_blur2d` | 参数 `ksize` 和 `sigma` 需适配。 |
| **阈值化** | `cv2.threshold` | `kornia.enhance.threshold` (类似) | 或直接用 `torch.where(img > thresh, ...)` 实现。 |
| **填充** | `cv2.copyMakeBorder` | `torch.nn.functional.pad` | cv2 `(top, bottom, left, right)` -> torch `(left, right, top, bottom)` (顺序不同!)。 |
| **归一化** | `cv2.normalize` | `kornia.enhance.normalize` | cv2 是 MinMax 或 Norm，Kornia 通常是 Mean/Std。需确认语义。 |

---

### 5. PIL 生态 (Pillow)

PIL 是对象式的，无法像 NumPy 那样直接由 Tensor 承载（除非 hack）。通常建议将 PIL 操作替换为 `torchvision.transforms.v2`。

*   **策略**: `PILToTensor(device)` -> `TorchVisionV2Func`

| 操作类型 | PIL 方法 (Image类) | TorchVision v2 (`functional`) | 备注 |
| :--- | :--- | :--- | :--- |
| **缩放** | `img.resize(size)` | `F.resize(img_t, size)` | 需处理插值方法映射 (PIL.BICUBIC -> InterpolationMode.BICUBIC)。 |
| **裁剪** | `img.crop(box)` | `F.crop(img_t, top, left, h, w)` | PIL `box=(l, t, r, b)` 需转换为 `(t, l, h, w)`。 |
| **旋转** | `img.rotate(angle)` | `F.rotate(img_t, angle)` | - |
| **翻转** | `img.transpose(METHOD)` | `F.hflip` / `F.vflip` | 需解析 `PIL.Image.FLIP_LEFT_RIGHT` 等常量。 |
| **调整大小** | `img.thumbnail` | (组合操作) | 需计算目标尺寸后调用 `F.resize`。 |

---

### 6. Pandas 生态 (Data Processing)

Pandas 主要用于处理表格数据。

*   **策略**: `DataFrame/Series -> Values(Numpy) -> Tensor`

| 操作类型 | Pandas 操作 | PyTorch 替代 | 备注 |
| :--- | :--- | :--- | :--- |
| **取值** | `df.values`, `df.to_numpy()` | `Tensor` 构造 | 触发数据搬运。 |
| **填充缺失** | `df.fillna(0)` | `torch.nan_to_num` | - |
| **算术运算** | `df['a'] + df['b']` | `tensor_a + tensor_b` | 列运算转为向量运算。 |
| **One-Hot** | `get_dummies` | `torch.nn.functional.one_hot` | 需先将字符串类别转为整数索引。 |

---

### 7. Python 内置与算子 (Built-in Operators)

这些操作通常隐含在代码中，如 `image / 255.0`。

*   **策略**: 只要输入被之前的步骤成功迁移到了 GPU，这些操作会自动分发（Dispatch）到 GPU 执行。

| 操作类型 | Python 算子 | PyTorch 行为 |
| :--- | :--- | :--- |
| **加减乘除** | `+`, `-`, `*`, `/`, `//` | 对应 `torch.add` 等。支持 Tensor与标量、Tensor与Tensor 运算。 |
| **逻辑运算** | `and`, `or`, `not` (对应 `&`, `|`, `~`) | Bitwise 运算，GPU 支持。 |
| **比较** | `>`, `<`, `==` | 返回 Bool Tensor，GPU 支持。 |
| **切片/索引** | `x[0:10]`, `x[:, ::-1]` | **零拷贝** (View) 操作，GPU 上极快。 |

---

### 8. 总结：迁移库的实现优先级建议

1.  **Priority 0 (必须支持)**:
    *   **PyTorch Core**: `to(device)`。
    *   **PyTorch Compose**: 识别并批量迁移输入。
    *   **Operator**: 依赖 Tensor 传播机制。

2.  **Priority 1 (高频瓶颈)**:
    *   **OpenCV Resize**: 这是数据加载中最慢的一环，必须用 Kornia 替换。
    *   **OpenCV CvtColor/Normalize**: 紧随 Resize 之后。
    *   **NumPy Array -> Tensor**: 数据入口点。

3.  **Priority 2 (常用增强)**:
    *   **Augmentations**: Flip, Rotate, Crop (无论是 cv2 还是 PIL)。

4.  **Priority 3 (长尾)**:
    *   Pandas, Scipy, 以及复杂的 Albumentations 管道（后者建议建议用户改写为 Kornia）。

这是一个非常好的切入点。**从“任务场景”出发**，能更准确地捕捉到数据管道中的真实瓶颈，并且能覆盖到一些之前按库分类时容易忽略的领域（特别是**音频处理**和**文本处理**，这两者的预处理往往也是计算密集型的）。

我们将迁移策略库（Migration Registry）的维度重构为：**任务领域 -> 核心操作 -> (原始CPU库 vs 目标GPU库)**。

以下是基于常见机器学习任务的详细梳理：

---

### 1. 计算机视觉 (CV) —— 图像与视频任务
**瓶颈特征**：高I/O，大量的像素级矩阵运算，几何变换复杂。
**主要CPU库**：OpenCV, PIL, Albumentations, Scikit-image, MoviePy (Video).
**主要GPU目标**：TorchVision, Kornia.

#### 1.1 几何变换 (Geometry)
这是数据增强中最常见的操作。

| 操作名称 | 常见 CPU 写法 | 迁移 GPU 方案 (Kornia/TorchVision) | 关键适配点 |
| :--- | :--- | :--- | :--- |
| **Resize** (缩放) | `cv2.resize(img, (w,h))` <br> `pil_img.resize((w,h))` | `kornia.geometry.resize(t, (h,w))` <br> `T.v2.Resize((h,w))` | **坑**: OpenCV是(W,H)，PyTorch是(H,W)。需交换参数。 |
| **Rotate** (旋转) | `cv2.getRotationMatrix2D` + `warpAffine` <br> `scipy.ndimage.rotate` | `kornia.geometry.rotate` <br> `T.v2.RandomRotation` | Kornia 支持可微旋转，直接操作 Tensor。 |
| **Crop** (裁剪) | `img[y:y+h, x:x+w]` (Numpy Slicing) | `t[..., y:y+h, x:x+w]` (Tensor Slicing) <br> `kornia.augmentation.CenterCrop` | Tensor 切片是**零拷贝**的，极快。 |
| **Flip** (翻转) | `cv2.flip(img, code)` <br> `np.flip(img, axis)` | `torch.flip(t, dims)` <br> `kornia.augmentation.RandomHorizontalFlip` | 需映射 flip code 到 dims。 |
| **Affine** (仿射) | `cv2.warpAffine` | `kornia.geometry.warp_affine` | 需将变换矩阵也转为 Tensor。 |

#### 1.2 色彩与像素运算 (Color & Intensity)
逐像素操作，GPU 并行优势最大。

| 操作名称 | 常见 CPU 写法 | 迁移 GPU 方案 | 关键适配点 |
| :--- | :--- | :--- | :--- |
| **色彩空间转换** | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | `kornia.color.bgr_to_rgb(t)` | Kornia 提供了极全的色彩空间转换。 |
| **归一化** | `(img - mean) / std` (Numpy) | `T.v2.Normalize(mean, std)` <br> `kornia.enhance.normalize` | 注意 dtype，通常需先转 float32。 |
| **Gamma 校正** | `np.power(img, gamma)` | `torch.pow(t, gamma)` <br> `kornia.enhance.adjust_gamma` | - |
| **直方图均衡** | `cv2.equalizeHist` | `kornia.enhance.equalize_clahe` | 传统算法在 GPU 上实现较复杂，Kornia 实现了部分。 |
| **高斯模糊** | `cv2.GaussianBlur` | `kornia.filters.gaussian_blur2d` | 卷积操作，GPU 加速比极高。 |

#### 1.3 视频特有操作 (Video)
视频通常是 5D Tensor `(Batch, Channel, Time, Height, Width)`。

| 操作名称 | 常见 CPU 写法 | 迁移 GPU 方案 | 关键适配点 |
| :--- | :--- | :--- | :--- |
| **读取** | `cv2.VideoCapture`, `decord` | `torchvision.io.read_video` (支持NVDEC) | 视频解码是超级瓶颈。若能用 NVDEC 硬件解码最好。 |
| **时序采样** | `frames[::step]` (List/Numpy) | `video_tensor[:, :, ::step, ...]` | Tensor 切片。 |
| **时序维度重排** | `np.transpose` (T,H,W,C) -> (C,T,H,W) | `tensor.permute` | 视频数据 Layout 极其混乱，需严格管理。 |

---

### 2. 音频处理 (Audio) —— 语音识别、TTS、声纹
**瓶颈特征**：**STFT (短时傅里叶变换)** 和重采样是计算密集型操作，往往比模型推理还慢。
**主要CPU库**：Librosa (基于NumPy/Scipy), Pydub, SoundFile.
**主要GPU目标**：**Torchaudio**.

#### 2.1 频谱转换 (Spectrograms)
这是音频任务的核心，从时域转频域。

| 操作名称 | 常见 CPU 写法 (Librosa) | 迁移 GPU 方案 (Torchaudio) | 关键适配点 |
| :--- | :--- | :--- | :--- |
| **加载** | `librosa.load(path, sr)` | `torchaudio.load(path)` | Torchaudio 返回 `(waveform, sample_rate)`。 |
| **重采样** | `librosa.resample` | `torchaudio.transforms.Resample` | 这是最慢的 CPU 操作之一，**必须迁移**。 |
| **STFT** | `librosa.stft` | `torch.stft` | 参数 `n_fft`, `hop_length`, `win_length` 需对齐。 |
| **Mel 频谱** | `librosa.feature.melspectrogram` | `torchaudio.transforms.MelSpectrogram` | **注意**: Librosa 默认 `pad_mode='reflect'`, PyTorch 默认不同，需对齐参数以保证数值一致。 |
| **MFCC** | `librosa.feature.mfcc` | `torchaudio.transforms.MFCC` | 同上。 |
| **分贝转换** | `librosa.amplitude_to_db` | `torchaudio.transforms.AmplitudeToDB` | 对数运算。 |

#### 2.2 波形增强 (Waveform Augmentation)
| 操作名称 | 常见 CPU 写法 | 迁移 GPU 方案 | 备注 |
| :--- | :--- | :--- | :--- |
| **加噪声** | `wav + noise * factor` (Numpy) | `wav + noise * factor` (Tensor) | 简单的 Tensor 加法。 |
| **Time Stretch** | `librosa.effects.time_stretch` | `torchaudio.transforms.TimeStretch` | 频域操作，GPU 加速明显。 |
| **Fade In/Out** | (手动 Numpy 切片乘法) | `torchaudio.transforms.Fade` | - |

---

### 3. 自然语言处理 (NLP) —— 翻译、分类、LLM微调
**瓶颈特征**：主要是字符串操作（Tokenization）。字符串处理 GPU 很不擅长，但**Tensor化后的操作**可以迁移。
**主要CPU库**：HuggingFace Tokenizers, NLTK, Spacy, NumPy.
**主要GPU目标**：PyTorch Core.

#### 3.1 序列处理 (Sequence Processing)
Tokenization 通常发生在 CPU（因为涉及复杂的查表和正则），但产出后的 `List[int]` 应尽快转 GPU。

| 操作名称 | 常见 CPU 写法 | 迁移 GPU 方案 | 关键适配点 |
| :--- | :--- | :--- | :--- |
| **Padding** | `pad_sequence` (循环+Numpy) | `torch.nn.utils.rnn.pad_sequence` | 输入 List[Tensor]，输出 GPU Tensor。 |
| **Mask 生成** | `[1]*len + [0]*pad` (List推导) | `torch.arange` + logic | 利用广播机制在 GPU 生成 Mask，避免 CPU 循环。 |
| **One-Hot** | `sklearn.preprocessing.OneHotEncoder` | `torch.nn.functional.one_hot` | - |
| **Embedding** | (查表) | `torch.nn.Embedding` | 虽然是模型的一部分，但有时会在预处理做。 |

#### 3.2 文本增强 (Text Augmentation)
| 操作名称 | 常见 CPU 写法 | 迁移 GPU 方案 | 备注 |
| :--- | :--- | :--- | :--- |
| **Token Masking** | `numpy.random.choice` 替换 ID | `torch.rand` + `torch.where` | BERT 预训练常见操作，完全可以用 Tensor 实现。 |
| **Mixup** | (在 Embeddings 层面) | `lambda * emb1 + (1-lambda) * emb2` | 向量插值。 |

---

### 4. 结构化数据与科学计算 (Tabular & SciComp)
**主要CPU库**：Pandas, Scikit-learn, Scipy.
**主要GPU目标**：PyTorch Core.

#### 4.1 特征工程 (Feature Engineering)
| 操作名称 | 常见 CPU 写法 (Pandas/Sklearn) | 迁移 GPU 方案 | 关键适配点 |
| :--- | :--- | :--- | :--- |
| **缺失值填充** | `df.fillna(v)` / `SimpleImputer` | `torch.nan_to_num` | - |
| **MinMax 缩放** | `MinMaxScaler` | `(t - min) / (max - min)` | 需预先计算 min/max 统计量。 |
| **Standard 缩放** | `StandardScaler` | `(t - mean) / std` | 需预先计算 mean/std。 |
| **列交互** | `df['a'] * df['b']` | `t[:, i] * t[:, j]` | 向量乘法。 |

#### 4.2 信号处理 (Signal Processing - Scipy)
| 操作名称 | 常见 CPU 写法 (Scipy) | 迁移 GPU 方案 | 备注 |
| :--- | :--- | :--- | :--- |
| **傅里叶变换** | `scipy.fft` | `torch.fft` | 科学计算核心。 |
| **卷积/滤波** | `scipy.signal.convolve` | `torch.nn.functional.conv1d` | 用 CNN 算子实现信号滤波。 |
| **插值** | `scipy.interpolate` | `torch.nn.functional.interpolate` | - |

---

### 5. 迁移策略库的完善建议

基于上述梳理，你的策略库 `Registry` 应该具备一种层级结构，或者使用 `Tags` 来管理，方便检索。

**建议的 Registry 键设计 (Key Design)：**

不要只用函数名，建议用 `Domain` 前缀，或者保留原始库的完整路径。

```python
MIGRATION_REGISTRY = {
    # --- CV Domain ---
    "cv2.resize": Strategy(
        domain="CV",
        backend=kornia.geometry.resize,
        args_adapter=SwapDimsAdapter((1, 0)) # H,W <-> W,H
    ),
    "albumentations.Compose": Strategy(
        domain="CV",
        backend=AlbumentationsToKorniaConverter # 这是一个复杂的转换器工厂
    ),

    # --- Audio Domain ---
    "librosa.feature.melspectrogram": Strategy(
        domain="Audio",
        backend=torchaudio.transforms.MelSpectrogram,
        # Librosa 和 Torchaudio 的默认参数差异巨大，这里需要一个专门的 Config 转换器
        kwargs_adapter=LibrosaToTorchAudioConfigAdapter 
    ),
    "librosa.resample": Strategy(
        domain="Audio",
        backend=torchaudio.transforms.Resample
    ),

    # --- Tensor Ops (NumPy) ---
    "numpy.transpose": Strategy(
        domain="General",
        backend=torch.permute,
        input_processor=NumpyToTensor
    )
}
```


# 迁移策略库架构设计 (Migration Strategy Library Architecture)

## 1. 核心类图与数据结构

我们采用 **“策略模式 (Strategy Pattern)” + “组合模式”**。核心思想是将一个函数的迁移行为拆解为三个独立的步骤：**输入处理 -> 参数重组与后端执行 -> 输出处理**。

### 1.1 基础组件定义

```python
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple

# --- A. 输入处理器 (Input Processor) ---
# 职责：负责将单个参数从 CPU 格式转换为 GPU 格式（如 Numpy -> Tensor）
class InputProcessor:
    def __call__(self, arg: Any, device: str) -> Any:
        raise NotImplementedError

# --- B. 参数映射器 (Argument Mapper) ---
# 职责：负责解决 CPU 库与 GPU 库之间函数签名不一致的问题（如参数改名、顺序交换、默认值填充）
class ArgMapper:
    def __call__(self, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        raise NotImplementedError

# --- C. 后端执行器 (Backend Implementation) ---
# 职责：实际执行 GPU 上的计算逻辑
# 可以是一个简单的 kornia 函数，也可以是一个复杂的闭包
BackendFunc = Callable[..., Any]

# --- D. 输出处理器 (Output Processor) ---
# 职责：决定计算结果去向（留在 GPU 还是回传 CPU，格式是否还原）
class OutputProcessor:
    def __call__(self, result: Any) -> Any:
        raise NotImplementedError

# --- E. 迁移策略实体 (The Strategy Object) ---
@dataclass
class MigrationStrategy:
    domain: str                         # 领域标识 (CV, Audio, etc.)
    input_processors: List[InputProcessor]  # 针对 args 的处理器列表
    kwarg_processors: Dict[str, InputProcessor] # 针对 kwargs 的处理器
    arg_mapper: Optional[ArgMapper]     # 参数重组逻辑 (可选)
    backend: BackendFunc                # GPU 实现函数
    output_processor: OutputProcessor   # 结果处理
```

---

## 2. 详细组件库 (Component Library)

为了复用代码，我们需要预置一批通用的处理器。

### 2.1 通用 Input Processors
*   `EnsureTensor(device, dtype)`: 检查输入，如果是 Tensor 则 `.to(device)`；如果是 Numpy/List，则 `torch.as_tensor(...)`。
*   `ImageHWC2BCHW(device)`: 专门针对 OpenCV 图片。`HWC(uint8) -> BCHW(float32) -> Device`。
*   `PassThrough()`: 对于非数据类参数（如 `kernel_size`, `flag`），不做任何处理。

### 2.2 通用 Arg Mappers
*   `SwapArgs(indices=(1,0))`: 交换前两个位置参数的顺序。
*   `RenameKwarg(old="dsize", new="size")`: 键名替换。
*   `DropArg(name)`: 丢弃 GPU 实现不支持的参数。

### 2.3 通用 Output Processors
*   `KeepOnDevice()`: 默认策略，不做操作。
*   `ToNumpy()`: `.cpu().numpy()`，用于对接不支持 Tensor 的下游代码。
*   `BCHW2HWC()`: 还原图像布局。

---

# 3. 流程演示：从简单到困难

以下通过三个具体的例子，展示这个架构如何工作。

## 案例一：简单场景 —— `cv2.resize` (图像缩放)

**难点**：
1.  **布局差异**：OpenCV 是 HWC，PyTorch/Kornia 需要 BCHW。
2.  **参数差异**：OpenCV 接受 `(width, height)`，Kornia/PyTorch 接受 `(height, width)`。

### 策略配置 (Registry Entry)
```python
Strategy(
    domain="CV",
    # 1. 输入处理：第一个参数是图(转Tensor)，第二个是size(透传)
    input_processors=[ImageHWC2BCHW(device='cuda'), PassThrough()],
    kwarg_processors={}, 
    
    # 2. 参数映射：交换 dsize 的值 (W,H -> H,W)
    # 注意：这里需要一个自定义 Mapper，或者在 Backend 中处理
    arg_mapper=CV2ResizeMapper(), 
    
    # 3. 后端：Kornia 的 resize
    backend=kornia.geometry.transform.resize,
    
    # 4. 输出：留在 GPU
    output_processor=KeepOnDevice()
)
```

### 执行流程 (Runtime Flow)
假设调用：`cv2.resize(img_np, (256, 128))` (W=256, H=128)

1.  **Input Processing**:
    *   `Arg[0] (img_np)` -> `ImageHWC2BCHW` -> `img_tensor` (1, C, 128, 256) on GPU.
    *   `Arg[1] ((256, 128))` -> `PassThrough` -> `(256, 128)`.
2.  **Arg Mapping (CV2ResizeMapper)**:
    *   接收 `args=(img_tensor, (256, 128))`。
    *   逻辑：读取第二个参数，反转元组。
    *   输出 `new_args=(img_tensor, (128, 256))`.
3.  **Backend Execution**:
    *   调用 `kornia.geometry.transform.resize(img_tensor, (128, 256))`.
4.  **Output**:
    *   返回 `(1, C, 128, 256)` 的 GPU Tensor。

---

## 案例二：困难场景 A (参数复杂) —— `librosa.feature.melspectrogram`

**难点**：
1.  **默认值黑盒**：Librosa 有很多默认参数（如 `pad_mode='reflect'`），而 Torchaudio 的默认值可能不同（`pad_mode='constant'`）。如果不显式对齐，结果不一致。
2.  **实现差异**：Librosa 的逻辑是 `Audio -> STFT -> Power -> MelScale`。Torchaudio 的 `MelSpectrogram` 类封装了这一过程，但参数构造方式不同。

### 策略配置
这里不能简单映射函数，需要一个 **"Factory Backend"**。

```python
# 自定义后端：动态构建 Torchaudio 的 Transform 对象并执行
def torchaudio_melspec_proxy(waveform, sr=22050, n_fft=2048, hop_length=512, ...):
    # 1. 实例化转换器 (这一步通常需要缓存实例以提升性能)
    transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        # 显式修正差异参数
        pad_mode='reflect' if 'pad_mode' not in kwargs else kwargs['pad_mode']
    ).to(waveform.device)
    
    # 2. 执行
    return transformer(waveform)

Strategy(
    domain="Audio",
    # 输入是音频波形数组
    input_processors=[EnsureTensor(device='cuda', dtype=torch.float32)],
    # 映射全部透传，依靠 Proxy 函数处理参数名
    backend=torchaudio_melspec_proxy,
    output_processor=KeepOnDevice()
)
```

### 执行流程
调用：`librosa.feature.melspectrogram(y=wav_np, sr=16000, n_mels=128)`

1.  **Input**: `wav_np` -> `wav_tensor` (GPU).
2.  **Backend**: 
    *   进入 `torchaudio_melspec_proxy`。
    *   初始化 `MelSpectrogram(sample_rate=16000, n_mels=128, ...)`。
    *   **关键点**：代理函数内部必须处理 Librosa 参数名到 Torchaudio 参数名的转换（如果名字不同）。
    *   执行计算。
3.  **Output**: GPU Spectrogram Tensor.

---

## 案例三：困难场景 B (管道转换) —— `albumentations.Compose`

**难点**：
1.  **不是单一函数**：`Compose` 接收的是一个变换列表 `[Resize(), Normalize(), ...]`。
2.  **结构转换**：我们需要遍历这个列表，把每一个 Albumentations 对象转换成对应的 Kornia/TorchVision 对象，并组装成一个新的 GPU 管道。

### 策略配置
这是一个**对象级（Object-Level）**的迁移，通常发生在初始化阶段，或者在第一次运行时动态编译。

```python
class AlbumentationsConverter:
    def __call__(self, augmentations_list, *args, **kwargs):
        # 这是一个“编译”过程
        gpu_transforms = []
        for aug in augmentations_list:
            if isinstance(aug, A.Resize):
                gpu_transforms.append(kornia.geometry.resize(..., size=(aug.height, aug.width)))
            elif isinstance(aug, A.Normalize):
                gpu_transforms.append(kornia.enhance.Normalize(mean=aug.mean, std=aug.std))
            # ... 更多映射
        
        # 返回一个可执行的 GPU 容器
        return kornia.augmentation.container.Sequential(*gpu_transforms)

Strategy(
    domain="CV_Pipeline",
    # 这里是对 Compose 构造函数的 Patch，或者是对 Compose 实例 __call__ 的 Patch
    # 假设我们 Hook 也就是拦截了 pipeline(image=img) 这一调用
    input_processors=[KwargProcessor("image", ImageHWC2BCHW('cuda'))],
    
    # Backend 负责将原有的 CPU Pipeline 逻辑“偷梁换柱”
    backend=KorniaPipelineExecutor, 
    
    output_processor=KeepOnDevice()
)
```

### 执行流程
用户代码：
```python
transform = A.Compose([A.Resize(256, 256), A.Normalize()])
data = transform(image=img_np) # Hook 这里
```

1.  **Wrapper 拦截**: 获取到 `self` (即 A.Compose 实例) 和 `kwargs={'image': img_np}`。
2.  **Instance Check**: Wrapper 发现这是一个 `A.Compose` 实例。
3.  **JIT Compilation (Lazy)**: 
    *   第一次运行时，Wrapper 检查 `self` 内部的 `transforms` 列表。
    *   调用转换器，生成等价的 `kornia_sequential` 对象，并**缓存**在 Wrapper 或 Context 中。
4.  **Data Move**: `img_np` -> `img_tensor` (GPU).
5.  **Execution**: 调用缓存好的 `kornia_sequential(img_tensor)`.
6.  **Output**: 返回字典 `{'image': result_tensor}` (保持与 Albumentations 接口一致)。

### 总结

这个架构的核心优势在于**分层治理**：
1.  **简单层**：`InputProcessor` 处理数据搬运。
2.  **逻辑层**：`ArgMapper` 处理参数对齐。
3.  **实现层**：`Backend` 处理具体计算。
4.  **复杂的管道逻辑**（如案例三）被封装在特殊的 Backend 或 Converter 中，Wrapper 依然保持“哑”和通用。