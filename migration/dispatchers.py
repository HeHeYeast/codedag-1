"""
动态后端分发器模块 (Dynamic Backend Dispatchers)

处理"源库用一个函数名处理多种逻辑，而目标库拆分为多个函数"的情况

主要场景：
- cv2.cvtColor: 根据 color code 选择不同的 kornia 函数
- PIL.Image.transpose: 根据 method 选择 hflip/vflip/rotate
- cv2.threshold: 根据 type 选择不同的实现
"""

import logging
from typing import Any, Optional, Union, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# 1. OpenCV Dispatchers
# =============================================================================

class CV2Dispatchers:
    """OpenCV 函数动态分发器"""

    # cv2.cvtColor color code -> kornia 函数映射
    COLOR_FUNC_MAP = {
        "bgr_to_rgb": "bgr_to_rgb",
        "rgb_to_bgr": "rgb_to_bgr",
        "bgr_to_grayscale": "bgr_to_grayscale",
        "rgb_to_grayscale": "rgb_to_grayscale",
        "grayscale_to_rgb": "grayscale_to_rgb",
        "bgr_to_hsv": "bgr_to_hsv",
        "hsv_to_bgr": "hsv_to_bgr",
        "rgb_to_hsv": "rgb_to_hsv",
        "hsv_to_rgb": "hsv_to_rgb",
        "bgr_to_lab": "bgr_to_lab",
        "lab_to_bgr": "lab_to_bgr",
        "bgr_to_yuv": "bgr_to_yuv",
        "yuv_to_bgr": "yuv_to_bgr",
    }

    # cv2.threshold type 常量值
    # THRESH_BINARY = 0, THRESH_BINARY_INV = 1, THRESH_TRUNC = 2
    # THRESH_TOZERO = 3, THRESH_TOZERO_INV = 4
    # THRESH_OTSU = 8, THRESH_TRIANGLE = 16 (不支持)
    THRESHOLD_TYPES = {
        0: "binary",
        1: "binary_inv",
        2: "trunc",
        3: "tozero",
        4: "tozero_inv",
    }

    @staticmethod
    def cvtcolor(img: Any, code: Optional[str] = None, **kwargs) -> Any:
        """
        cv2.cvtColor 动态分发器

        根据 color code 选择对应的 kornia 函数并执行

        Args:
            img: 输入图像 (已转换为 BCHW Tensor)
            code: 转换后的 color code 字符串 (由 cv2_color_code mapper 处理)

        Returns:
            转换后的图像 Tensor
        """
        if code is None:
            raise ValueError("cv2.cvtColor: missing color code")

        func_name = CV2Dispatchers.COLOR_FUNC_MAP.get(code)

        if func_name is None:
            raise NotImplementedError(
                f"cv2.cvtColor: unsupported color code '{code}'. "
                f"Supported: {list(CV2Dispatchers.COLOR_FUNC_MAP.keys())}"
            )

        try:
            import kornia.color
            func = getattr(kornia.color, func_name, None)
            if func is None:
                raise AttributeError(f"kornia.color has no function '{func_name}'")
            return func(img)
        except ImportError:
            raise ImportError("kornia is required for cv2.cvtColor migration")

    @staticmethod
    def threshold(img: Any, thresh: float = 0, maxval: float = 255,
                  type: int = 0, **kwargs) -> Tuple[float, Any]:
        """
        cv2.threshold 动态分发器

        根据 threshold type 使用不同的 PyTorch 实现

        注意：
        - 不支持 THRESH_OTSU (8) 和 THRESH_TRIANGLE (16)
        - 对于不支持的类型，始终返回输入的 thresh 作为 retval

        Args:
            img: 输入图像 (已转换为 BCHW Tensor)
            thresh: 阈值
            maxval: 最大值 (用于 BINARY 类型)
            type: 阈值类型 (int)，仅支持 0-4

        Returns:
            (retval, dst) 元组，与 cv2.threshold 返回格式一致
            retval: 输入的阈值（不支持 OTSU/TRIANGLE 自动计算）
            dst: 处理后的图像 Tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for cv2.threshold migration")

        # 检查是否使用了不支持的 flag
        # OTSU=8, TRIANGLE=16，可能与基本类型组合使用
        base_type = type & 0x07  # 取低3位
        if type & 0x08:  # THRESH_OTSU
            logger.warning("cv2.threshold: THRESH_OTSU 不支持，将忽略此 flag")
        if type & 0x10:  # THRESH_TRIANGLE
            logger.warning("cv2.threshold: THRESH_TRIANGLE 不支持，将忽略此 flag")

        type_name = CV2Dispatchers.THRESHOLD_TYPES.get(base_type, "binary")

        if type_name == "binary":
            # dst = maxval if src > thresh else 0
            dst = torch.where(img > thresh, maxval, 0.0)
        elif type_name == "binary_inv":
            # dst = 0 if src > thresh else maxval
            dst = torch.where(img > thresh, 0.0, maxval)
        elif type_name == "trunc":
            # dst = thresh if src > thresh else src
            dst = torch.where(img > thresh, thresh, img)
        elif type_name == "tozero":
            # dst = src if src > thresh else 0
            dst = torch.where(img > thresh, img, 0.0)
        elif type_name == "tozero_inv":
            # dst = 0 if src > thresh else src
            dst = torch.where(img > thresh, 0.0, img)
        else:
            raise NotImplementedError(
                f"cv2.threshold: unsupported type {type} ('{type_name}')"
            )

        # cv2.threshold 返回 (retval, dst)
        # 由于不支持 OTSU/TRIANGLE，retval 始终等于输入的 thresh
        return thresh, dst


# =============================================================================
# 2. PIL Dispatchers
# =============================================================================

class PILDispatchers:
    """PIL 函数动态分发器"""

    @staticmethod
    def transpose(img: Any, method: Optional[str] = None, **kwargs) -> Any:
        """
        PIL.Image.transpose 动态分发器

        根据 method 选择 hflip/vflip/rotate

        Args:
            img: 输入图像 (已转换为 Tensor)
            method: 转换后的 method 字符串 (由 pil_flip_method mapper 处理)

        Returns:
            变换后的图像 Tensor
        """
        if method is None:
            raise ValueError("PIL.Image.transpose: missing method")

        try:
            from torchvision.transforms.v2 import functional as F
        except ImportError:
            try:
                from torchvision.transforms import functional as F
            except ImportError:
                raise ImportError("torchvision is required for PIL.Image.transpose migration")

        if method == "hflip":
            return F.hflip(img)
        elif method == "vflip":
            return F.vflip(img)
        elif method == "rotate_90":
            return F.rotate(img, 90)
        elif method == "rotate_180":
            return F.rotate(img, 180)
        elif method == "rotate_270":
            return F.rotate(img, 270)
        else:
            raise NotImplementedError(
                f"PIL.Image.transpose: unsupported method '{method}'. "
                "Supported: hflip, vflip, rotate_90, rotate_180, rotate_270"
            )


# =============================================================================
# 3. NumPy Random Dispatchers
# =============================================================================

def _normalize_size(size: Any) -> Tuple[tuple, int, bool]:
    """
    规范化 size 参数

    Args:
        size: None, int, list, 或 tuple

    Returns:
        (shape_tuple, n_elements, should_squeeze)
    """
    if size is None:
        return (), 1, True
    elif isinstance(size, int):
        return (size,), size, False
    else:
        # list 或 tuple
        shape = tuple(size)
        n_elements = 1
        for s in shape:
            n_elements *= s
        return shape, n_elements, False


class NumpyRandomDispatchers:
    """NumPy 随机函数分发器"""

    @staticmethod
    def choice(a: Any, size: Any = None, replace: bool = True,
               p: Any = None, device: str = "cuda:0", **kwargs) -> Any:
        """
        np.random.choice 分发器

        Args:
            a: 如果是整数，从 arange(a) 中选择；如果是数组，从中选择
            size: 输出 shape (None, int, list, 或 tuple)
            replace: 是否有放回采样
            p: 概率分布
            device: 目标设备

        Returns:
            采样结果 Tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for np.random.choice migration")

        # 处理 a 参数
        if isinstance(a, int):
            population = torch.arange(a, device=device, dtype=torch.long)
        else:
            population = torch.as_tensor(a, device=device)

        n_pop = len(population)

        # 规范化 size 参数
        shape, n_samples, squeeze = _normalize_size(size)

        # 检查无放回采样的约束
        if not replace and n_samples > n_pop:
            raise ValueError(
                f"Cannot take a larger sample ({n_samples}) than "
                f"population ({n_pop}) when replace=False"
            )

        # 采样
        if p is not None:
            p_tensor = torch.as_tensor(p, device=device, dtype=torch.float32)
            indices = torch.multinomial(p_tensor, n_samples, replacement=replace)
        else:
            if replace:
                indices = torch.randint(0, n_pop, (n_samples,), device=device)
            else:
                perm = torch.randperm(n_pop, device=device)
                indices = perm[:n_samples]

        result = population[indices]

        # 调整形状
        if shape and len(shape) > 1:
            result = result.reshape(shape)
        elif squeeze:
            result = result.squeeze()

        return result

    @staticmethod
    def normal(loc: float = 0.0, scale: float = 1.0, size: Any = None,
               device: str = "cuda:0", **kwargs) -> Any:
        """
        np.random.normal 分发器

        Args:
            loc: 均值
            scale: 标准差
            size: 输出 shape (None, int, list, 或 tuple)
            device: 目标设备

        Returns:
            采样结果 Tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for np.random.normal migration")

        shape, _, squeeze = _normalize_size(size)

        if not shape:
            # size=None，返回标量
            result = torch.normal(loc, scale, size=(1,), device=device)
            return result.squeeze() if squeeze else result
        else:
            return torch.normal(loc, scale, size=shape, device=device)

    @staticmethod
    def uniform(low: float = 0.0, high: float = 1.0, size: Any = None,
                device: str = "cuda:0", **kwargs) -> Any:
        """
        np.random.uniform 分发器

        Args:
            low: 下界
            high: 上界
            size: 输出 shape (None, int, list, 或 tuple)
            device: 目标设备

        Returns:
            采样结果 Tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for np.random.uniform migration")

        shape, _, squeeze = _normalize_size(size)

        if not shape:
            # size=None，返回标量
            result = torch.rand(1, device=device) * (high - low) + low
            return result.squeeze() if squeeze else result
        else:
            return torch.rand(shape, device=device) * (high - low) + low


# =============================================================================
# 4. Dispatcher Registry
# =============================================================================

class DispatcherRegistry:
    """
    分发器注册表

    提供统一的分发器查找接口
    """

    DISPATCHERS = {
        # OpenCV
        "cv2.cvtColor": CV2Dispatchers.cvtcolor,
        "cv2.threshold": CV2Dispatchers.threshold,
        # PIL
        "PIL.Image.Image.transpose": PILDispatchers.transpose,
        # NumPy Random
        "numpy.random.choice": NumpyRandomDispatchers.choice,
        "numpy.random.normal": NumpyRandomDispatchers.normal,
        "numpy.random.uniform": NumpyRandomDispatchers.uniform,
    }

    @classmethod
    def get(cls, func_path: str):
        """
        获取分发器函数

        Args:
            func_path: 函数路径，如 "cv2.cvtColor"

        Returns:
            分发器函数或 None
        """
        return cls.DISPATCHERS.get(func_path)

    @classmethod
    def has_dispatcher(cls, func_path: str) -> bool:
        """检查是否有对应的分发器"""
        return func_path in cls.DISPATCHERS

    @classmethod
    def register(cls, func_path: str, dispatcher):
        """注册自定义分发器"""
        cls.DISPATCHERS[func_path] = dispatcher

    @classmethod
    def all_dispatchers(cls):
        """获取所有已注册的分发器路径"""
        return list(cls.DISPATCHERS.keys())
