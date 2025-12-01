"""
迁移处理器模块 (Migration Processors)
定义所有原子处理器：输入处理器、值映射器、输出处理器、结构性操作
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Input Processors (输入数据转换)
# =============================================================================

class InputProcessors:
    """输入数据处理器集合"""

    @staticmethod
    def ensure_tensor(x: Any, device: str) -> Any:
        """
        通用 Tensor 转换
        - Tensor: to(device)
        - Numpy/List: torch.as_tensor(x).to(device)
        """
        if not TORCH_AVAILABLE:
            return x

        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            return torch.as_tensor(x).to(device)
        elif isinstance(x, (list, tuple)):
            return torch.as_tensor(x).to(device)
        else:
            try:
                return torch.as_tensor(x).to(device)
            except Exception:
                return x

    @staticmethod
    def to_tensor_float(x: Any, device: str) -> Any:
        """转为 float32 Tensor"""
        if not TORCH_AVAILABLE:
            return x

        result = InputProcessors.ensure_tensor(x, device)
        if isinstance(result, torch.Tensor):
            return result.float()
        return result

    @staticmethod
    def to_tensor_long(x: Any, device: str) -> Any:
        """转为 int64 Tensor (用于索引、标签)"""
        if not TORCH_AVAILABLE:
            return x

        result = InputProcessors.ensure_tensor(x, device)
        if isinstance(result, torch.Tensor):
            return result.long()
        return result

    @staticmethod
    def list_to_tensor_stack(x: Any, device: str) -> Any:
        """
        List[Array] -> List[Tensor] 并搬运到 GPU
        用于 np.stack, torch.cat 等操作
        """
        if not TORCH_AVAILABLE:
            return x

        if not isinstance(x, (list, tuple)):
            return InputProcessors.ensure_tensor(x, device)

        result = []
        for item in x:
            if isinstance(item, np.ndarray):
                result.append(torch.from_numpy(item).to(device))
            elif isinstance(item, torch.Tensor):
                result.append(item.to(device))
            else:
                result.append(torch.as_tensor(item).to(device))
        return result

    # ===================== CV 专用 =====================

    @staticmethod
    def cv2_hwc_to_bchw(x: Any, device: str) -> Any:
        """
        OpenCV HWC 格式转 PyTorch BCHW 格式
        from_numpy(HWC) -> permute(2,0,1) -> unsqueeze(0) -> .to(device).float()
        """
        if not TORCH_AVAILABLE:
            return x

        if isinstance(x, np.ndarray):
            if x.ndim == 3:  # HWC
                tensor = torch.from_numpy(x.copy()).permute(2, 0, 1).unsqueeze(0)
            elif x.ndim == 2:  # HW (灰度图)
                tensor = torch.from_numpy(x.copy()).unsqueeze(0).unsqueeze(0)
            else:
                tensor = torch.from_numpy(x.copy())
            return tensor.to(device).float()
        elif isinstance(x, torch.Tensor):
            return x.to(device).float()
        return x

    @staticmethod
    def cv2_hwc_to_chw(x: Any, device: str) -> Any:
        """
        OpenCV HWC 格式转 CHW (无 batch)
        from_numpy(HWC) -> permute(2,0,1) -> .to(device).float()
        """
        if not TORCH_AVAILABLE:
            return x

        if isinstance(x, np.ndarray):
            if x.ndim == 3:  # HWC
                tensor = torch.from_numpy(x.copy()).permute(2, 0, 1)
            elif x.ndim == 2:  # HW
                tensor = torch.from_numpy(x.copy()).unsqueeze(0)
            else:
                tensor = torch.from_numpy(x.copy())
            return tensor.to(device).float()
        elif isinstance(x, torch.Tensor):
            return x.to(device).float()
        return x

    @staticmethod
    def pil_to_tensor(x: Any, device: str) -> Any:
        """
        PIL Image 转 Tensor
        使用 torchvision.transforms.functional.to_tensor 保证返回 float [0, 1]
        """
        if not TORCH_AVAILABLE:
            return x

        try:
            from PIL import Image
            if isinstance(x, Image.Image):
                # 使用 to_tensor 保证返回 float [0, 1] 的一致性行为
                # 这对 Kornia/PyTorch 后续计算操作更友好
                from torchvision.transforms.functional import to_tensor
                return to_tensor(x).to(device)
        except ImportError:
            pass

        return InputProcessors.ensure_tensor(x, device)

    @staticmethod
    def bbox_list_to_tensor(x: Any, device: str) -> Any:
        """
        Bounding box 列表转 Tensor
        [[x,y,w,h], ...] -> Tensor
        """
        if not TORCH_AVAILABLE:
            return x

        if isinstance(x, (list, tuple)) and len(x) > 0:
            if isinstance(x[0], (list, tuple)):
                return torch.tensor(x, dtype=torch.float32, device=device)

        return InputProcessors.ensure_tensor(x, device)

    @staticmethod
    def mask_to_tensor(x: Any, device: str) -> Any:
        """
        Mask array 转 Tensor
        支持 bool 或 long 类型
        """
        if not TORCH_AVAILABLE:
            return x

        if isinstance(x, np.ndarray):
            if x.dtype == np.bool_:
                return torch.from_numpy(x.copy()).bool().to(device)
            else:
                return torch.from_numpy(x.copy()).long().to(device)

        return InputProcessors.ensure_tensor(x, device)

    # ===================== Audio 专用 =====================

    @staticmethod
    def audio_to_tensor(x: Any, device: str) -> Any:
        """
        音频数据转 Tensor
        from_numpy -> .float() -> .to(device) -> unsqueeze(0) (添加 batch)
        """
        if not TORCH_AVAILABLE:
            return x

        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x.copy()).float().to(device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor
        elif isinstance(x, torch.Tensor):
            tensor = x.float().to(device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor

        return x

    @staticmethod
    def audio_to_tensor_no_batch(x: Any, device: str) -> Any:
        """
        音频数据转 Tensor (不添加 batch)
        from_numpy -> .float() -> .to(device)
        """
        if not TORCH_AVAILABLE:
            return x

        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.copy()).float().to(device)
        elif isinstance(x, torch.Tensor):
            return x.float().to(device)

        return x

    # ===================== 直通与忽略 =====================

    @staticmethod
    def pass_through(x: Any, device: str) -> Any:
        """直接透传，不做任何处理"""
        return x

    @staticmethod
    def to_device(x: Any, device: str) -> Any:
        """仅搬运到设备 (已是 Tensor 的情况)"""
        if not TORCH_AVAILABLE:
            return x

        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x


# =============================================================================
# 2. Arg Value Mappers (参数值变换)
# =============================================================================

class ValueMappers:
    """参数值变换器集合"""

    # ===================== 几何参数 =====================

    @staticmethod
    def swap_hw(value: Any) -> Any:
        """
        交换 (W, H) 为 (H, W)
        用于 cv2.resize dsize 参数
        """
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (value[1], value[0])
        return value

    @staticmethod
    def xy_to_yx(value: Any) -> Any:
        """坐标点变换 (x, y) -> (y, x)"""
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (value[1], value[0])
        return value

    @staticmethod
    def ensure_tuple(value: Any) -> Any:
        """
        确保是 tuple 类型
        int 256 -> (256, 256)
        """
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, list):
            return tuple(value)
        return value

    @staticmethod
    def ensure_list(value: Any) -> Any:
        """确保是 list 类型"""
        if isinstance(value, tuple):
            return list(value)
        elif not isinstance(value, list):
            return [value]
        return value

    @staticmethod
    def to_float(value: Any) -> float:
        """强制转 float"""
        return float(value)

    @staticmethod
    def to_int(value: Any) -> int:
        """强制转 int"""
        return int(value)

    # ===================== OpenCV 常量映射 =====================

    @staticmethod
    def cv2_flip_code(code: int) -> List[int]:
        """
        cv2.flip flipCode 转 torch.flip dims
        0 (垂直翻转) -> [-2]
        1 (水平翻转) -> [-1]
        -1 (双向翻转) -> [-2, -1]
        """
        if code == 0:
            return [-2]
        elif code == 1:
            return [-1]
        elif code == -1:
            return [-2, -1]
        return [-1]

    @staticmethod
    def cv2_border_type(border_type: int) -> str:
        """cv2 border type 转 torch pad mode"""
        try:
            import cv2
            border_map = {
                cv2.BORDER_CONSTANT: "constant",
                cv2.BORDER_REPLICATE: "replicate",
                cv2.BORDER_REFLECT: "reflect",
                cv2.BORDER_REFLECT_101: "reflect",
                cv2.BORDER_WRAP: "circular",
            }
            return border_map.get(border_type, "constant")
        except ImportError:
            border_map = {
                0: "constant",
                1: "replicate",
                2: "reflect",
                4: "reflect",
                3: "circular",
            }
            return border_map.get(border_type, "constant")

    @staticmethod
    def cv2_interp_mode(interp: int) -> str:
        """cv2 interpolation mode 转 kornia/torch 模式"""
        try:
            import cv2
            interp_map = {
                cv2.INTER_NEAREST: "nearest",
                cv2.INTER_LINEAR: "bilinear",
                cv2.INTER_AREA: "area",
                cv2.INTER_CUBIC: "bicubic",
                cv2.INTER_LANCZOS4: "bicubic",
            }
            return interp_map.get(interp, "bilinear")
        except ImportError:
            interp_map = {
                0: "nearest",
                1: "bilinear",
                2: "bicubic",
                3: "area",
                4: "bicubic",
            }
            return interp_map.get(interp, "bilinear")

    @staticmethod
    def cv2_color_code(code: int) -> Optional[str]:
        """
        cv2 COLOR_* 常量转 kornia 函数名
        返回 None 表示不支持，需要 fallback
        """
        try:
            import cv2
            color_map = {
                cv2.COLOR_BGR2RGB: "bgr_to_rgb",
                cv2.COLOR_RGB2BGR: "rgb_to_bgr",
                cv2.COLOR_BGR2GRAY: "bgr_to_grayscale",
                cv2.COLOR_RGB2GRAY: "rgb_to_grayscale",
                cv2.COLOR_GRAY2BGR: "grayscale_to_rgb",
                cv2.COLOR_GRAY2RGB: "grayscale_to_rgb",
                cv2.COLOR_BGR2HSV: "bgr_to_hsv",
                cv2.COLOR_HSV2BGR: "hsv_to_bgr",
                cv2.COLOR_RGB2HSV: "rgb_to_hsv",
                cv2.COLOR_HSV2RGB: "hsv_to_rgb",
                cv2.COLOR_BGR2LAB: "bgr_to_lab",
                cv2.COLOR_LAB2BGR: "lab_to_bgr",
                cv2.COLOR_BGR2YUV: "bgr_to_yuv",
                cv2.COLOR_YUV2BGR: "yuv_to_bgr",
            }
            return color_map.get(code)
        except ImportError:
            color_map = {
                4: "bgr_to_rgb",
                2: "rgb_to_bgr",
                6: "bgr_to_grayscale",
                7: "rgb_to_grayscale",
                8: "grayscale_to_rgb",
                40: "bgr_to_hsv",
                54: "hsv_to_bgr",
            }
            return color_map.get(code)

    # ===================== PIL 常量映射 =====================

    @staticmethod
    def pil_interp_mode(resample: Any) -> Any:
        """PIL resample 模式转 torchvision InterpolationMode"""
        if not TORCH_AVAILABLE:
            return resample

        try:
            from PIL import Image
            from torchvision.transforms import InterpolationMode

            pil_map = {
                Image.NEAREST: InterpolationMode.NEAREST,
                Image.BILINEAR: InterpolationMode.BILINEAR,
                Image.BICUBIC: InterpolationMode.BICUBIC,
                Image.LANCZOS: InterpolationMode.LANCZOS,
                Image.HAMMING: InterpolationMode.HAMMING,
                Image.BOX: InterpolationMode.BOX,
            }
            if isinstance(resample, int):
                int_map = {
                    0: InterpolationMode.NEAREST,
                    1: InterpolationMode.LANCZOS,
                    2: InterpolationMode.BILINEAR,
                    3: InterpolationMode.BICUBIC,
                    4: InterpolationMode.BOX,
                    5: InterpolationMode.HAMMING,
                }
                return int_map.get(resample, InterpolationMode.BILINEAR)

            return pil_map.get(resample, InterpolationMode.BILINEAR)
        except ImportError:
            return resample

    @staticmethod
    def pil_flip_method(method: Any) -> str:
        """PIL transpose method 转操作类型"""
        try:
            from PIL import Image
            if method == Image.FLIP_LEFT_RIGHT:
                return "hflip"
            elif method == Image.FLIP_TOP_BOTTOM:
                return "vflip"
            elif method == Image.ROTATE_90:
                return "rotate_90"
            elif method == Image.ROTATE_180:
                return "rotate_180"
            elif method == Image.ROTATE_270:
                return "rotate_270"
        except ImportError:
            pass

        method_map = {
            0: "hflip",
            1: "vflip",
            2: "rotate_90",
            3: "rotate_180",
            4: "rotate_270",
        }
        return method_map.get(method, "hflip")

    # ===================== Audio 参数映射 =====================

    @staticmethod
    def librosa_pad_mode(mode: str) -> str:
        """librosa pad_mode 转 torchaudio 兼容模式"""
        mode_map = {
            "reflect": "reflect",
            "constant": "constant",
            "edge": "replicate",
            "wrap": "circular",
        }
        return mode_map.get(mode, mode)


# =============================================================================
# 3. Output Processors (输出结果处理)
# =============================================================================

class OutputProcessors:
    """输出结果处理器集合"""

    @staticmethod
    def keep_on_device(x: Any) -> Any:
        """保持在 GPU，不做处理"""
        return x

    @staticmethod
    def to_numpy(x: Any) -> Any:
        """转回 NumPy 数组"""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    @staticmethod
    def to_numpy_uint8(x: Any) -> Any:
        """转 NumPy uint8 (图像输出)"""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            if arr.dtype in (np.float32, np.float64):
                arr = (arr * 255).clip(0, 255)
            return arr.astype(np.uint8)
        return x

    @staticmethod
    def bchw_to_hwc(x: Any) -> Any:
        """BCHW 转 HWC (对接 CPU CV 库)"""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            if x.ndim == 4:  # BCHW
                return x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            elif x.ndim == 3:  # CHW
                return x.permute(1, 2, 0).detach().cpu().numpy()
        return x

    @staticmethod
    def bchw_to_hwc_uint8(x: Any) -> Any:
        """BCHW 转 HWC uint8 (OpenCV 格式)"""
        result = OutputProcessors.bchw_to_hwc(x)
        if isinstance(result, np.ndarray):
            if result.dtype in (np.float32, np.float64):
                result = (result * 255).clip(0, 255)
            return result.astype(np.uint8)
        return result

    @staticmethod
    def chw_to_hwc(x: Any) -> Any:
        """CHW 转 HWC (无 batch)"""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            if x.ndim == 3:  # CHW
                return x.permute(1, 2, 0).detach().cpu().numpy()
        return x

    @staticmethod
    def to_pil(x: Any) -> Any:
        """Tensor 转 PIL Image"""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            try:
                from torchvision.transforms.functional import to_pil_image
                # detach 并移到 CPU
                x = x.detach().cpu()
                # 移除 batch 维度
                if x.ndim == 4:
                    x = x.squeeze(0)
                return to_pil_image(x)
            except ImportError:
                pass
        return x

    @staticmethod
    def squeeze_batch(x: Any) -> Any:
        """移除 batch 维度"""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            if x.ndim >= 1 and x.shape[0] == 1:
                return x.squeeze(0)
        return x


# =============================================================================
# 4. Structural Operations (结构性操作)
# =============================================================================

class StructuralOps:
    """
    结构性操作集合

    注意：所有操作的签名必须是 (args: tuple, kwargs: dict) -> Tuple[tuple, dict]
    不支持需要额外元数据的操作（如 args_to_kwargs 需要 param_names）
    """

    @staticmethod
    def swap_args_0_1(args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """交换位置参数 0 和 1"""
        if len(args) >= 2:
            args = (args[1], args[0]) + args[2:]
        return args, kwargs

    @staticmethod
    def swap_args_1_2(args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """交换位置参数 1 和 2"""
        if len(args) >= 3:
            args = (args[0], args[2], args[1]) + args[3:]
        return args, kwargs

    @staticmethod
    def flatten_list_arg_0(args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """展开第一个参数 (如果是列表)"""
        if len(args) >= 1 and isinstance(args[0], (list, tuple)):
            return tuple(args[0]), kwargs
        return args, kwargs

    @staticmethod
    def cv2_border_args_to_pad(args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """
        cv2.copyMakeBorder 参数转 torch.nn.functional.pad 参数
        cv2: (src, top, bottom, left, right, borderType)
        torch: (input, pad=(left, right, top, bottom), mode)
        """
        if len(args) >= 5:
            src = args[0]
            top, bottom, left, right = args[1], args[2], args[3], args[4]
            pad_tuple = (left, right, top, bottom)
            new_kwargs = kwargs.copy()
            new_kwargs['pad'] = pad_tuple
            return (src,), new_kwargs
        return args, kwargs

    @staticmethod
    def pil_box_to_crop_params(args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """
        PIL crop box 转 torchvision crop 参数
        PIL: Image.crop(box) where box=(left, top, right, bottom)
        torchvision: F.crop(img, top, left, height, width)

        args[0] 是 img (self)
        args[1] 是 box，或者 box 在 kwargs 中
        """
        img = args[0] if len(args) >= 1 else None
        box = None

        if len(args) >= 2:
            box = args[1]
        elif 'box' in kwargs:
            box = kwargs.pop('box')

        if img is not None and box is not None:
            left, top, right, bottom = box
            height = bottom - top
            width = right - left
            # F.crop 签名: (img, top, left, height, width)
            return (img, top, left, height, width), kwargs

        return args, kwargs


# =============================================================================
# 5. Processor Registry (处理器注册表)
# =============================================================================

class ProcessorRegistry:
    """处理器注册表 - 提供统一的处理器查找接口"""

    # 输入处理器映射
    INPUT_PROCESSORS = {
        "ensure_tensor": InputProcessors.ensure_tensor,
        "to_tensor_float": InputProcessors.to_tensor_float,
        "to_tensor_long": InputProcessors.to_tensor_long,
        "list_to_tensor_stack": InputProcessors.list_to_tensor_stack,
        "cv2_hwc_to_bchw": InputProcessors.cv2_hwc_to_bchw,
        "cv2_hwc_to_chw": InputProcessors.cv2_hwc_to_chw,
        "pil_to_tensor": InputProcessors.pil_to_tensor,
        "bbox_list_to_tensor": InputProcessors.bbox_list_to_tensor,
        "mask_to_tensor": InputProcessors.mask_to_tensor,
        "audio_to_tensor": InputProcessors.audio_to_tensor,
        "audio_to_tensor_no_batch": InputProcessors.audio_to_tensor_no_batch,
        "pass": InputProcessors.pass_through,
        "to_device": InputProcessors.to_device,
    }

    # 值映射器映射
    VALUE_MAPPERS = {
        "swap_hw": ValueMappers.swap_hw,
        "xy_to_yx": ValueMappers.xy_to_yx,
        "ensure_tuple": ValueMappers.ensure_tuple,
        "ensure_list": ValueMappers.ensure_list,
        "to_float": ValueMappers.to_float,
        "to_int": ValueMappers.to_int,
        "cv2_flip_code": ValueMappers.cv2_flip_code,
        "cv2_border_type": ValueMappers.cv2_border_type,
        "cv2_interp_mode": ValueMappers.cv2_interp_mode,
        "cv2_color_code": ValueMappers.cv2_color_code,
        "pil_interp_mode": ValueMappers.pil_interp_mode,
        "pil_flip_method": ValueMappers.pil_flip_method,
        "librosa_pad_mode": ValueMappers.librosa_pad_mode,
    }

    # 输出处理器映射
    OUTPUT_PROCESSORS = {
        "keep_on_device": OutputProcessors.keep_on_device,
        "to_numpy": OutputProcessors.to_numpy,
        "to_numpy_uint8": OutputProcessors.to_numpy_uint8,
        "bchw_to_hwc": OutputProcessors.bchw_to_hwc,
        "bchw_to_hwc_uint8": OutputProcessors.bchw_to_hwc_uint8,
        "chw_to_hwc": OutputProcessors.chw_to_hwc,
        "to_pil": OutputProcessors.to_pil,
        "squeeze_batch": OutputProcessors.squeeze_batch,
    }

    # 结构性操作映射
    STRUCTURAL_OPS = {
        "swap_args_0_1": StructuralOps.swap_args_0_1,
        "swap_args_1_2": StructuralOps.swap_args_1_2,
        "flatten_list_arg_0": StructuralOps.flatten_list_arg_0,
        "cv2_border_args_to_pad": StructuralOps.cv2_border_args_to_pad,
        "pil_box_to_crop_params": StructuralOps.pil_box_to_crop_params,
    }

    @classmethod
    def get_input_processor(cls, name: str):
        """获取输入处理器"""
        return cls.INPUT_PROCESSORS.get(name)

    @classmethod
    def get_value_mapper(cls, name: str):
        """获取值映射器"""
        return cls.VALUE_MAPPERS.get(name)

    @classmethod
    def get_output_processor(cls, name: str):
        """获取输出处理器"""
        return cls.OUTPUT_PROCESSORS.get(name)

    @classmethod
    def get_structural_op(cls, name: str):
        """获取结构性操作"""
        return cls.STRUCTURAL_OPS.get(name)

    @classmethod
    def apply_input_processor(cls, name: str, value: Any, device: str) -> Any:
        """应用输入处理器"""
        processor = cls.get_input_processor(name)
        if processor:
            return processor(value, device)
        logger.warning(f"未知的输入处理器: {name}")
        return value

    @classmethod
    def apply_value_mapper(cls, name: str, value: Any) -> Any:
        """应用值映射器"""
        mapper = cls.get_value_mapper(name)
        if mapper:
            return mapper(value)
        logger.warning(f"未知的值映射器: {name}")
        return value

    @classmethod
    def apply_output_processor(cls, name: str, value: Any) -> Any:
        """应用输出处理器"""
        processor = cls.get_output_processor(name)
        if processor:
            return processor(value)
        logger.warning(f"未知的输出处理器: {name}")
        return value

    @classmethod
    def apply_structural_op(cls, name: str, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """应用结构性操作"""
        op = cls.get_structural_op(name)
        if op:
            return op(args, kwargs)
        logger.warning(f"未知的结构性操作: {name}")
        return args, kwargs
