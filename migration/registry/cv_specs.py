"""
计算机视觉函数策略 (CV Specs)

OpenCV -> Kornia/PyTorch 映射
PIL -> TorchVision 映射
"""

from ..core import OpSpec, StrategyType
from .base import DefaultStrategyRegistry


def register_cv_specs(registry: DefaultStrategyRegistry) -> None:
    """注册 CV 相关函数策略"""

    # =====================================================
    # 1. 不可迁移的 I/O 函数
    # =====================================================

    io_funcs = [
        "cv2.imread",
        "cv2.imwrite",
        "cv2.VideoCapture",
        "cv2.VideoWriter",
        "cv2.imdecode",
        "cv2.imencode",
        "PIL.Image.open",
        "PIL.Image.Image.save",
    ]

    for func in io_funcs:
        registry.register(OpSpec(
            source=func,
            strategy=StrategyType.NON_MIGRATABLE,
            priority=0,
            notes="I/O 操作，不可迁移"
        ))

    # =====================================================
    # 2. OpenCV -> Kornia (StandardOp)
    # =====================================================

    # cv2.resize - P1 (最常见的瓶颈)
    registry.register(OpSpec(
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
        notes="最常见的数据管道瓶颈，需交换 (W,H) 为 (H,W)"
    ))

    # cv2.cvtColor - P1 (使用 Dispatcher 动态选择函数)
    # 注意：target_func 为空，由 dispatchers.py 中的 CV2Dispatchers.cvtcolor 处理
    registry.register(OpSpec(
        source="cv2.cvtColor",
        strategy=StrategyType.STANDARD_OP,
        priority=1,
        target_lib="",  # 使用 Dispatcher
        target_func="",  # 使用 Dispatcher
        args_trans={0: "cv2_hwc_to_bchw"},
        arg_value_maps={"code": "cv2_color_code"},
        output_rule="keep_on_device",
        notes="使用 Dispatcher 根据 color code 动态选择 kornia 函数"
    ))

    # cv2.GaussianBlur - P2
    registry.register(OpSpec(
        source="cv2.GaussianBlur",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="kornia.filters",
        target_func="gaussian_blur2d",
        args_trans={0: "cv2_hwc_to_bchw"},
        arg_renames={"ksize": "kernel_size", "sigmaX": "sigma"},
        arg_value_maps={"ksize": "ensure_tuple"},
        output_rule="keep_on_device"
    ))

    # cv2.flip - P2
    registry.register(OpSpec(
        source="cv2.flip",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="flip",
        args_trans={0: "cv2_hwc_to_bchw"},
        arg_renames={"flipCode": "dims"},
        arg_value_maps={"flipCode": "cv2_flip_code"},
        output_rule="keep_on_device"
    ))

    # cv2.normalize - P2
    registry.register(OpSpec(
        source="cv2.normalize",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="kornia.enhance",
        target_func="normalize_min_max",
        args_trans={0: "cv2_hwc_to_bchw"},
        output_rule="keep_on_device",
        notes="语义可能与 cv2 不完全一致"
    ))

    # cv2.threshold - P3 (使用 Dispatcher)
    registry.register(OpSpec(
        source="cv2.threshold",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="",  # 使用 Dispatcher
        target_func="",  # 使用 Dispatcher
        args_trans={0: "cv2_hwc_to_bchw"},
        output_rule="keep_on_device",
        notes="使用 Dispatcher，不支持 OTSU/TRIANGLE"
    ))

    # cv2.warpAffine - P3
    registry.register(OpSpec(
        source="cv2.warpAffine",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="kornia.geometry.transform",
        target_func="warp_affine",
        args_trans={0: "cv2_hwc_to_bchw", 1: "ensure_tensor"},
        arg_renames={"M": "M", "dsize": "dsize"},
        arg_value_maps={"dsize": "swap_hw"},
        output_rule="keep_on_device"
    ))

    # cv2.warpPerspective - P3
    registry.register(OpSpec(
        source="cv2.warpPerspective",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="kornia.geometry.transform",
        target_func="warp_perspective",
        args_trans={0: "cv2_hwc_to_bchw", 1: "ensure_tensor"},
        arg_value_maps={"dsize": "swap_hw"},
        output_rule="keep_on_device"
    ))

    # cv2.copyMakeBorder - P3
    registry.register(OpSpec(
        source="cv2.copyMakeBorder",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="torch.nn.functional",
        target_func="pad",
        args_trans={0: "cv2_hwc_to_bchw"},
        arg_value_maps={"borderType": "cv2_border_type"},
        structural_ops=["cv2_border_args_to_pad"],
        output_rule="keep_on_device"
    ))

    # cv2.medianBlur - P3
    registry.register(OpSpec(
        source="cv2.medianBlur",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="kornia.filters",
        target_func="median_blur",
        args_trans={0: "cv2_hwc_to_bchw"},
        arg_renames={"ksize": "kernel_size"},
        arg_value_maps={"ksize": "ensure_tuple"},
        output_rule="keep_on_device"
    ))

    # cv2.blur - P3
    registry.register(OpSpec(
        source="cv2.blur",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="kornia.filters",
        target_func="box_blur",
        args_trans={0: "cv2_hwc_to_bchw"},
        arg_renames={"ksize": "kernel_size"},
        arg_value_maps={"ksize": "ensure_tuple"},
        output_rule="keep_on_device"
    ))

    # cv2.Canny - P3 (边缘检测)
    registry.register(OpSpec(
        source="cv2.Canny",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="kornia.filters",
        target_func="canny",
        args_trans={0: "cv2_hwc_to_bchw"},
        arg_renames={"threshold1": "low_threshold", "threshold2": "high_threshold"},
        output_rule="keep_on_device"
    ))

    # cv2.Sobel - P3
    registry.register(OpSpec(
        source="cv2.Sobel",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="kornia.filters",
        target_func="sobel",
        args_trans={0: "cv2_hwc_to_bchw"},
        output_rule="keep_on_device",
        notes="参数映射可能需要进一步调整"
    ))

    # cv2.Laplacian - P3
    registry.register(OpSpec(
        source="cv2.Laplacian",
        strategy=StrategyType.STANDARD_OP,
        priority=3,
        target_lib="kornia.filters",
        target_func="laplacian",
        args_trans={0: "cv2_hwc_to_bchw"},
        arg_renames={"ksize": "kernel_size"},
        output_rule="keep_on_device"
    ))

    # =====================================================
    # 3. PIL -> TorchVision (StandardOp)
    # =====================================================

    # PIL.Image.resize
    registry.register(OpSpec(
        source="PIL.Image.Image.resize",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torchvision.transforms.v2.functional",
        target_func="resize",
        args_trans={0: "pil_to_tensor"},  # self (Image)
        kwargs_trans={"size": "pass"},
        arg_value_maps={"resample": "pil_interp_mode"},
        output_rule="keep_on_device"
    ))

    # PIL.Image.crop
    registry.register(OpSpec(
        source="PIL.Image.Image.crop",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torchvision.transforms.v2.functional",
        target_func="crop",
        args_trans={0: "pil_to_tensor"},  # self (Image)
        structural_ops=["pil_box_to_crop_params"],
        output_rule="keep_on_device"
    ))

    # PIL.Image.rotate
    registry.register(OpSpec(
        source="PIL.Image.Image.rotate",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torchvision.transforms.v2.functional",
        target_func="rotate",
        args_trans={0: "pil_to_tensor"},  # self (Image)
        output_rule="keep_on_device"
    ))

    # PIL.Image.transpose (使用 Dispatcher)
    registry.register(OpSpec(
        source="PIL.Image.Image.transpose",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="",  # 使用 Dispatcher
        target_func="",  # 使用 Dispatcher
        args_trans={0: "pil_to_tensor"},
        arg_value_maps={"method": "pil_flip_method"},
        output_rule="keep_on_device",
        notes="使用 Dispatcher 根据 method 动态选择 hflip/vflip/rotate"
    ))
