"""
迁移模块核心逻辑
包含: OpSpec, ContextTracker, UniversalWrapper, PatchInjector
"""

import re
import logging
import threading
import functools
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 策略类型枚举
# ============================================================

class StrategyType(Enum):
    """迁移策略类型"""
    MOVE_ONLY = "MoveOnly"           # 仅搬运 Tensor (PyTorch 原生)
    STANDARD_OP = "StandardOp"       # 函数替换 (cv2 -> kornia)
    FACTORY_OP = "FactoryOp"         # 对象工厂 (librosa -> torchaudio class)
    PIPELINE_OP = "PipelineOp"       # 管道容器 (Compose)
    NON_MIGRATABLE = "NonMigratable" # 显式标记不可迁移


# ============================================================
# 操作规格 (OpSpec)
# ============================================================

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
    output_rule: str = "keep_on_device"

    # --- 6. 运行时控制 ---
    fallback_condition: Optional[Callable] = None
    notes: str = ""

    # 设计说明：
    # 不使用 is_method 标志来跳过 args[0]
    # 对于类方法（如 PIL.Image.resize），args[0] 是 self（即图像对象）
    # OpSpec 定义者应该在 args_trans 中明确配置 {0: "pil_to_tensor"} 来处理
    # 这样统一了模块函数和实例方法的处理逻辑


# ============================================================
# 上下文追踪器
# ============================================================

class SparseContextTracker:
    """
    稀疏上下文追踪器

    - 线程安全
    - 只记录被 Patch 的函数
    - 维护运行时调用栈
    """

    def __init__(self):
        self._local = threading.local()

    @property
    def _stack(self) -> List[str]:
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack

    def enter(self, name: str) -> None:
        """函数入栈"""
        self._stack.append(name)

    def exit(self) -> None:
        """函数出栈"""
        if self._stack:
            self._stack.pop()

    def current_key(self) -> str:
        """获取当前上下文路径 (使用 -> 分隔符)"""
        return "->".join(self._stack)

    def depth(self) -> int:
        """获取当前栈深度"""
        return len(self._stack)

    def is_top_level(self) -> bool:
        """是否是栈底（最外层被 Hook 的函数）"""
        return len(self._stack) == 1

    def reset(self) -> None:
        """重置栈（用于错误恢复）"""
        self._stack.clear()


# ============================================================
# Context Key 工具函数
# ============================================================

def normalize_context_key(key: str) -> str:
    """
    归一化 context key：去除调用计数 #数字

    'main#1->process#2->cv2.resize#1' -> 'main->process->cv2.resize'
    """
    if not key:
        return ""
    return re.sub(r'#\d+', '', key)


# ============================================================
# 工具函数
# ============================================================

def make_hashable(obj: Any) -> Any:
    """
    递归地将对象转换为可哈希类型

    - list -> tuple
    - dict -> frozenset of (key, value) tuples
    - set -> frozenset
    """
    if isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, set):
        return frozenset(make_hashable(item) for item in obj)
    else:
        # 尝试直接哈希，如果失败则返回 str
        try:
            hash(obj)
            return obj
        except TypeError:
            return str(obj)


def resolve_object_path(obj_path: str) -> Tuple[Any, Any, str]:
    """
    递归解析对象路径，返回 (parent_obj, target_obj, attr_name)

    处理场景：
    - cv2.resize -> (cv2_module, resize_func, 'resize')
    - torchvision.transforms.Resize -> (transforms_module, Resize_class, 'Resize')
    - torchvision.transforms.Resize.forward -> (Resize_class, forward_method, 'forward')

    Args:
        obj_path: 对象的完整路径，如 "cv2.resize"

    Returns:
        (parent_obj, target_obj, attr_name) 元组
        parent_obj: 目标对象的父对象（用于 setattr）
        target_obj: 目标对象本身
        attr_name: 目标对象在父对象中的属性名
    """
    import importlib

    parts = obj_path.split('.')
    if len(parts) < 2:
        raise ValueError(f"Invalid object path: {obj_path}")

    # 从第一个部分开始，尝试导入模块
    # 然后逐步向下查找属性
    current_obj = None
    module_end_idx = 0

    # 尝试找到最长的可导入模块路径
    for i in range(len(parts), 0, -1):
        module_path = '.'.join(parts[:i])
        try:
            current_obj = importlib.import_module(module_path)
            module_end_idx = i
            break
        except ImportError:
            continue

    if current_obj is None:
        raise ImportError(f"Cannot import any module from path: {obj_path}")

    # 从模块开始，逐步获取属性
    parent_obj = None
    attr_name = None

    for i in range(module_end_idx, len(parts)):
        parent_obj = current_obj
        attr_name = parts[i]

        if not hasattr(current_obj, attr_name):
            raise AttributeError(f"'{type(current_obj).__name__}' has no attribute '{attr_name}'")

        current_obj = getattr(current_obj, attr_name)

    return parent_obj, current_obj, attr_name


# ============================================================
# 通用包装器
# ============================================================

class UniversalWrapper:
    """
    通用函数包装器

    负责运行时的调度：
    1. 上下文管理
    2. 决策是否迁移
    3. 执行迁移策略
    4. 异常降级
    """

    def __init__(
        self,
        original_func: Callable,
        func_path: str,
        spec: OpSpec,
        context_tracker: SparseContextTracker,
        context_device_map: Dict[str, str],
        processors: 'ProcessorRegistry',
        default_device: str = "cuda:0"
    ):
        self.original_func = original_func
        self.func_path = func_path
        self.spec = spec
        self.context_tracker = context_tracker
        self.context_device_map = context_device_map
        self.processors = processors
        self.default_device = default_device

        # Lazy loaded backend
        self._backend = None
        self._backend_instance_cache = {}

        # 保持原函数的元信息
        functools.update_wrapper(self, original_func)

    def __call__(self, *args, **kwargs):
        """拦截函数调用"""
        # 1. 上下文入栈
        self.context_tracker.enter(self.func_path)
        current_key = self.context_tracker.current_key()
        normalized_key = normalize_context_key(current_key)

        try:
            # 2. 决策：是否迁移？
            target_device = self._get_target_device(normalized_key)
            should_migrate = self._should_migrate(target_device)

            if should_migrate:
                # --- GPU 分支 ---
                result = self._execute_migrated(args, kwargs, target_device)
            else:
                # --- CPU 分支 ---
                result = self.original_func(*args, **kwargs)

        except Exception as e:
            # --- 异常降级 ---
            logger.warning(f"Migration failed at {current_key}: {e}. Falling back to CPU.")
            try:
                result = self.original_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise

        finally:
            # 3. 上下文出栈
            self.context_tracker.exit()

        return result

    def __get__(self, obj, objtype=None):
        """
        支持作为描述符，使 Wrapper 能够正确处理实例方法

        当 Wrapper 被设置为类属性时，__get__ 会被调用
        """
        if obj is None:
            return self
        # 返回一个绑定了 self 的方法
        return functools.partial(self.__call__, obj)

    def __getattr__(self, name):
        """
        代理原始函数的属性访问

        这对于 numpy.ufunc 等特殊对象非常重要，因为它们有
        reduce, accumulate, outer, at 等方法需要被代理。
        """
        # 避免无限递归：只代理不在自身 __dict__ 中的属性
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.original_func, name)

    def _get_target_device(self, normalized_key: str) -> Optional[str]:
        """获取目标设备"""
        # 精确匹配
        if normalized_key in self.context_device_map:
            return self.context_device_map[normalized_key]

        # 后缀匹配（从长到短）
        parts = normalized_key.split('->')
        for i in range(1, len(parts)):
            suffix = '->'.join(parts[i:])
            if suffix in self.context_device_map:
                return self.context_device_map[suffix]

        # 函数名匹配
        func_name = parts[-1] if parts else None
        if func_name and func_name in self.context_device_map:
            return self.context_device_map[func_name]

        return None

    def _should_migrate(self, target_device: Optional[str]) -> bool:
        """判断是否应该迁移"""
        if target_device is None:
            return False

        if self.spec.strategy == StrategyType.NON_MIGRATABLE:
            return False

        if not target_device.startswith('cuda'):
            return False

        # 检查 fallback 条件
        if self.spec.fallback_condition:
            try:
                if self.spec.fallback_condition():
                    return False
            except Exception:
                pass

        return True

    def _execute_migrated(self, args: tuple, kwargs: dict, target_device: str) -> Any:
        """执行迁移后的计算"""
        strategy = self.spec.strategy

        if strategy == StrategyType.MOVE_ONLY:
            return self._execute_move_only(args, kwargs, target_device)
        elif strategy == StrategyType.STANDARD_OP:
            return self._execute_standard_op(args, kwargs, target_device)
        elif strategy == StrategyType.FACTORY_OP:
            return self._execute_factory_op(args, kwargs, target_device)
        elif strategy == StrategyType.PIPELINE_OP:
            return self._execute_pipeline_op(args, kwargs, target_device)
        else:
            return self.original_func(*args, **kwargs)

    def _execute_move_only(self, args: tuple, kwargs: dict, target_device: str) -> Any:
        """MoveOnly 策略：只搬运输入"""
        new_args = self._transform_args(args, target_device)
        new_kwargs = self._transform_kwargs(kwargs, target_device)
        result = self.original_func(*new_args, **new_kwargs)
        return self._process_output(result)

    def _execute_standard_op(self, args: tuple, kwargs: dict, target_device: str) -> Any:
        """StandardOp 策略：函数替换"""
        # 1. 转换输入参数
        new_args = self._transform_args(args, target_device)
        new_kwargs = self._transform_kwargs(kwargs, target_device)

        # 2. 参数值映射
        new_args, new_kwargs = self._apply_value_maps(new_args, new_kwargs)

        # 3. 参数改名
        new_kwargs = self._apply_renames(new_kwargs)

        # 4. 注入默认参数
        new_kwargs = self._inject_kwargs(new_kwargs, target_device)

        # 5. 结构性调整
        new_args, new_kwargs = self._apply_structural_ops(new_args, new_kwargs)

        # 6. 获取后端函数
        backend = self._get_backend()

        # 7. 执行
        result = backend(*new_args, **new_kwargs)

        # 8. 处理输出
        return self._process_output(result)

    def _execute_factory_op(self, args: tuple, kwargs: dict, target_device: str) -> Any:
        """FactoryOp 策略：工厂模式"""
        # 1. 转换输入参数
        new_args = self._transform_args(args, target_device)
        new_kwargs = self._transform_kwargs(kwargs, target_device)

        # 2. 参数值映射和改名
        new_args, new_kwargs = self._apply_value_maps(new_args, new_kwargs)
        new_kwargs = self._apply_renames(new_kwargs)
        new_kwargs = self._inject_kwargs(new_kwargs, target_device)

        # 3. 分离数据参数和配置参数
        data_arg = None
        config_kwargs = dict(new_kwargs)

        # 提取数据参数（通常是第一个位置参数或 'y'/'x' 关键字参数）
        if new_args:
            data_arg = new_args[0]
            new_args = new_args[1:]
        else:
            for data_key in ['y', 'x', 'input', 'data', 'waveform']:
                if data_key in config_kwargs:
                    data_arg = config_kwargs.pop(data_key)
                    break

        # 4. 获取或创建 Transform 实例
        instance = self._get_or_create_factory_instance(config_kwargs, target_device)

        # 5. 执行
        if data_arg is not None:
            result = instance(data_arg)
        else:
            result = instance(*new_args)

        # 6. 处理输出
        return self._process_output(result)

    def _execute_pipeline_op(self, args: tuple, kwargs: dict, target_device: str) -> Any:
        """PipelineOp 策略：管道编译"""
        logger.warning(f"PipelineOp not fully implemented for {self.func_path}, falling back")
        return self.original_func(*args, **kwargs)

    def _transform_args(self, args: tuple, target_device: str) -> tuple:
        """
        转换位置参数
        """
        new_args = list(args)

        for idx, rule_name in self.spec.args_trans.items():
            if idx < len(new_args):
                processor = self.processors.get_input_processor(rule_name)
                if processor:
                    new_args[idx] = processor(new_args[idx], target_device)

        return tuple(new_args)

    def _transform_kwargs(self, kwargs: dict, target_device: str) -> dict:
        """转换关键字参数"""
        new_kwargs = dict(kwargs)
        for key, rule_name in self.spec.kwargs_trans.items():
            if key in new_kwargs:
                processor = self.processors.get_input_processor(rule_name)
                if processor:
                    new_kwargs[key] = processor(new_kwargs[key], target_device)
        return new_kwargs

    def _apply_value_maps(self, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """应用参数值映射"""
        new_args = list(args)
        new_kwargs = dict(kwargs)

        for param_name, mapper in self.spec.arg_value_maps.items():
            if isinstance(mapper, str):
                map_func = self.processors.get_value_mapper(mapper)
            else:
                map_func = mapper

            if map_func is None:
                continue

            if param_name in new_kwargs:
                new_kwargs[param_name] = map_func(new_kwargs[param_name])

        return tuple(new_args), new_kwargs

    def _apply_renames(self, kwargs: dict) -> dict:
        """应用参数改名"""
        new_kwargs = dict(kwargs)
        for old_name, new_name in self.spec.arg_renames.items():
            if old_name in new_kwargs:
                new_kwargs[new_name] = new_kwargs.pop(old_name)
        return new_kwargs

    def _inject_kwargs(self, kwargs: dict, target_device: str) -> dict:
        """注入默认参数"""
        new_kwargs = dict(kwargs)
        for key, value in self.spec.injected_kwargs.items():
            if key not in new_kwargs:
                if value == "TARGET_DEVICE":
                    new_kwargs[key] = target_device
                else:
                    new_kwargs[key] = value
        return new_kwargs

    def _apply_structural_ops(self, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """应用结构性操作"""
        new_args = list(args)
        new_kwargs = dict(kwargs)

        for op in self.spec.structural_ops:
            if op == "swap_args_0_1" and len(new_args) >= 2:
                new_args[0], new_args[1] = new_args[1], new_args[0]
            elif op == "swap_args_1_2" and len(new_args) >= 3:
                new_args[1], new_args[2] = new_args[2], new_args[1]

        return tuple(new_args), new_kwargs

    def _get_backend(self) -> Callable:
        """
        Lazy 加载后端函数

        查找顺序：
        1. 如果有 Dispatcher，优先使用 Dispatcher
        2. 如果有 target_lib + target_func，加载目标函数
        3. 否则使用原始函数
        """
        if self._backend is not None:
            return self._backend

        # 1. 检查是否有 Dispatcher
        try:
            from .dispatchers import DispatcherRegistry
            dispatcher = DispatcherRegistry.get(self.func_path)
            if dispatcher is not None:
                self._backend = dispatcher
                return self._backend
        except ImportError:
            pass

        # 2. 加载目标函数
        if not self.spec.target_lib or not self.spec.target_func:
            self._backend = self.original_func
            return self._backend

        try:
            import importlib
            module = importlib.import_module(self.spec.target_lib)
            self._backend = getattr(module, self.spec.target_func)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to load backend {self.spec.target_lib}.{self.spec.target_func}: {e}")
            self._backend = self.original_func

        return self._backend

    def _get_or_create_factory_instance(self, config_kwargs: dict, target_device: str):
        """获取或创建 FactoryOp 的实例（带缓存）"""
        # 生成可哈希的缓存 key
        hashable_kwargs = make_hashable(config_kwargs)
        cache_key = (target_device, hashable_kwargs)

        if cache_key in self._backend_instance_cache:
            return self._backend_instance_cache[cache_key]

        try:
            import importlib
            module = importlib.import_module(self.spec.target_lib)
            cls = getattr(module, self.spec.target_func)
            instance = cls(**config_kwargs)

            if hasattr(instance, 'to'):
                instance = instance.to(target_device)

            self._backend_instance_cache[cache_key] = instance
            return instance

        except Exception as e:
            logger.warning(f"Failed to create factory instance: {e}")
            raise

    def _process_output(self, result: Any) -> Any:
        """处理输出"""
        output_processor = self.processors.get_output_processor(self.spec.output_rule)
        if output_processor:
            return output_processor(result)
        return result


# ============================================================
# Patch 注入器
# ============================================================

class PatchInjector:
    """
    Patch 注入器

    负责在运行时将目标函数替换为 UniversalWrapper
    """

    def __init__(
        self,
        context_device_map: Dict[str, str],
        registry: 'StrategyRegistry',
        processors: 'ProcessorRegistry',
        context_tracker: Optional[SparseContextTracker] = None,
        default_device: str = "cuda:0"
    ):
        self.context_device_map = context_device_map
        self.registry = registry
        self.processors = processors
        self.context_tracker = context_tracker or SparseContextTracker()
        self.default_device = default_device

        # 保存原始函数和父对象，用于恢复
        self._original_funcs: Dict[str, Tuple[Any, Callable, str]] = {}  # path -> (parent, func, attr_name)

    def apply_all(self) -> Dict[str, int]:
        """
        应用所有 Patch

        Returns:
            统计信息 {'patched': n, 'failed': m, 'skipped': k}
        """
        patched = 0
        failed = 0
        skipped = 0

        # 从 context_device_map 和 registry 中提取需要 patch 的函数
        target_funcs = self._extract_target_functions()

        for func_path in target_funcs:
            try:
                result = self._patch_function(func_path)
                if result == 'patched':
                    patched += 1
                    logger.debug(f"Patched: {func_path}")
                elif result == 'skipped':
                    skipped += 1
                    logger.debug(f"Skipped: {func_path}")
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to patch {func_path}: {e}")

        logger.info(f"Patch complete: {patched} patched, {skipped} skipped, {failed} failed")
        return {'patched': patched, 'failed': failed, 'skipped': skipped}

    def restore_all(self) -> int:
        """
        恢复所有原始函数

        Returns:
            恢复的数量
        """
        restored = 0
        for func_path, (parent_obj, original_func, attr_name) in self._original_funcs.items():
            try:
                setattr(parent_obj, attr_name, original_func)
                restored += 1
            except Exception as e:
                logger.warning(f"Failed to restore {func_path}: {e}")

        self._original_funcs.clear()
        logger.info(f"Restored {restored} functions")
        return restored

    def _extract_target_functions(self) -> Set[str]:
        """从 context_device_map 和 registry 中提取需要 patch 的函数名"""
        funcs = set()

        # 从 context_device_map 提取
        for context_key in self.context_device_map.keys():
            parts = context_key.split('->')
            if parts:
                func_name = parts[-1]
                if '.' in func_name:
                    funcs.add(func_name)

        # 从 registry 获取所有已注册的函数
        for source in self.registry.all_sources():
            funcs.add(source)

        return funcs

    def _patch_function(self, func_path: str) -> str:
        """
        Patch 单个函数

        Args:
            func_path: 函数路径，如 "cv2.resize" 或 "torchvision.transforms.Resize.forward"

        Returns:
            'patched' | 'skipped' | 'failed'
        """
        # 获取策略
        spec = self.registry.get(func_path)
        if spec is None:
            return 'skipped'

        if spec.strategy == StrategyType.NON_MIGRATABLE:
            return 'skipped'

        # 检查是否已经 patch
        if func_path in self._original_funcs:
            return 'skipped'

        try:
            # 使用递归解析器获取对象
            parent_obj, original_func, attr_name = resolve_object_path(func_path)
        except (ImportError, AttributeError, ValueError) as e:
            logger.warning(f"Cannot resolve {func_path}: {e}")
            return 'failed'

        # 保存原始信息
        self._original_funcs[func_path] = (parent_obj, original_func, attr_name)

        # 创建包装器
        wrapper = UniversalWrapper(
            original_func=original_func,
            func_path=func_path,
            spec=spec,
            context_tracker=self.context_tracker,
            context_device_map=self.context_device_map,
            processors=self.processors,
            default_device=self.default_device
        )

        # 替换
        setattr(parent_obj, attr_name, wrapper)

        return 'patched'


# ============================================================
# 处理器注册表接口
# ============================================================

class ProcessorRegistry:
    """处理器注册表接口"""

    def get_input_processor(self, rule_name: str) -> Optional[Callable]:
        raise NotImplementedError

    def get_value_mapper(self, rule_name: str) -> Optional[Callable]:
        raise NotImplementedError

    def get_output_processor(self, rule_name: str) -> Optional[Callable]:
        raise NotImplementedError


# ============================================================
# 策略注册表接口
# ============================================================

class StrategyRegistry:
    """策略注册表接口"""

    def get(self, func_path: str) -> Optional[OpSpec]:
        raise NotImplementedError

    def register(self, spec: OpSpec) -> None:
        raise NotImplementedError

    def all_sources(self) -> List[str]:
        raise NotImplementedError
