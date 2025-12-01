"""
NumPy 函数策略 (NumPy Specs)

NumPy -> PyTorch 映射
"""

from ..core import OpSpec, StrategyType
from .base import DefaultStrategyRegistry


def register_numpy_specs(registry: DefaultStrategyRegistry) -> None:
    """注册 NumPy 相关函数策略"""

    # =====================================================
    # 1. 数组创建 - 工厂函数
    # =====================================================

    # np.array
    registry.register(OpSpec(
        source="numpy.array",
        strategy=StrategyType.STANDARD_OP,
        priority=1,
        target_lib="torch",
        target_func="tensor",
        args_trans={},  # 不转换输入，让 torch.tensor 自己处理
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.asarray
    registry.register(OpSpec(
        source="numpy.asarray",
        strategy=StrategyType.STANDARD_OP,
        priority=1,
        target_lib="torch",
        target_func="as_tensor",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.zeros
    registry.register(OpSpec(
        source="numpy.zeros",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="zeros",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.ones
    registry.register(OpSpec(
        source="numpy.ones",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="ones",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.empty
    registry.register(OpSpec(
        source="numpy.empty",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="empty",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.full
    registry.register(OpSpec(
        source="numpy.full",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="full",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.arange
    registry.register(OpSpec(
        source="numpy.arange",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="arange",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.linspace
    registry.register(OpSpec(
        source="numpy.linspace",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="linspace",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.eye
    registry.register(OpSpec(
        source="numpy.eye",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="eye",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # *_like 函数
    for func in ["zeros_like", "ones_like", "empty_like", "full_like"]:
        registry.register(OpSpec(
            source=f"numpy.{func}",
            strategy=StrategyType.STANDARD_OP,
            priority=2,
            target_lib="torch",
            target_func=func,
            args_trans={0: "ensure_tensor"},
            output_rule="keep_on_device"
        ))

    # =====================================================
    # 2. 形状操作
    # =====================================================

    # np.reshape
    registry.register(OpSpec(
        source="numpy.reshape",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="reshape",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # np.transpose
    registry.register(OpSpec(
        source="numpy.transpose",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="permute",
        args_trans={0: "ensure_tensor"},
        arg_renames={"axes": "dims"},
        output_rule="keep_on_device"
    ))

    # np.expand_dims
    registry.register(OpSpec(
        source="numpy.expand_dims",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="unsqueeze",
        args_trans={0: "ensure_tensor"},
        arg_renames={"axis": "dim"},
        output_rule="keep_on_device"
    ))

    # np.squeeze
    registry.register(OpSpec(
        source="numpy.squeeze",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="squeeze",
        args_trans={0: "ensure_tensor"},
        arg_renames={"axis": "dim"},
        output_rule="keep_on_device"
    ))

    # np.ravel
    registry.register(OpSpec(
        source="numpy.ravel",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="ravel",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # =====================================================
    # 3. 拼接与分割
    # =====================================================

    # np.concatenate
    registry.register(OpSpec(
        source="numpy.concatenate",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="cat",
        args_trans={0: "list_to_tensor_stack"},
        arg_renames={"axis": "dim"},
        output_rule="keep_on_device"
    ))

    # np.stack
    registry.register(OpSpec(
        source="numpy.stack",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="stack",
        args_trans={0: "list_to_tensor_stack"},
        arg_renames={"axis": "dim"},
        output_rule="keep_on_device"
    ))

    # np.vstack
    registry.register(OpSpec(
        source="numpy.vstack",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="vstack",
        args_trans={0: "list_to_tensor_stack"},
        output_rule="keep_on_device"
    ))

    # np.hstack
    registry.register(OpSpec(
        source="numpy.hstack",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="hstack",
        args_trans={0: "list_to_tensor_stack"},
        output_rule="keep_on_device"
    ))

    # np.split
    registry.register(OpSpec(
        source="numpy.split",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="split",
        args_trans={0: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # =====================================================
    # 4. 数学运算 - 二元
    # =====================================================

    binary_ops = [
        ("add", "add"),
        ("subtract", "sub"),
        ("multiply", "mul"),
        ("divide", "div"),
        ("true_divide", "true_divide"),
        ("floor_divide", "floor_divide"),
        ("matmul", "matmul"),
        ("dot", "matmul"),
        ("power", "pow"),
        ("mod", "remainder"),
        ("maximum", "maximum"),
        ("minimum", "minimum"),
    ]

    for np_func, torch_func in binary_ops:
        registry.register(OpSpec(
            source=f"numpy.{np_func}",
            strategy=StrategyType.STANDARD_OP,
            priority=2,
            target_lib="torch",
            target_func=torch_func,
            args_trans={0: "ensure_tensor", 1: "ensure_tensor"},
            output_rule="keep_on_device"
        ))

    # =====================================================
    # 5. 数学运算 - 一元
    # =====================================================

    unary_ops = [
        "abs", "sqrt", "exp", "log", "log2", "log10",
        "sin", "cos", "tan", "arcsin", "arccos", "arctan",
        "sinh", "cosh", "tanh",
        "floor", "ceil", "round",
        "negative", "sign",
    ]

    # NumPy 到 PyTorch 函数名映射
    unary_name_map = {
        "arcsin": "asin",
        "arccos": "acos",
        "arctan": "atan",
        "negative": "neg",
    }

    for func in unary_ops:
        torch_func = unary_name_map.get(func, func)
        registry.register(OpSpec(
            source=f"numpy.{func}",
            strategy=StrategyType.STANDARD_OP,
            priority=2,
            target_lib="torch",
            target_func=torch_func,
            args_trans={0: "ensure_tensor"},
            output_rule="keep_on_device"
        ))

    # np.clip
    registry.register(OpSpec(
        source="numpy.clip",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="clamp",
        args_trans={0: "ensure_tensor"},
        arg_renames={"a_min": "min", "a_max": "max"},
        output_rule="keep_on_device"
    ))

    # =====================================================
    # 6. 统计运算
    # =====================================================

    stat_ops = ["mean", "sum", "std", "var", "prod"]

    for func in stat_ops:
        registry.register(OpSpec(
            source=f"numpy.{func}",
            strategy=StrategyType.STANDARD_OP,
            priority=2,
            target_lib="torch",
            target_func=func,
            args_trans={0: "ensure_tensor"},
            arg_renames={"axis": "dim"},
            output_rule="keep_on_device"
        ))

    # np.max / np.min
    for func in ["max", "min"]:
        registry.register(OpSpec(
            source=f"numpy.{func}",
            strategy=StrategyType.STANDARD_OP,
            priority=2,
            target_lib="torch",
            target_func=func,
            args_trans={0: "ensure_tensor"},
            arg_renames={"axis": "dim"},
            output_rule="keep_on_device"
        ))

    # np.argmax / np.argmin
    for func in ["argmax", "argmin"]:
        registry.register(OpSpec(
            source=f"numpy.{func}",
            strategy=StrategyType.STANDARD_OP,
            priority=2,
            target_lib="torch",
            target_func=func,
            args_trans={0: "ensure_tensor"},
            arg_renames={"axis": "dim"},
            output_rule="keep_on_device"
        ))

    # =====================================================
    # 7. 逻辑运算
    # =====================================================

    # np.where
    registry.register(OpSpec(
        source="numpy.where",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="where",
        args_trans={0: "ensure_tensor", 1: "ensure_tensor", 2: "ensure_tensor"},
        output_rule="keep_on_device"
    ))

    # 逻辑操作
    logic_ops = [
        ("logical_and", "logical_and", 2),
        ("logical_or", "logical_or", 2),
        ("logical_not", "logical_not", 1),
        ("logical_xor", "logical_xor", 2),
    ]

    for np_func, torch_func, n_args in logic_ops:
        args_trans = {i: "ensure_tensor" for i in range(n_args)}
        registry.register(OpSpec(
            source=f"numpy.{np_func}",
            strategy=StrategyType.STANDARD_OP,
            priority=2,
            target_lib="torch",
            target_func=torch_func,
            args_trans=args_trans,
            output_rule="keep_on_device"
        ))

    # =====================================================
    # 8. 随机数
    # =====================================================

    # np.random.rand
    registry.register(OpSpec(
        source="numpy.random.rand",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="rand",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device",
        notes="Seed 机制不同，需同步 torch.manual_seed"
    ))

    # np.random.randn
    registry.register(OpSpec(
        source="numpy.random.randn",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="randn",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.random.randint
    registry.register(OpSpec(
        source="numpy.random.randint",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="torch",
        target_func="randint",
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.random.choice (使用 Dispatcher)
    registry.register(OpSpec(
        source="numpy.random.choice",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="",  # 使用 Dispatcher
        target_func="",  # 使用 Dispatcher
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device",
        notes="使用 Dispatcher 处理复杂的采样逻辑"
    ))

    # np.random.normal (使用 Dispatcher)
    registry.register(OpSpec(
        source="numpy.random.normal",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="",  # 使用 Dispatcher
        target_func="",  # 使用 Dispatcher
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))

    # np.random.uniform (使用 Dispatcher)
    registry.register(OpSpec(
        source="numpy.random.uniform",
        strategy=StrategyType.STANDARD_OP,
        priority=2,
        target_lib="",  # 使用 Dispatcher
        target_func="",  # 使用 Dispatcher
        args_trans={},
        injected_kwargs={"device": "TARGET_DEVICE"},
        output_rule="keep_on_device"
    ))
