"""
增强追踪器 - 整合内存分析、调用上下文和变量版本追踪
"""
import sys
import time
import re
import torch
import logging
import ast
import inspect
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from contextlib import contextmanager

from .memory_profiler import MemoryProfiler, VariableInfo
from .base_tracer import BaseTracer
from .dag_builder import DAGBuilder
from .function_filter import UnifiedFunctionFilter, FilterLevel

logger = logging.getLogger(__name__)

@dataclass 
class VariableSnapshot:
    """变量快照，用于检测变量变化"""
    value_id: int  # id(value)
    value_type: str  # type(value).__name__
    actual_value: Optional[Any] = None  # 简单类型的实际值
    shape: Optional[Tuple] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    size_mb: Optional[float] = None
    hash_value: Optional[int] = None  # 对于小对象的哈希值
    
    @staticmethod
    def create_snapshot(value: Any) -> 'VariableSnapshot':
        """创建变量快照（兼容 PyTorch/TensorFlow/NumPy 的鲁棒版本）"""
        snapshot = VariableSnapshot(
            value_id=id(value),
            value_type=type(value).__name__
        )

        try:
            # 1. PyTorch Tensor 特殊处理
            # 优先使用 is_tensor 避免触发 _OpNamespace 的 __getattr__ 错误
            if 'torch' in sys.modules and torch.is_tensor(value):
                snapshot.shape = tuple(value.shape)
                snapshot.dtype = str(value.dtype)
                snapshot.device = str(value.device)
                # 某些特殊 Tensor (如 Quantized) 计算大小时可能报错，加保护
                try:
                    snapshot.size_mb = value.numel() * value.element_size() / (1024**2)
                except Exception:
                    snapshot.size_mb = 0.0
                return snapshot

            # 2. TensorFlow Tensor 特殊处理 (避免 AttributeError: no attribute 'numel')
            # 通过类名判断，避免强依赖 tensorflow 库
            is_tf_tensor = False
            v_type = type(value).__name__
            if ('Tensor' in v_type or 'Variable' in v_type) and hasattr(value, 'get_shape'):
                is_tf_tensor = True
                try:
                    # TF 处理逻辑
                    shape = value.shape
                    # 处理 Symbolic Tensor 的动态 Shape (可能包含 None)
                    if shape and hasattr(shape, 'as_list'):
                        shape_list = shape.as_list()
                        # 将 None 替换为 -1 或 0，以便记录
                        snapshot.shape = tuple((s if s is not None else -1) for s in shape_list)

                    snapshot.dtype = str(getattr(value, 'dtype', 'unknown'))
                    snapshot.device = str(getattr(value, 'device', 'unknown'))

                    # TF 计算大小比较复杂，这里简化估算
                    # 注意：Symbolic Tensor 无法计算准确大小，设为 0
                    snapshot.size_mb = 0.0
                except Exception:
                    pass
                return snapshot

            # 3. 基础类型 (int, float, str, bool)
            if isinstance(value, (int, float, str, bool)):
                snapshot.actual_value = value
                try:
                    snapshot.hash_value = hash(value)
                except Exception:
                    pass
                return snapshot

            # 4. 容器类型 (Tuple/List) - 浅层估算
            if isinstance(value, (tuple, list)):
                snapshot.size_mb = len(value) * 0.001
                # 仅对短元组保存值
                if isinstance(value, tuple) and len(value) <= 10:
                    if VariableSnapshot._is_json_serializable(value):
                        snapshot.actual_value = value
                        try:
                            snapshot.hash_value = hash(value)
                        except Exception:
                            pass
                return snapshot

            # 5. 字典类型
            if isinstance(value, dict):
                snapshot.size_mb = len(value) * 0.001
                return snapshot

            # 6. 鸭子类型兜底 (NumPy 等其他数组)
            # 必须放在最后，且使用异常捕获防止 _OpNamespace 等怪异对象崩溃
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                snapshot.shape = tuple(value.shape)
                snapshot.dtype = str(value.dtype)
                # 尝试计算大小
                if hasattr(value, 'nbytes'):  # NumPy
                    snapshot.size_mb = value.nbytes / (1024**2)
                elif hasattr(value, 'numel') and hasattr(value, 'element_size'):
                    snapshot.size_mb = value.numel() * value.element_size() / (1024**2)

        except Exception:
            # 捕获所有意料之外的错误 (RuntimeError, RecursionError 等)
            # 确保快照创建绝对不会导致主程序崩溃
            pass

        return snapshot
    
    @staticmethod
    def _is_json_serializable(obj: Any, max_depth: int = 3, current_depth: int = 0) -> bool:
        """检查对象是否可以JSON序列化，并且适合存储"""
        # 防止无限递归
        if current_depth > max_depth:
            return False
        
        # 基本类型
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return True
        
        # 不可序列化的类型
        if hasattr(obj, 'shape') and hasattr(obj, 'device'):  # Tensor
            return False
        if hasattr(obj, '__dict__') and not isinstance(obj, (list, dict, tuple)):  # 自定义类实例
            return False
        if callable(obj):  # 函数、方法等
            return False
        
        # 容器类型递归检查
        if isinstance(obj, (list, tuple)):
            if len(obj) > 50:  # 大型容器
                return False
            return all(VariableSnapshot._is_json_serializable(item, max_depth, current_depth + 1) for item in obj)
        
        elif isinstance(obj, dict):
            if len(obj) > 20:  # 大型字典
                return False
            # 检查键和值都可序列化
            for key, value in obj.items():
                if not isinstance(key, (str, int, float, bool)):  # JSON要求键为字符串
                    return False
                if not VariableSnapshot._is_json_serializable(value, max_depth, current_depth + 1):
                    return False
            return True
        
        # 其他未知类型默认不安全
        return False
    
    def has_changed(self, other: 'VariableSnapshot') -> bool:
        """检测变量是否发生变化"""
        if self.value_id != other.value_id:
            return True
        if self.shape != other.shape or self.dtype != other.dtype or self.device != other.device:
            return True
        if self.hash_value is not None and other.hash_value is not None:
            return self.hash_value != other.hash_value
        return False

@dataclass
class ContextKeyManager:
    """Context Key管理器"""
    call_stack: List[str] = field(default_factory=list)
    variable_versions: Dict[str, int] = field(default_factory=dict)
    function_call_counts: Dict[str, int] = field(default_factory=dict)
    
    def enter_function(self, func_name: str) -> str:
        """进入函数，返回当前调用的context key"""
        # 基于当前调用路径的计数
        current_path = "->".join(self.call_stack)
        path_key = f"{current_path}->{func_name}" if current_path else func_name
        
        self.function_call_counts[path_key] = self.function_call_counts.get(path_key, 0) + 1
        call_count = self.function_call_counts[path_key]
        func_context = f"{func_name}#{call_count}"
        self.call_stack.append(func_context)
        return "->".join(self.call_stack)
    
    def exit_function(self):
        """退出函数"""
        if self.call_stack:
            self.call_stack.pop()
    
    def get_variable_context_key(self, var_name: str, is_global: bool = False) -> str:
        """获取变量的完整Context Key"""
        if is_global:
            version = self.get_next_variable_version(var_name, is_global=True)
            return f"{var_name}#{version}"
        
        call_chain = "->".join(self.call_stack)
        version = self.get_next_variable_version(var_name, call_chain)
        return f"{call_chain}->{var_name}#{version}"
    
    def get_next_variable_version(self, var_name: str, context_chain: str = "") -> int:
        """获取下一个变量版本号"""
        if context_chain:
            key = f"{context_chain}->{var_name}"
        else:
            key = var_name
        self.variable_versions[key] = self.variable_versions.get(key, 0) + 1
        return self.variable_versions[key]

class SimpleASTAnalyzer(ast.NodeVisitor):
    """AST分析器：负责静态解析代码行，创建中间变量节点并建立连边"""

    def __init__(self, tracer, func_context: str, frame=None):
        self.tracer = tracer
        self.func_context = func_context
        self.frame = frame  # 保存当前栈帧用于名字解析
        # 预编译正则，用于匹配去除版本号后缀的节点名
        self.version_pattern = re.compile(r'#\d+$')
        # 当前正在分析的代码行号 (用于区分自引用赋值中的变量版本)
        self.current_lineno = 0

    def visit_Assign(self, node):
        """处理赋值语句: targets = value"""
        # 0. 记录当前行号 (用于区分自引用赋值中的变量版本)
        # 注意：如果外部已经设置了 current_lineno（如在 _process_previous_line 中），
        # 则不要覆盖。因为 ast.parse() 单行代码时，AST 节点的 lineno=1，不是原始行号。
        if self.current_lineno == 0:
            self.current_lineno = getattr(node, 'lineno', 0)

        # 1. 分析右值 (RHS)，获取产生数据的源节点 (Operator 或 Function)
        # 这一步非常关键：self.visit 会分发到 visit_BinOp 或 visit_Call
        # 它们必须返回一个 DAGNode，否则链条就断了
        producer_node = self.visit(node.value)

        if not producer_node:
            return

        # 2. 获取左值 (LHS) 变量名列表
        target_names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_names.append(target.id)
            elif isinstance(target, ast.Tuple):  # 支持解包: a, b = func()
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        target_names.append(elt.id)

        # 3. 建立连边: Producer -> Creates/Produces -> Variable
        if target_names:
            edge_type = 'produces'

            for var_name in target_names:
                # 获取或创建变量节点 (使用 lineno 机制避免循环)
                var_node = self._ensure_variable_node(var_name)

                # 建立连边
                if var_node and var_node.node_id != producer_node.node_id:
                    self.tracer._add_edge(producer_node.node_id, var_node.node_id, edge_type)

    def visit_BinOp(self, node):
        """处理二元运算: left op right (返回 Operator Node)"""
        # 递归处理子节点
        self.visit(node.left)
        self.visit(node.right)

        # 1. 创建算子节点
        op_type = type(node.op)
        op_name = {
            ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul',
            ast.Div: 'div', ast.Pow: 'pow', ast.Mod: 'mod',
            ast.MatMult: 'matmul'
        }.get(op_type, 'unknown_op')

        lineno = getattr(node, 'lineno', 0)
        # 使用更具体的上下文，防止同名算子合并错误
        operator_context = f"{self.func_context}->operator.{op_name}#L{lineno}"

        # 获取或创建算子节点
        operator_node = self._get_or_create_operator_node(op_name, operator_context)

        # 2. 建立输入边: Operand -> Uses -> Operator
        for child in [node.left, node.right]:
            var_name = self._get_variable_name(child)
            if var_name:
                var_node = self._find_or_create_input_variable(var_name)
                if var_node:
                    self.tracer._add_edge(var_node.node_id, operator_node.node_id, 'uses')

        # [关键] 必须返回节点，供 visit_Assign 使用
        return operator_node

    def visit_Call(self, node):
        """处理函数调用: func(args) (返回 Function Node)"""
        # 1. 解析函数名
        runtime_name = self._resolve_runtime_name(node.func)
        ast_name = self._get_func_name(node.func)

        target_func_node = None
        if runtime_name:
            target_func_node = self._find_latest_function_node(runtime_name)
        if not target_func_node and ast_name and ast_name != runtime_name:
            target_func_node = self._find_latest_function_node(ast_name)

        # [关键修复 A] 防止递归时的自引用
        # 获取当前正在分析的函数节点的 ID
        current_func_node_id = self.tracer.context_map.get(self.func_context)

        if target_func_node and current_func_node_id and target_func_node.node_id == current_func_node_id:
            # 如果找到的是自己，说明 Runtime 还没创建子调用节点（或被过滤），或者匹配错了
            # 这种情况下，必须强制放弃这个节点，转而创建幽灵节点
            logger.debug(f"[防环] 忽略自引用函数节点: {target_func_node.name}")
            target_func_node = None

        # 2. 兜底创建 (Phantom Node)
        if not target_func_node:
            final_name = ast_name or runtime_name or "unknown_call"
            lineno = getattr(node, 'lineno', 0)
            target_func_node = self.tracer.dag_builder.dag.add_node(
                name=final_name,
                node_type="function_call",
                module="library",
                context=f"{self.func_context}->{final_name}#L{lineno}",
                version=0
            )
            target_func_node.performance['execution_time'] = 0.001

        # 3. 建立输入边
        # 合并 args 和 keywords
        all_args = list(node.args) + [kw.value for kw in node.keywords]

        for arg in all_args:
            self.visit(arg)  # 递归处理嵌套调用
            var_name = self._get_variable_name(arg)
            if var_name:
                var_node = self._find_or_create_input_variable(var_name)
                if var_node:
                    # [关键修复 B] 最后的防线：检查显式循环
                    if not self._would_create_cycle(var_node, target_func_node):
                        self.tracer._add_edge(var_node.node_id, target_func_node.node_id, 'uses')
                    else:
                        logger.debug(f"[防环] 拦截循环边: {var_node.name} -> {target_func_node.name}")

        # [关键] 必须返回节点，供 visit_Assign 使用
        return target_func_node

    # --- 辅助方法 ---

    def _ensure_variable_node(self, var_name: str):
        """确保(输出)变量节点存在，如果不存在则立即创建

        输出变量的严格匹配规则：
        1. 必须严格属于当前函数作用域（不能是父级或子函数的变量）
        2. 如果 Runtime 已经为当前行创建了变量，直接复用
        3. 否则强制创建新版本
        """
        current_context = "->".join(self.tracer.context_key_manager.call_stack)

        # 1. 查找 Runtime 已创建的、严格属于当前作用域的变量节点
        if var_name in self.tracer.variable_nodes:
            # 对于输出变量，我们只关心当前函数内的变量，不关心父级
            # 使用严格匹配：var_scope 必须完全等于 func_context
            candidates = []
            for node in self.tracer.variable_nodes[var_name]:
                var_scope = node.context.rsplit('->', 1)[0]
                if var_scope == self.func_context:
                    candidates.append(node)

            if candidates:
                latest = candidates[-1]
                # 如果 Runtime 已经为当前行创建了新版本，直接复用
                created_lineno = latest.attributes.get('created_at_lineno', 0)
                if created_lineno == self.current_lineno and self.current_lineno > 0:
                    return latest

        # 2. 强制创建新版本 (传入 lineno)
        var_node = self.tracer._create_variable_node(
            var_name, None, current_context,
            increment_version=True,
            lineno=self.current_lineno
        )
        return var_node

    def _find_or_create_input_variable(self, var_name: str):
        """查找(输入)变量，如果找不到返回None（避免产生噪音）"""
        return self._find_variable_node(var_name)

    def _get_or_create_operator_node(self, op_name: str, context: str):
        """获取或创建 Operator 节点"""
        for node in self.tracer.dag_builder.dag.nodes.values():
            if node.node_type == "operator" and node.context == context:
                return node

        node = self.tracer.dag_builder.dag.add_node(
            name=f"operator.{op_name}",
            node_type="operator",
            module="builtin",
            context=context,
            version=0
        )
        # 初始化性能数据
        node.performance.update({
            'execution_time': 0.001,
            'execution_time_ms': 0.01,
        })
        return node

    def _find_latest_function_node(self, func_name: str):
        """查找最近调用的匹配函数节点"""
        dag_nodes = self.tracer.dag_builder.dag.nodes
        for node_id in sorted(dag_nodes.keys(), reverse=True):
            node = dag_nodes[node_id]
            # [修复] 同时匹配 'function' (Runtime创建) 和 'function_call' (AST创建)
            if node.node_type in ('function', 'function_call'):
                normalized_name = self.version_pattern.sub('', node.name)
                # 匹配：完全相等 或 以后缀形式匹配 (如 numpy.dot 匹配 dot)
                if normalized_name == func_name or normalized_name.endswith(f".{func_name}"):
                    return node
        return None

    def _is_relevant_context(self, node) -> bool:
        """判断变量节点是否属于当前可见的作用域

        逻辑：
        1. 必须是当前函数的局部变量 (Context 完全匹配前缀)
        2. 或者是父级作用域的变量（闭包/外层函数）
        3. 或者是全局变量
        4. [关键] 不能是子函数的变量！(子函数的 context 会比当前更长)

        例如:
        - self.func_context = "run#1->__getitem__#1"
        - node.context = "run#1->__getitem__#1->img#1" -> 当前作用域，可见
        - node.context = "run#1->__getitem__#1->_normalize#1->img#1" -> 子函数，不可见！
        - node.context = "run#1->img#1" -> 父级作用域，可见
        """
        if not node.context:
            return False

        # 提取变量所在的函数上下文 (去除 ->var_name#version 后缀)
        # node.context 格式: "root#1->func#1->var#1"
        # 我们需要提取: "root#1->func#1"
        var_scope = node.context.rsplit('->', 1)[0]

        # 1. 精确匹配当前作用域 (最常见情况)
        #    var_scope == self.func_context
        if var_scope == self.func_context:
            return True

        # 2. 匹配父级作用域 (闭包/外层函数的变量)
        #    self.func_context 以 var_scope 开头，说明 var_scope 是父级
        #    例如: self.func_context = "run#1->inner#1", var_scope = "run#1"
        if self.func_context.startswith(var_scope + "->"):
            return True

        # 3. 全局变量
        if "global" in node.module:
            return True

        # 4. 其他情况（包括子函数的变量）不可见
        return False

    def _find_variable_node(self, var_name: str):
        """查找输入变量节点

        使用严格的作用域匹配 + 行号防环机制：
        1. 只查找当前作用域或父级作用域的变量（不包括子函数）
        2. 如果找到的变量是当前行创建的，回溯到上一个版本（防止自引用循环）
        """
        if var_name not in self.tracer.variable_nodes:
            return None

        nodes = self.tracer.variable_nodes[var_name]

        # 1. 严格筛选可见作用域的节点
        candidates = [n for n in nodes if self._is_relevant_context(n)]
        if not candidates:
            return None

        # 2. 选取最新的
        latest = candidates[-1]

        # 3. [防环] 如果最新版本是当前行产生的，回溯取上一个版本
        #    对于 img = self._normalize(img)：
        #    - 右边的 img 是输入，应该是之前某行创建的
        #    - 左边的 img 是输出，是当前行创建的新版本
        created_lineno = latest.attributes.get('created_at_lineno', 0)
        if created_lineno == self.current_lineno and self.current_lineno > 0:
            if len(candidates) >= 2:
                logger.debug(f"[防环] 变量 {var_name} 回溯版本 v{latest.version} -> v{candidates[-2].version}")
                return candidates[-2]
            # 只有当前行产生的版本，无法作为输入
            return None

        return latest

    def _get_variable_name(self, node):
        """从AST节点获取变量名"""
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _would_create_cycle(self, var_node, func_node):
        """检查添加 var_node -> func_node 的 uses 边是否会形成循环

        循环条件：如果 func_node 已经是 var_node 的生产者 (creates/modifies/produces)，
        那么 var_node 不能反过来作为 func_node 的输入 (uses)。
        """
        # 快速检查：节点 ID 相同（虽然不太可能，因为类型不同）
        if var_node.node_id == func_node.node_id:
            return True

        # 方法 1: 检查 Context 包含关系 (快)
        # 如果 var 是 func 的局部变量，且 func_node 就是当前 func，那么大概率是循环
        # var_node.context: "root->func#1->var#1"
        # func_node.context: "root->func#1"
        if var_node.context and func_node.context:
            if var_node.context.startswith(func_node.context + "->"):
                return True

        # 方法 2: 严格检查 Edge (慢但准确)
        # 遍历 DAG 的边，检查是否存在 func_node -> var_node 的 creates/modifies 边
        for edge in self.tracer.dag_builder.dag.edges:
            src, dst, edge_type = edge
            if src == func_node.node_id and dst == var_node.node_id:
                if edge_type in ('creates', 'modifies', 'produces'):
                    return True

        return False

    def _get_func_name(self, node):
        """从AST节点获取函数名（返回完整路径，如 cv2.resize）"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # 递归构建完整路径: cv2.resize, np.random.rand 等
            return self._get_full_func_path(node)
        return None

    def _get_full_func_path(self, node):
        """递归获取完整的函数调用路径

        例如:
            cv2.resize -> "cv2.resize"
            np.random.rand -> "np.random.rand"
            obj.method -> "obj.method"
        """
        parts = []
        curr = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value

        if isinstance(curr, ast.Name):
            parts.append(curr.id)
        else:
            # 复杂表达式如 func().method，只返回最后的属性名
            return node.attr if isinstance(node, ast.Attribute) else None

        parts.reverse()
        return ".".join(parts)

    def _resolve_runtime_name(self, func_node):
        """利用当前 Frame 动态解析 AST 对应的运行时对象名称

        例如: 代码写的是 np.max，解析对象后发现是 np.amax，返回 'amax'
        """
        if not self.frame:
            return None

        try:
            # 1. 提取 AST 中的名字链 (如 'np.max' -> ['np', 'max'])
            parts = []
            curr = func_node
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value

            if isinstance(curr, ast.Name):
                parts.append(curr.id)
            else:
                return None  # 复杂表达式无法简单解析

            parts.reverse()  # ['np', 'max']

            # 2. 在 Frame 中查找基础对象 ('np')
            base_name = parts[0]
            obj = None

            # 查找顺序: locals -> globals -> builtins
            if base_name in self.frame.f_locals:
                obj = self.frame.f_locals[base_name]
            elif base_name in self.frame.f_globals:
                obj = self.frame.f_globals[base_name]
            else:
                # 检查 builtins
                import builtins
                if hasattr(builtins, base_name):
                    obj = getattr(builtins, base_name)
                else:
                    return None

            # 3. 递归获取属性 (.max)
            for part in parts[1:]:
                obj = getattr(obj, part)

            # 4. 获取真实函数名
            if hasattr(obj, '__name__'):
                return obj.__name__
            elif hasattr(obj, '__class__'):
                return obj.__class__.__name__

        except Exception:
            # 解析过程中可能出现各种异常（如属性不存在），安全忽略
            pass

        return None

@dataclass
class EnhancedNode:
    """增强的节点信息"""
    id: int
    name: str
    type: str  # function/variable
    module: str
    context: str  # 调用上下文路径
    version: int  # 对于变量节点
    
    # 性能数据
    start_time: float
    end_time: float
    execution_time_ms: float
    
    # 内存数据
    memory_before_mb: float
    memory_after_mb: float
    memory_allocated_mb: float
    peak_memory_mb: float
    
    # 变量特定信息 - 统一使用VariableSnapshot管理
    variable_snapshot: Optional['VariableSnapshot'] = None
    
    # 便利属性：为了兼容性保留，但委托给variable_snapshot
    @property
    def actual_value(self) -> Optional[Any]:
        return self.variable_snapshot.actual_value if self.variable_snapshot else None
    
    @property
    def shape(self) -> Optional[Tuple]:
        return self.variable_snapshot.shape if self.variable_snapshot else None
    
    @property
    def dtype(self) -> Optional[str]:
        return self.variable_snapshot.dtype if self.variable_snapshot else None
    
    @property
    def device(self) -> Optional[str]:
        return self.variable_snapshot.device if self.variable_snapshot else None
    
    @property
    def size_mb(self) -> Optional[float]:
        return self.variable_snapshot.size_mb if self.variable_snapshot else None

class EnhancedTracer(BaseTracer):
    """增强的追踪器"""
    
    def __init__(self, max_depth: int = 12, track_memory: bool = True, track_gpu: bool = True, 
                 filter_level: FilterLevel = FilterLevel.CONSERVATIVE,
                 custom_filters: Optional[List[str]] = None,
                 custom_protections: Optional[List[str]] = None):
        """
        初始化增强追踪器
        
        Args:
            max_depth: 最大追踪深度
            track_memory: 是否追踪内存
            track_gpu: 是否追踪GPU
            filter_level: 过滤级别 (DISABLED/CONSERVATIVE/BALANCED/AGGRESSIVE)
            custom_filters: 自定义过滤规则 (支持通配符 *)
            custom_protections: 自定义保护规则 (支持通配符 *)
        """
        self.max_depth = max_depth
        self.tracing_active = False
        self.call_stack = []  # 调用栈
        
        # 装饰器模式的函数级别追踪
        self.function_tracing_mode = False  # 是否启用函数级别追踪模式
        self.target_functions = set()       # 被标记的目标函数名称
        self.function_trace_depth = max_depth  # 函数级追踪的深度
        self.in_target_function = False     # 是否正在目标函数内部
        self.target_call_depth = 0         # 目标函数调用深度
        
        # 使用统一的过滤系统（整合了原有的硬编码规则）
        self.function_filter = UnifiedFunctionFilter(
            filter_level=filter_level,
            custom_filters=custom_filters,
            custom_protections=custom_protections
        )
        
        self.track_memory = track_memory
        self.memory_profiler = MemoryProfiler(track_gpu=track_gpu) if track_memory else None

        # 节点存储 - 统一使用DAG
        self.dag_builder = DAGBuilder()
        self.variable_nodes: Dict[str, List[int]] = {}  # 变量名 -> node_id列表
        self.context_map: Dict[str, int] = {}  # context -> node_id
        
        # 当前追踪状态
        self.current_depth = 0
        self.context_stack: List[str] = []
        self.node_stack: List[EnhancedNode] = []
        
        # 新增：变量追踪和Context Key管理
        # 基于context key的返回值存储机制
        self.pending_returns: Dict[str, List[EnhancedNode]] = {}  # {context_key: [return_var_nodes]}
        self.latest_return_value_node = None  # 兼容旧代码
        self.context_key_manager = ContextKeyManager()
        self.variable_snapshots: Dict[str, VariableSnapshot] = {}  # context_key -> snapshot
        self.global_variables: Dict[str, Any] = {}  # 全局变量缓存
        self.ast_cache: Dict[str, ast.AST] = {}  # 函数源码AST缓存
        self.analyzed_lines: Set[str] = set()  # 已分析的行，防止重复分析
        
        # 上一行处理机制
        self.previous_line_info: Dict[str, Any] = {}  # 存储上一行的信息
        
        # 峰值内存追踪 - 每个函数记录执行期间的峰值
        self.function_peak_memory: Dict[int, float] = {}  # node_id -> peak_memory_during_execution
        
    def start_tracing(self):
        """开始追踪"""
        self.tracing_active = True
        # 尝试启用opcode级别的追踪
        import threading
        frame = threading.current_thread().ident
        try:
            sys.settrace(self._trace_function)
            # Python 3.7+ 支持设置 trace_opcodes
            if hasattr(sys, 'gettrace'):
                current_thread = threading.current_thread()
                if hasattr(current_thread, 'trace_opcodes'):
                    current_thread.trace_opcodes = True
        except:
            sys.settrace(self._trace_function)
            
        if self.memory_profiler:
            self.memory_profiler.start_tracking()
        logger.info("增强追踪已启动")
    
    def stop_tracing(self):
        """停止追踪"""
        self.tracing_active = False
        sys.settrace(None)
        if self.memory_profiler:
            self.memory_profiler.stop_tracking()
        logger.info("增强追踪已停止")
    
    def _trace_function(self, frame, event, arg):
        """追踪函数（增强版 - 修复栈破坏和性能问题）"""
        if not self.tracing_active:
            return None

        # 深度检查：如果超出深度，直接停止追踪此分支
        if self.current_depth >= self.max_depth:
            return None

        if event == 'call':
            func_name = frame.f_code.co_name
            module_name = frame.f_globals.get('__name__', 'unknown')

            # 1. 函数级别追踪模式检查
            if self.function_tracing_mode:
                # 只有目标函数才返回 tracer，否则返回 None 跳过
                if self._should_trace_in_function_mode(func_name):
                    return self._handle_function_tracing(frame, func_name, module_name)
                return None

            # 2. 统一过滤检查
            # [核心修复] 如果应该过滤，返回 None！
            # 这告诉 Python解释器：不要追踪这个函数内部的任何事件(line/return)
            # 这样既解决了栈不平衡问题，又极大提升了性能
            if self.function_filter.should_filter(func_name, module_name):
                return None

            # 进入函数
            self.current_depth += 1
            context = self._enter_function(func_name, module_name, frame)

            # 追踪局部变量
            self._track_local_variables(frame, "enter")

            # 返回自身以继续追踪此作用域内的事件
            return self._trace_function

        elif event == 'return':
            # 因为过滤的函数已经返回 None，能进到这里的肯定是未过滤的函数
            # 所以这里的逻辑可以大大简化，不再需要防御性检查
            self.current_depth -= 1

            if self.function_tracing_mode:
                self._handle_function_return_tracing()

            if self.node_stack:
                # 追踪返回值
                if arg is not None:
                    self._track_return_value(arg)

                # 追踪局部变量变化
                self._track_local_variables(frame, "exit")

                # 退出函数
                self._exit_function(frame)

        elif event == 'line':
            # 因为过滤的函数不会触发 line 事件，这里不需要再做过滤检查
            if self.node_stack:
                lineno = frame.f_lineno
                filename = frame.f_code.co_filename

                # 1. 处理上一行的执行结果
                self._process_previous_line(frame)

                # 2. 准备当前行的AST分析
                import linecache
                source_line = linecache.getline(filename, lineno).strip()

                self._prepare_current_line(frame, lineno, source_line)

        elif event == 'opcode':
            # 同理，过滤函数不触发 opcode
            if self.node_stack and self.memory_profiler:
                current_memory = self.memory_profiler._get_current_memory()
                for node in self.node_stack:
                    if node.node_id in self.function_peak_memory:
                        self.function_peak_memory[node.node_id] = max(
                            self.function_peak_memory[node.node_id],
                            current_memory
                        )

        return self._trace_function

    def _should_trace_in_function_mode(self, func_name: str) -> bool:
        """[辅助方法] 用于函数模式判断"""
        if func_name in self.target_functions:
            return True
        if self.in_target_function:
            return self.target_call_depth < self.function_trace_depth
        return False
    
    def _add_edge(self, from_id: int, to_id: int, edge_type: str):
        """统一的边创建方法 - 直接使用DAG"""
        # 检查边是否已存在
        existing_edge = next((e for e in self.dag_builder.dag.edges
                             if e[0] == from_id and e[1] == to_id and e[2] == edge_type), None)
        if existing_edge:
            return

        # 使用DAG的add_edge方法
        self.dag_builder.dag.add_edge(from_id, to_id, edge_type)
        logger.debug(f"创建边: {from_id} --{edge_type}--> {to_id}")
    
    def _track_local_variables(self, frame, event_type: str, lineno: int = 0):
        """追踪局部变量变化

        Args:
            frame: 当前栈帧
            event_type: 事件类型
            lineno: 导致变量变化的源代码行号（来自 previous_line_info）
                   注意：不能使用 frame.f_lineno，因为 'line' 事件触发时
                   frame.f_lineno 指向的是即将执行的下一行，而不是刚执行完的那行
        """
        try:
            if not self.context_key_manager.call_stack:
                return

            current_context = "->".join(self.context_key_manager.call_stack)

            # [新增] 噪音变量忽略列表 - 过滤掉无意义的辅助变量
            IGNORED_VARS = {'self', 'cls', 'args', 'kwargs', '__class__', '__dict__', '__doc__'}

            # 检测局部变量
            for var_name, var_value in frame.f_locals.items():
                # 跳过内部变量（以下划线开头）
                if var_name.startswith('_'):
                    continue

                # [新增] 跳过噪音变量
                if var_name in IGNORED_VARS:
                    continue

                # [新增] 过滤掉模块对象、函数对象和类对象，我们只关心数据
                if inspect.ismodule(var_value) or inspect.isfunction(var_value) or inspect.isclass(var_value):
                    continue

                # [新增] 过滤掉方法对象
                if inspect.ismethod(var_value) or inspect.isbuiltin(var_value):
                    continue

                # 创建变量的Context Key
                var_context_key = f"{current_context}->{var_name}"

                # 创建当前变量快照
                current_snapshot = VariableSnapshot.create_snapshot(var_value)

                # 检查变量是否发生变化
                if var_context_key in self.variable_snapshots:
                    old_snapshot = self.variable_snapshots[var_context_key]
                    if current_snapshot.has_changed(old_snapshot):
                        # 变量值发生变化，创建新版本节点
                        self._create_variable_node(var_name, var_value, current_context,
                                                   increment_version=True, lineno=lineno)
                else:
                    # 新变量，创建首个版本节点
                    self._create_variable_node(var_name, var_value, current_context,
                                               increment_version=False, lineno=lineno)

                # 更新快照
                self.variable_snapshots[var_context_key] = current_snapshot

        except Exception as e:
            logger.debug(f"变量追踪失败: {e}")
    
    def _create_variable_node(self, var_name: str, var_value: Any, context: str,
                               increment_version: bool = True, lineno: int = 0) -> EnhancedNode:
        """创建变量节点并建立函数创建边

        Args:
            var_name: 变量名
            var_value: 变量值
            context: 上下文路径
            increment_version: 是否自增版本号
            lineno: 创建此变量的源代码行号 (用于区分自引用赋值)
        """
        try:
            # 获取版本号
            version = 1
            if increment_version:
                version = self.context_key_manager.get_next_variable_version(var_name, context)
            else:
                self.context_key_manager.variable_versions[f"{context}->{var_name}"] = 1

            # 创建变量快照
            snapshot = VariableSnapshot.create_snapshot(var_value)

            # 创建DAG变量节点
            var_node = self.dag_builder.dag.add_node(
                name=var_name,
                node_type="variable",
                module=context.split("->")[0] if "->" in context else "global",
                context=f"{context}->{var_name}#{version}",
                version=version
            )

            # 设置性能数据和变量快照
            var_node.performance['start_time'] = time.time()
            var_node.performance['end_time'] = time.time()
            var_node.performance['execution_time_ms'] = 0
            var_node.performance['memory_before_mb'] = 0
            var_node.performance['memory_after_mb'] = 0
            var_node.performance['memory_allocated_mb'] = 0
            var_node.performance['peak_memory_mb'] = 0
            var_node.attributes['variable_snapshot'] = snapshot
            # 记录创建行号 (用于 AST 分析器区分自引用赋值中的变量版本)
            var_node.attributes['created_at_lineno'] = lineno

            # 添加到变量节点映射（直接存储node对象）
            if var_name not in self.variable_nodes:
                self.variable_nodes[var_name] = []
            self.variable_nodes[var_name].append(var_node)
            
            # 关键：创建函数到变量的边
            if self.node_stack:
                func_node = self.node_stack[-1]
                edge_type = 'creates' if version == 1 else 'modifies'
                self._add_edge(func_node.node_id, var_node.node_id, edge_type)
            
            # 检查是否有待连接的返回值
            self._check_and_connect_return_values(var_node)
            
            logger.debug(f"创建变量节点: {var_name} v{version} in {context}")
            return var_node
            
        except Exception as e:
            logger.error(f"创建变量节点失败: {e}")
            return None
    
    def _track_return_value(self, return_value):
        """追踪函数返回值"""
        if self.node_stack and return_value is not None:
            current_func_node = self.node_stack[-1]
            current_context = "->".join(self.context_key_manager.call_stack)
            
            # 解析return语句中的实际变量
            return_var_nodes = self._parse_return_variables(current_context, return_value)
            
            if return_var_nodes:
                # 存储到基于context key的返回值映射
                self.pending_returns[current_context] = return_var_nodes
                logger.debug(f"函数 {current_func_node.name} 返回变量: {[node.name for node in return_var_nodes]}")
                
                # 兼容旧代码
                self.latest_return_value_node = return_var_nodes[0] if return_var_nodes else None
    
    def _parse_return_variables(self, context: str, return_value) -> List[EnhancedNode]:
        """解析return语句中的实际变量节点（使用ID匹配 - 修正版）"""
        return_nodes = []

        logger.debug(f"[返回值解析] 上下文: {context}, 返回值类型: {type(return_value)}")

        # 处理单个返回值
        if not isinstance(return_value, (tuple, list)):
            return_values = [return_value]
        else:
            return_values = list(return_value)

        # 对每个返回值使用 ID 匹配
        for idx, ret_val in enumerate(return_values):
            ret_val_id = id(ret_val)
            found_node = None

            # 1. 尝试在已知变量中查找
            # 遍历所有变量节点，通过 value_id 匹配
            for var_name, var_node_list in self.variable_nodes.items():
                for node in reversed(var_node_list):  # 倒序遍历，优先匹配最新版本
                    # 检查是否在当前上下文中
                    if not node.context.startswith(context):
                        continue

                    # 检查 value_id 是否匹配（必须确保节点有 snapshot 属性）
                    if (hasattr(node, 'attributes') and
                        node.attributes.get('variable_snapshot') and
                        node.attributes['variable_snapshot'].value_id == ret_val_id):
                        found_node = node
                        logger.debug(f"[返回值解析] 通过ID匹配找到变量: {node.name} (id={node.node_id})")
                        break

                if found_node:
                    break

            # 2. 如果找不到匹配的节点，创建临时节点（表示返回表达式）
            if not found_node:
                logger.debug(f"[返回值解析] 未找到匹配变量，创建临时节点 <return_expression>")

                # 使用工厂方法统一创建快照
                snapshot = VariableSnapshot.create_snapshot(ret_val)

                # 使用 DAG Builder 的标准接口创建节点
                temp_name = "<return_expression>"
                temp_node = self.dag_builder.dag.add_node(
                    name=temp_name,
                    node_type='variable',
                    module='anonymous',
                    context=f"{context}->{temp_name}",
                    version=0
                )

                # 填充属性
                temp_node.attributes['variable_snapshot'] = snapshot
                temp_node.attributes['is_temporary'] = True
                temp_node.performance['start_time'] = time.time()
                temp_node.performance['end_time'] = time.time()  # 瞬时节点

                # 添加到 variable_nodes 索引中（确保键存在）
                if temp_name not in self.variable_nodes:
                    self.variable_nodes[temp_name] = []
                self.variable_nodes[temp_name].append(temp_node)

                found_node = temp_node

            return_nodes.append(found_node)

        logger.debug(f"[返回值解析] 解析结果: {[n.name for n in return_nodes]}")
        return return_nodes
    
    def _check_and_connect_return_values(self, target_var_node: EnhancedNode):
        """检查并连接待处理的返回值到目标变量（使用ID匹配 - 支持多返回值解包）"""
        if not self.pending_returns:
            return

        # 1. 安全获取目标变量的 value_id
        if not (hasattr(target_var_node, 'attributes') and
                target_var_node.attributes.get('variable_snapshot')):
            return

        target_value_id = target_var_node.attributes['variable_snapshot'].value_id

        matched_context_key = None
        matched_node = None

        # 2. 遍历查找匹配 (ID 强匹配)
        # 注意：这里一旦找到匹配就 break，因为一个变量只能有一个来源
        for return_context, return_var_nodes in self.pending_returns.items():
            for return_node in return_var_nodes:
                if not (hasattr(return_node, 'attributes') and
                        return_node.attributes.get('variable_snapshot')):
                    continue

                return_value_id = return_node.attributes['variable_snapshot'].value_id

                if return_value_id == target_value_id:
                    matched_context_key = return_context
                    matched_node = return_node

                    logger.debug(
                        f"[连接检查] ID匹配成功: {return_node.name} -> {target_var_node.name} "
                        f"(ID={target_value_id})"
                    )
                    break  # 找到源头，停止内层循环

            if matched_node:
                break  # 找到源头，停止外层循环

        # 3. 连边与清理
        if matched_node and matched_context_key:
            # 建立连边
            self._add_edge(matched_node.node_id, target_var_node.node_id, 'flows_to')

            # 精细化清理：只移除已使用的那个节点，而不是删除整个 Context
            # 这样可以支持 a, b = func() 的情况
            node_list = self.pending_returns[matched_context_key]
            if matched_node in node_list:
                node_list.remove(matched_node)

            # 只有当该 Context 下所有返回值都被认领后，才彻底删除 Key
            if not node_list:
                del self.pending_returns[matched_context_key]
                logger.debug(f"[连接检查] Context已空，清理: {matched_context_key}")
    
    def _analyze_current_line(self, frame):
        """分析当前行的源码，检测赋值语句并创建运算符节点"""
        try:
            if not self.context_key_manager.call_stack:
                return
            
            # 获取当前行号和文件名
            lineno = frame.f_lineno
            filename = frame.f_code.co_filename
            
            # 创建唯一标识符防止重复分析
            line_key = f"{filename}:{lineno}"
            if line_key in self.analyzed_lines:
                return
            self.analyzed_lines.add(line_key)
            
            # 获取函数名
            func_name = frame.f_code.co_name
            if func_name in self.SKIP_FUNCTIONS:
                return
            
            # 构建当前函数上下文
            current_context = "->".join(self.context_key_manager.call_stack)
            
            # 获取当前行的源码
            try:
                import linecache
                source_line = linecache.getline(filename, lineno).strip()
                if not source_line:
                    return
                
                # 解析当前行为AST
                try:
                    # 解析单行代码
                    parsed = ast.parse(source_line)

                    # 创建AST分析器并分析 (传入 frame 用于运行时名字解析)
                    analyzer = SimpleASTAnalyzer(self, current_context, frame)
                    # [关键] 设置当前行号，用于防环机制
                    analyzer.current_lineno = lineno
                    analyzer.visit(parsed)

                    logger.debug(f"AST分析第{lineno}行: {source_line}")
                    
                except SyntaxError:
                    # 某些行可能无法单独解析，忽略
                    pass
                    
            except Exception as e:
                logger.debug(f"获取源码失败 {filename}:{lineno}: {e}")
                
        except Exception as e:
            logger.debug(f"行级AST分析失败: {e}")
    
    def _process_previous_line(self, frame):
        """处理上一行的执行结果

        关键时序：此时上一行已经执行完成，Runtime 函数节点已创建，
        所以 AST 分析可以正确找到匹配的函数节点，避免创建重复节点。
        """
        try:
            func_name = frame.f_code.co_name
            func_key = f"{func_name}_{id(frame.f_code)}"

            if func_key not in self.previous_line_info:
                return

            prev_info = self.previous_line_info[func_key]
            source_line = prev_info.get('source_line', '')

            prev_lineno = prev_info.get('lineno', 0)
            logger.debug(f"处理上一行 第{prev_lineno}行: {source_line}")

            # 1. 先进行变量变化检测（此时上一行已执行完成）
            # 传入上一行的行号，而不是 frame.f_lineno（那是下一行的行号）
            self._track_local_variables(frame, "line", lineno=prev_lineno)

            # 2. [核心修复] 在上一行执行完成后再进行 AST 分析
            # 此时 Runtime 函数节点已经创建，AST 可以正确匹配到它们
            if source_line.strip():
                try:
                    current_context = "->".join(self.context_key_manager.call_stack)
                    parsed = ast.parse(source_line)
                    analyzer = SimpleASTAnalyzer(self, current_context, frame)
                    # 设置 AST 分析器的当前行号
                    analyzer.current_lineno = prev_lineno
                    analyzer.visit(parsed)
                except SyntaxError:
                    # 某些行可能无法单独解析，忽略
                    pass

            # 3. 处理待处理的运算符连边（如果有）
            if prev_info.get('pending_operators'):
                self._process_pending_operators(prev_info['pending_operators'], frame)

        except Exception as e:
            logger.debug(f"处理上一行失败: {e}")
    
    def _prepare_current_line(self, frame, lineno: int, source_line: str):
        """准备当前行信息，供下一行处理时使用

        注意：不在这里进行 AST 分析！AST 分析移至 _process_previous_line，
        在当前行执行完成后再分析，确保 Runtime 函数节点已创建。
        """
        try:
            func_name = frame.f_code.co_name
            func_key = f"{func_name}_{id(frame.f_code)}"

            # 只存储源码信息，不进行 AST 分析
            self.previous_line_info[func_key] = {
                'lineno': lineno,
                'source_line': source_line,
                'pending_operators': []  # 延迟到 _process_previous_line 处理
            }

            logger.debug(f"准备当前行 第{lineno}行: {source_line[:50]}...")

        except Exception as e:
            logger.debug(f"准备当前行失败: {e}")
    
    def _analyze_line_ast(self, source_line: str, lineno: int, frame) -> List[Dict]:
        """分析单行AST，返回待处理的运算符信息"""
        try:
            if not source_line.strip():
                return []
            
            # 构建当前函数上下文
            current_context = "->".join(self.context_key_manager.call_stack)
            
            # 解析AST
            try:
                parsed = ast.parse(source_line)
                analyzer = SimpleASTAnalyzer(self, current_context, frame)
                # [关键] 设置当前行号，用于防环机制
                analyzer.current_lineno = lineno
                analyzer.visit(parsed)

                # 收集待处理的运算符信息
                pending_operators = []
                for op_node in analyzer.operator_nodes:
                    # 查找这个运算符对应的目标变量
                    for node in parsed.body:
                        if isinstance(node, ast.Assign) and len(node.targets) == 1:
                            if hasattr(node.targets[0], 'id'):
                                target_var = node.targets[0].id
                                pending_operators.append({
                                    'operator_node': op_node,
                                    'target_var': target_var,
                                    'lineno': lineno
                                })
                
                return pending_operators
                
            except SyntaxError:
                return []
                
        except Exception as e:
            logger.debug(f"分析行AST失败: {e}")
            return []
    
    def _process_pending_operators(self, pending_operators: List[Dict], frame):
        """处理待处理的运算符连边"""
        try:
            for op_info in pending_operators:
                operator_node = op_info['operator_node']
                target_var = op_info['target_var']
                
                # 查找目标变量节点（此时应该已经创建）
                result_var_node = self._find_variable_node_in_frame(target_var, frame)
                if result_var_node and operator_node:
                    # 创建 运算符 -> 结果变量 的边
                    self._add_edge(operator_node.node_id, result_var_node.node_id, 'produces')
                    logger.debug(f"连边成功: {operator_node.name}(ID:{operator_node.node_id}) -> {result_var_node.name}(ID:{result_var_node.node_id})")
                else:
                    logger.debug(f"连边失败: 找不到变量 {target_var}")
                    
        except Exception as e:
            logger.debug(f"处理待处理运算符失败: {e}")
    
    def _find_variable_node_in_frame(self, var_name: str, frame) -> EnhancedNode:
        """在当前帧中查找变量节点"""
        if var_name not in self.variable_nodes:
            return None
        
        # 获取当前上下文
        current_context = "->".join(self.context_key_manager.call_stack)
        
        # 查找最匹配的变量节点
        for node in reversed(self.variable_nodes[var_name]):
            if node.context.startswith(current_context):
                return node
        return None
    
    def _process_final_line(self, frame):
        """处理函数的最后一行"""
        try:
            func_name = frame.f_code.co_name
            func_key = f"{func_name}_{id(frame.f_code)}"
            
            if func_key in self.previous_line_info:
                logger.debug(f"处理最后一行: 函数 {func_name} 退出")
                
                # 最后进行一次变量检测
                self._track_local_variables(frame, "exit")
                
                # 处理最后一行的待处理运算符
                prev_info = self.previous_line_info[func_key]
                if prev_info.get('pending_operators'):
                    self._process_pending_operators(prev_info['pending_operators'], frame)
                
                # 清理这个函数的信息
                del self.previous_line_info[func_key]
                
        except Exception as e:
            logger.debug(f"处理最后一行失败: {e}")
    
    def _handle_function_parameters(self, frame, func_node: EnhancedNode):
        """处理函数参数传递，创建参数变量节点和边（使用ID匹配）"""
        try:
            # 获取函数参数名称
            code = frame.f_code
            param_names = code.co_varnames[:code.co_argcount]

            if not param_names:
                return

            # 当前函数上下文
            current_context = "->".join(self.context_key_manager.call_stack)

            # 如果有父函数上下文，寻找对应的变量节点
            if len(self.context_key_manager.call_stack) > 1:
                parent_context = "->".join(self.context_key_manager.call_stack[:-1])

                for param_name in param_names:
                    if param_name in frame.f_locals:
                        param_value = frame.f_locals[param_name]

                        # 关键修复：先通过 id(param_value) 查找父变量，再创建参数节点
                        param_id = id(param_value)
                        parent_var_node = None

                        # 遍历所有变量节点，通过 value_id 匹配（倒序遍历优先匹配最新版本）
                        for var_name, node_list in self.variable_nodes.items():
                            for node in reversed(node_list):  # 倒序遍历，优先匹配最新版本
                                # 检查 value_id 是否匹配
                                if (hasattr(node, 'attributes') and
                                    'variable_snapshot' in node.attributes and
                                    node.attributes['variable_snapshot'].value_id == param_id):

                                    # 优先匹配：父函数的局部变量
                                    # Context格式：parent_context->var_name#version
                                    is_parent_local = False
                                    if node.context.startswith(parent_context + "->"):
                                        remainder = node.context[len(parent_context) + 2:]
                                        # 检查是否是直接子变量（不包含更深的嵌套）
                                        var_part = remainder.split("#")[0]
                                        if "->" not in var_part:
                                            is_parent_local = True

                                    # 备选匹配：全局变量（context不包含 "->" ）
                                    is_global = "->" not in node.context

                                    if is_parent_local:
                                        parent_var_node = node
                                        break  # 找到父局部变量，优先使用
                                    elif is_global and parent_var_node is None:
                                        parent_var_node = node  # 备选：全局变量

                            if parent_var_node and parent_var_node.context.startswith(parent_context + "->"):
                                break  # 已找到父局部变量，停止搜索

                        # 现在创建参数变量节点（避免在查找前创建导致自环）
                        param_var_node = self._create_variable_node(
                            param_name, param_value, current_context, increment_version=False
                        )

                        if parent_var_node and param_var_node:
                            # 创建参数传递边：父变量 -> 子参数
                            self._add_edge(parent_var_node.node_id, param_var_node.node_id, 'parameter')
                            logger.debug(
                                f"创建参数传递边: {parent_var_node.name} (id={parent_var_node.node_id}) "
                                f"-> {param_var_node.name} (id={param_var_node.node_id})"
                            )

        except Exception as e:
            logger.debug(f"参数处理失败: {e}")
    
    def _find_parent_variable_node(self, var_name: str, parent_context: str) -> EnhancedNode:
        """在父上下文中查找变量节点"""
        if var_name not in self.variable_nodes:
            return None
        
        # 查找最匹配的变量节点
        for node in reversed(self.variable_nodes[var_name]):
            if parent_context in node.context:
                return node
        return None
    
    def _analyze_function_ast(self, func_name: str, frame) -> List[EnhancedNode]:
        """分析函数的AST来检测运算符节点"""
        try:
            # 获取函数源码
            func_obj = frame.f_globals.get(func_name)
            if not func_obj or not hasattr(func_obj, '__code__'):
                return []
            
            # 尝试获取源码
            try:
                source = inspect.getsource(func_obj)
                func_ast = ast.parse(source)
            except (OSError, TypeError):
                # 无法获取源码，跳过AST分析
                return []
            
            # 分析AST
            current_context = "->".join(self.context_key_manager.call_stack)
            analyzer = SimpleASTAnalyzer(self, current_context, frame)
            # [关键] 设置当前行号，用于防环机制
            # 对于整函数分析，使用 frame 的当前行号
            analyzer.current_lineno = frame.f_lineno if frame else 0
            analyzer.visit(func_ast)

            return analyzer.operator_nodes
            
        except Exception as e:
            logger.debug(f"AST分析失败: {e}")
            return []
    
    def _enter_function(self, func_name: str, module: str, frame) -> str:
        """进入函数"""
        # 使用Context Key管理器进入函数
        context = self.context_key_manager.enter_function(func_name)
        
        # 获取内存信息
        memory_before = 0
        if self.memory_profiler:
            self.memory_profiler.enter_function(func_name, module)
            memory_before = self.memory_profiler._get_current_memory()
        
        self.context_stack.append(context)

        # 创建DAG节点
        node = self.dag_builder.dag.add_node(
            name=func_name,
            node_type="function",
            module=module,
            context=context,
            version=0
        )

        # 设置详细性能数据
        node.performance['start_time'] = time.time()
        node.performance['end_time'] = 0
        node.performance['execution_time_ms'] = 0
        node.performance['memory_before_mb'] = memory_before
        node.performance['memory_after_mb'] = 0
        node.performance['memory_allocated_mb'] = 0
        node.performance['peak_memory_mb'] = memory_before
        self.node_stack.append(node)
        self.context_map[context] = node.node_id
        
        # 初始化该函数的峰值内存为当前内存
        self.function_peak_memory[node.node_id] = memory_before
        
        # 记录边关系 - 使用统一的边创建方法
        if self.call_stack:
            parent_id = self.call_stack[-1]["id"]
            self._add_edge(parent_id, node.node_id, "calls")
        
        # 处理函数参数传递边
        self._handle_function_parameters(frame, node)
        
        # 更新调用栈
        self.call_stack.append({
            "id": node.node_id,
            "name": func_name,
            "context": context
        })
        
        return context
    
    def _exit_function(self, frame):
        """退出函数"""
        if not self.node_stack:
            return
        
        node = self.node_stack.pop()
        node.performance['end_time'] = time.time()
        node.performance['execution_time_ms'] = (node.performance['end_time'] - node.performance['start_time']) * 1000

        # 获取内存信息
        if self.memory_profiler:
            node.performance['memory_after_mb'] = self.memory_profiler._get_current_memory()
            node.performance['memory_allocated_mb'] = max(0, node.performance['memory_after_mb'] - node.performance['memory_before_mb'])

            # 使用追踪期间的峰值内存（包括所有子函数）
            if node.node_id in self.function_peak_memory:
                # 确保包含退出时的内存
                self.function_peak_memory[node.node_id] = max(
                    self.function_peak_memory[node.node_id],
                    node.performance['memory_after_mb']
                )
                node.performance['peak_memory_mb'] = self.function_peak_memory[node.node_id]
            else:
                node.performance['peak_memory_mb'] = node.performance['memory_after_mb']
            
            self.memory_profiler.exit_function(node.context)
        
        # 处理最后一行（函数退出时）
        self._process_final_line(frame)
        
        # 使用Context Key管理器退出函数
        self.context_key_manager.exit_function()
        
        # 更新上下文栈
        if self.context_stack:
            self.context_stack.pop()
        if self.call_stack:
            self.call_stack.pop()

    def export_dataflow_graph(self) -> Dict:
        """导出数据流图 - 与原始接口兼容"""
        return self.get_enhanced_dataflow()
    
    def get_enhanced_dataflow(self) -> Dict:
        """获取增强的数据流图"""
        # 构建节点数据
        nodes = []
        for node in self.dag_builder.dag.nodes.values():
            node_data = {
                "id": node.node_id,
                "name": node.name,
                "type": node.node_type,
                "module": node.module,
                "context": node.context,
                "performance": {
                    "execution_time_ms": node.performance['execution_time_ms'],
                    "memory_before_mb": node.performance['memory_before_mb'],
                    "memory_after_mb": node.performance['memory_after_mb'],
                    "memory_allocated_mb": node.performance['memory_allocated_mb'],
                    "peak_memory_mb": node.performance['peak_memory_mb'],
                    # 为了兼容optimizer，提供memory_usage字段（单位：字节）
                    "memory_usage": int(node.performance['peak_memory_mb'] * 1024 * 1024)
                }
            }
            
            # 添加变量特定信息
            if node.node_type == "variable" and node.attributes.get('variable_snapshot'):
                snapshot = node.attributes['variable_snapshot']
                
                node_data["variable_info"] = {
                    "version": node.version,
                    "actual_value": snapshot.actual_value,
                    "shape": snapshot.shape,
                    "dtype": snapshot.dtype,
                    "device": snapshot.device,
                    "size_mb": snapshot.size_mb
                }
            
            # 添加连边信息
            input_edges = []
            output_edges = []
            
            for edge in self.dag_builder.dag.edges:
                source_id, target_id, edge_type = edge  # 解包元组

                if target_id == node.node_id:
                    # 找到源节点名称
                    source_node = next((n for n in self.dag_builder.dag.nodes.values() if n.node_id == source_id), None)
                    if source_node:
                        input_edges.append({
                            "type": edge_type,
                            "from_id": source_id,
                            "from_name": source_node.name
                        })

                if source_id == node.node_id:
                    # 找到目标节点名称
                    target_node = next((n for n in self.dag_builder.dag.nodes.values() if n.node_id == target_id), None)
                    if target_node:
                        output_edges.append({
                            "type": edge_type,
                            "to_id": target_id,
                            "to_name": target_node.name
                        })
            
            if input_edges:
                node_data["input_edges"] = input_edges
            if output_edges:
                node_data["output_edges"] = output_edges
            
            nodes.append(node_data)
        
        # 获取内存报告
        memory_report = {}
        if self.memory_profiler:
            memory_report = self.memory_profiler.get_memory_report()
        
        # 获取火焰图数据
        flame_graph = {}
        if self.memory_profiler:
            flame_graph = self.memory_profiler.generate_flame_graph_data()
        
        return {
            "nodes": nodes,
            "edges": self.dag_builder.dag.edges,
            "memory_report": memory_report,
            "flame_graph": flame_graph,
            "statistics": {
                "total_nodes": len(self.dag_builder.dag.nodes.values()),
                "function_nodes": sum(1 for n in self.dag_builder.dag.nodes.values() if n.node_type == "function"),
                "variable_nodes": sum(1 for n in self.dag_builder.dag.nodes.values() if n.node_type == "variable"),
                "total_variables_tracked": len(self.variable_nodes),
                "peak_memory_mb": self.memory_profiler.peak_memory_mb if self.memory_profiler else 0
            }
        }
    
    def print_detailed_nodes(self):
        """详细打印所有节点信息"""
        print("=" * 80)
        print("详细数据流图节点信息")
        print("=" * 80)
        
        # 按节点类型分组
        function_nodes = [n for n in self.dag_builder.dag.nodes.values() if n.node_type == "function"]
        variable_nodes = [n for n in self.dag_builder.dag.nodes.values() if n.node_type == "variable"]
        operator_nodes = [n for n in self.dag_builder.dag.nodes.values() if n.node_type == "operator"]
        
        # 构建边的查找表
        edges_from = {}
        edges_to = {}
        for edge in self.dag_builder.dag.edges:
            from_id, to_id, edge_type = edge  # 解包元组

            if from_id not in edges_from:
                edges_from[from_id] = []
            edges_from[from_id].append({"to": to_id, "type": edge_type})

            if to_id not in edges_to:
                edges_to[to_id] = []
            edges_to[to_id].append({"from": from_id, "type": edge_type})
        
        # 打印函数节点
        if function_nodes:
            print(f"\n函数节点 ({len(function_nodes)}个):")
            print("-" * 60)
            for node in function_nodes:
                self._print_single_node(node, edges_from, edges_to)
        
        # 打印变量节点
        if variable_nodes:
            print(f"\n变量节点 ({len(variable_nodes)}个):")
            print("-" * 60)
            for node in variable_nodes:
                self._print_single_node(node, edges_from, edges_to)
        
        # 打印运算符节点
        if operator_nodes:
            print(f"\n运算符节点 ({len(operator_nodes)}个):")
            print("-" * 60)
            for node in operator_nodes:
                self._print_single_node(node, edges_from, edges_to)
        
        # 打印边统计
        print(f"\n连边统计:")
        print("-" * 30)
        edge_types = {}
        for edge in self.dag_builder.dag.edges:
            _, _, edge_type = edge  # 解包元组获取edge_type
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        for edge_type, count in edge_types.items():
            print(f"  {edge_type}: {count}条")
        print(f"  总计: {len(self.dag_builder.dag.edges)}条边")
    
    def set_filter_level(self, level: FilterLevel):
        """设置过滤级别"""
        self.function_filter.set_filter_level(level)
        logger.info(f"过滤级别已设置为: {level.name}")
    
    def add_custom_filter(self, pattern: str):
        """添加自定义过滤规则"""
        self.function_filter.add_custom_filter(pattern)
        logger.info(f"已添加过滤规则: {pattern}")
    
    def add_custom_protection(self, pattern: str):
        """添加函数保护规则"""
        self.function_filter.add_custom_protection(pattern)
        logger.info(f"已添加保护规则: {pattern}")
    
    def get_filter_stats(self) -> dict:
        """获取过滤统计信息"""
        return self.function_filter.get_stats()
    
    def _print_single_node(self, node, edges_from, edges_to):
        """打印单个节点的详细信息"""
        print(f"\n节点 ID: {node.node_id}")
        print(f"  名称: {node.name}")
        print(f"  类型: {node.node_type}")
        print(f"  模块: {node.module}")
        print(f"  上下文: {node.context}")
        
        if node.node_type == "variable":
            print(f"  版本: {node.version}")
            if node.shape:
                print(f"  形状: {node.shape}")
            if node.dtype:
                print(f"  数据类型: {node.dtype}")
            if node.device:
                print(f"  设备: {node.device}")
            if node.size_mb is not None:
                print(f"  大小: {node.size_mb:.3f} MB")
        
        # 性能数据
        if node.performance['execution_time_ms'] > 0:
            print(f"  执行时间: {node.performance['execution_time_ms']:.3f} ms")
        if node.performance['memory_allocated_mb'] > 0:
            print(f"  内存分配: {node.performance['memory_allocated_mb']:.3f} MB")
        if node.performance['peak_memory_mb'] > 0:
            print(f"  峰值内存: {node.performance['peak_memory_mb']:.3f} MB")
        
        # 输入边
        incoming_edges = edges_to.get(node.node_id, [])
        if incoming_edges:
            print(f"  输入边 ({len(incoming_edges)}条):")
            for edge in incoming_edges:
                from_node = next((n for n in self.dag_builder.dag.nodes.values() if n.node_id == edge["from"]), None)
                if from_node:
                    print(f"    <- [{edge['type']}] {from_node.name} (ID: {from_node.node_id})")
        
        # 输出边
        outgoing_edges = edges_from.get(node.node_id, [])
        if outgoing_edges:
            print(f"  输出边 ({len(outgoing_edges)}条):")
            for edge in outgoing_edges:
                to_node = next((n for n in self.dag_builder.dag.nodes.values() if n.node_id == edge["to"]), None)
                if to_node:
                    print(f"    -> [{edge['type']}] {to_node.name} (ID: {to_node.node_id})")
    
    def _handle_function_tracing(self, frame, func_name: str, module_name: str):
        """处理函数级别追踪模式"""
        # 检查是否是目标函数
        if func_name in self.target_functions:
            # 进入目标函数
            self.in_target_function = True
            self.target_call_depth = 0
            logger.debug(f"进入目标函数: {func_name}")
        elif self.in_target_function:
            # 在目标函数内部，检查深度
            self.target_call_depth += 1
            if self.target_call_depth > self.function_trace_depth:
                # 超出函数级追踪深度
                self.current_depth -= 1
                return self._trace_function
        else:
            # 不在目标函数内部，不追踪
            self.current_depth -= 1
            return self._trace_function
        
        # 应用过滤规则（即使在函数级追踪中，仍需要过滤系统函数）
        if self.function_filter.should_filter(func_name, module_name):
            self.current_depth -= 1
            if self.in_target_function and self.target_call_depth > 0:
                self.target_call_depth -= 1
            return self._trace_function
        
        # 进入函数追踪
        context = self._enter_function(func_name, module_name, frame)
        self._track_local_variables(frame, "enter")
        
        return self._trace_function
    
    def _handle_function_return_tracing(self):
        """处理函数级别追踪的返回事件"""
        if self.in_target_function:
            if self.target_call_depth > 0:
                self.target_call_depth -= 1
            else:
                # 退出目标函数
                self.in_target_function = False
                logger.debug(f"退出目标函数")
    
    def trace_function(self, depth: int = None):
        """装饰器：标记函数进行追踪
        
        Args:
            depth: 从标记函数开始的追踪深度，None表示使用默认深度
            
        Usage:
            @tracer.trace_function(depth=5)
            def __next__(self):
                pass
        """
        def decorator(func):
            # 添加到目标函数集合
            self.target_functions.add(func.__name__)
            if depth is not None:
                self.function_trace_depth = depth
            
            # 启用函数级别追踪模式
            self.function_tracing_mode = True
            
            logger.info(f"标记函数进行追踪: {func.__name__}, 深度: {self.function_trace_depth}")
            
            return func
        return decorator
    
    @contextmanager
    def tracing_context(self):
        """追踪上下文管理器"""
        if self.function_tracing_mode:
            raise ValueError("函数级别追踪模式已启用，不能同时使用 tracing_context()。请选择其中一种追踪模式。")
        
        self.start_tracing()
        try:
            yield self
        finally:
            self.stop_tracing()
    
    @contextmanager  
    def function_tracing_context(self):
        """函数级别追踪上下文管理器"""
        if not self.function_tracing_mode:
            raise ValueError("请先使用 @trace_function 装饰器标记目标函数")
            
        self.start_tracing()
        try:
            yield self
        finally:
            self.stop_tracing()
    
    def export_visualization(self, output_path: str) -> bool:
        """便捷的SVG可视化导出函数
        
        Args:
            output_path: SVG输出路径
            
        Returns:
            bool: 是否导出成功
        """
        try:
            from visualization.graphviz_visualizer import create_dataflow_visualization
            success = create_dataflow_visualization(self, output_path)
            if success:
                logger.info(f"可视化已导出: {output_path}")
            else:
                logger.warning("可视化导出失败")
            return success
        except Exception as e:
            logger.error(f"可视化导出异常: {e}")
            return False