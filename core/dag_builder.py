"""
DAG构建器 - 专门负责构建执行DAG
从追踪数据中构建有向无环图
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DAGNode:
    """DAG节点数据结构"""

    def __init__(self, node_id: int, name: str, node_type: str = "function_call",
                 module: str = "", context: str = ""):
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.context_id = f"node_{node_id}"

        # 额外字段用于tracer
        self.module = module
        self.context = context
        self.version = 0  # 用于变量节点版本

        # 性能数据 - 扩展支持详细的性能追踪
        self.performance = {
            'execution_time': 0.0,  # 秒
            'call_count': 0,
            'memory_usage': 0,
            'device': 'cpu',
            # 详细性能字段
            'start_time': 0.0,
            'end_time': 0.0,
            'execution_time_ms': 0.0,  # 毫秒
            'memory_before_mb': 0.0,
            'memory_after_mb': 0.0,
            'memory_allocated_mb': 0.0,
            'peak_memory_mb': 0.0
        }

        # 属性数据 - 用于存储variable_snapshot等额外信息
        self.attributes = {}

    def update_performance(self, execution_time: float = 0.0, memory_usage: int = 0,
                          device: str = 'cpu', **kwargs):
        """更新性能数据 - 支持额外的性能字段"""
        if execution_time > 0:
            self.performance['execution_time'] += execution_time
            self.performance['call_count'] += 1
        if memory_usage > 0:
            self.performance['memory_usage'] = max(self.performance['memory_usage'], memory_usage)
        self.performance['device'] = device

        # 更新其他性能字段
        for key, value in kwargs.items():
            if key in self.performance:
                self.performance[key] = value
        
    def get_avg_execution_time(self) -> float:
        """获取平均执行时间"""
        if self.performance['call_count'] == 0:
            return 0.0
        return self.performance['execution_time'] / self.performance['call_count']


class ExecutionDAG:
    """执行DAG数据结构"""
    
    def __init__(self):
        self.nodes: Dict[int, DAGNode] = {}
        self.edges: List[tuple] = []  # (source_id, target_id, edge_type)
        self.next_node_id = 0
        
        # 索引
        self.name_to_nodes: Dict[str, List[DAGNode]] = defaultdict(list)
        
    def add_node(self, name: str, node_type: str = "function_call",
                 module: str = "", context: str = "", version: int = 0) -> DAGNode:
        """添加新节点"""
        node = DAGNode(self.next_node_id, name, node_type, module, context)
        node.version = version
        self.nodes[self.next_node_id] = node
        self.name_to_nodes[name].append(node)
        self.next_node_id += 1

        logger.debug(f"添加DAG节点: {name} (ID: {node.node_id})")
        return node
        
    def add_edge(self, source_id: int, target_id: int, edge_type: str = "call"):
        """添加边"""
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append((source_id, target_id, edge_type))
            logger.debug(f"添加DAG边: {source_id} -> {target_id} ({edge_type})")
        else:
            logger.warning(f"无效的边: {source_id} -> {target_id}")
            
    def get_node_by_name(self, name: str) -> Optional[DAGNode]:
        """根据名称获取节点（返回第一个匹配的）"""
        nodes = self.name_to_nodes.get(name, [])
        return nodes[0] if nodes else None
        
    def get_nodes_by_name(self, name: str) -> List[DAGNode]:
        """根据名称获取所有匹配的节点"""
        return self.name_to_nodes.get(name, [])
        
    def get_node_count(self) -> int:
        """获取节点总数"""
        return len(self.nodes)
        
    def get_edge_count(self) -> int:
        """获取边总数"""
        return len(self.edges)


class DAGBuilder:
    """
    DAG构建器
    负责从追踪数据构建执行DAG
    """
    
    def __init__(self):
        self.dag = ExecutionDAG()
        self.call_stack = []  # 调用栈，用于构建边
        self.function_node_map = {}  # 函数名到节点的映射
        
    def reset(self):
        """重置DAG构建器"""
        self.dag = ExecutionDAG()
        self.call_stack.clear()
        self.function_node_map.clear()
        
    def on_function_enter(self, func_name: str, execution_time: float = 0.0, 
                         memory_usage: int = 0, device: str = 'cpu') -> DAGNode:
        """处理函数进入事件"""
        # 查找或创建节点
        node = self._get_or_create_node(func_name)
        
        # 更新性能数据
        node.update_performance(execution_time, memory_usage, device)
        
        # 如果有调用栈，创建边
        if self.call_stack:
            parent_node = self.call_stack[-1]
            self.dag.add_edge(parent_node.node_id, node.node_id, "call")
            
        # 推入调用栈
        self.call_stack.append(node)
        
        return node
        
    def on_function_exit(self, func_name: str, execution_time: float = 0.0):
        """处理函数退出事件"""
        if not self.call_stack:
            logger.warning(f"调用栈为空，无法处理函数退出: {func_name}")
            return
            
        # 弹出调用栈
        current_node = self.call_stack.pop()
        
        # 更新执行时间
        current_node.update_performance(execution_time)
        
    def _get_or_create_node(self, func_name: str) -> DAGNode:
        """获取或创建函数节点"""
        # 首先尝试从映射中获取
        if func_name in self.function_node_map:
            return self.function_node_map[func_name]
            
        # 创建新节点
        node = self.dag.add_node(func_name, "function_call")
        self.function_node_map[func_name] = node
        
        return node
        
    def build_from_stats(self, function_stats: Dict[str, Any]) -> ExecutionDAG:
        """从函数统计数据构建DAG"""
        self.reset()
        
        for func_name, stats in function_stats.items():
            node = self._get_or_create_node(func_name)
            node.update_performance(
                execution_time=stats.get('total_time', 0.0),
                memory_usage=stats.get('memory_usage', 0),
                device=stats.get('device', 'cpu')
            )
            node.performance['call_count'] = stats.get('call_count', 1)
            
        logger.info(f"从统计数据构建DAG完成: {self.dag.get_node_count()} 个节点")
        return self.dag
        
    def get_dag(self) -> ExecutionDAG:
        """获取构建的DAG"""
        return self.dag
        
    def get_summary(self) -> Dict[str, Any]:
        """获取DAG构建摘要"""
        return {
            'node_count': self.dag.get_node_count(),
            'edge_count': self.dag.get_edge_count(),
            'function_count': len(self.function_node_map),
            'nodes': [
                {
                    'id': node.node_id,
                    'name': node.name,
                    'performance': node.performance
                }
                for node in self.dag.nodes.values()
            ]
        }