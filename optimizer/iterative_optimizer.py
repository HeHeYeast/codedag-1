"""
迭代优化器 - Iterative Optimizer
基于贪心迭代 + 局部增量更新的分区优化算法
"""

import logging
import heapq
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque

try:
    from ..core.dag_builder import ExecutionDAG, DAGNode
    from .kway_partitioner import Partition
except ImportError:
    import sys
    sys.path.append('/mnt/sda/gxy/codedag_clean')
    from core.dag_builder import ExecutionDAG, DAGNode
    from optimizer.kway_partitioner import Partition

logger = logging.getLogger(__name__)


class IterativeOptimizer:
    """迭代优化器"""

    def __init__(self, alpha: float = 0.4, beta: float = 0.6,
                 max_iterations: int = 200,
                 no_improvement_threshold: int = 20,
                 recompute_cp_interval: int = 50,
                 communication_overhead: float = 0.0001):
        """
        Args:
            alpha: 负载均衡权重（CV(loads)）
            beta: 关键路径权重（max(CPs)）
            max_iterations: 最大迭代次数
            no_improvement_threshold: 连续无改善阈值
            recompute_cp_interval: 关键路径重算间隔
            communication_overhead: 跨分区通信开销系数
        """
        # 参数验证
        assert alpha >= 0 and beta >= 0, "weights must be non-negative"
        weight_sum = alpha + beta
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"权重之和={weight_sum:.3f}，建议接近1.0")

        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.no_improvement_threshold = no_improvement_threshold
        self.recompute_cp_interval = recompute_cp_interval
        self.communication_overhead = communication_overhead

        # 预计算数据
        self.node_weights: Dict[int, float] = {}
        self.partition_map: Dict[int, int] = {}  # node_id -> partition_id
        self.cp_lengths: Dict[int, float] = {}  # partition_id -> CP_length
        self.boundary_nodes: Dict[int, Set[int]] = defaultdict(set)  # partition_id -> boundary_nodes

        # 邻接表（优化边遍历效率）
        self.predecessors: Dict[int, List[Tuple[int, str]]] = defaultdict(list)  # node_id -> [(src, etype), ...]
        self.successors: Dict[int, List[Tuple[int, str]]] = defaultdict(list)    # node_id -> [(tgt, etype), ...]

        # 初始负载比率（用于动态阈值）
        self.initial_max_load_ratio: float = 0.0

        # 统计信息
        self.stats = {
            'iterations': 0,
            'moves_executed': 0,
            'best_objective': float('inf'),
            'initial_objective': 0.0,
            'final_objective': 0.0
        }

    def optimize(self, partitions: List[Partition], dag: ExecutionDAG) -> List[Partition]:
        """
        执行迭代优化

        流程：初始化 -> 建立GainHeap -> 主循环 -> 终止
        """
        logger.info("开始迭代优化...")

        if not partitions or len(dag.nodes) == 0:
            logger.warning("分区或DAG为空")
            return partitions

        # 深拷贝分区，避免修改原始数据
        partitions = [self._copy_partition(p) for p in partitions]

        # 初始化
        self._initialize(partitions, dag)

        # 计算初始目标函数值
        initial_obj = self._compute_objective(partitions)
        self.stats['initial_objective'] = initial_obj
        self.stats['best_objective'] = initial_obj
        logger.info(f"初始目标函数值: {initial_obj:.6f}")

        # 建立初始GainHeap
        gain_heap = self._build_initial_heap(partitions, dag)
        logger.info(f"初始堆大小: {len(gain_heap)}")

        # 主循环
        no_improvement_count = 0
        iteration = 0

        while gain_heap and iteration < self.max_iterations:
            iteration += 1

            # 取出收益最大的迁移
            neg_gain, node_id, from_pid, to_pid = heapq.heappop(gain_heap)
            gain = -neg_gain

            # 懒惰删除：验证有效性
            if not self._is_valid_move(node_id, from_pid, to_pid, partitions, dag):
                continue

            # 重新计算收益（可能已过期）
            current_gain = self._compute_gain(node_id, from_pid, to_pid, partitions, dag)
            if current_gain <= 0:
                continue

            # 执行迁移
            self._execute_move(node_id, from_pid, to_pid, partitions, dag)
            self.stats['moves_executed'] += 1

            # 更新受影响节点的收益
            affected_nodes = self._get_affected_nodes(node_id, from_pid, to_pid, partitions, dag)
            for affected_node in affected_nodes:
                self._update_node_gains(affected_node, partitions, dag, gain_heap)

            # 周期性重算关键路径
            if iteration % self.recompute_cp_interval == 0:
                logger.debug(f"第{iteration}轮：重算所有关键路径")
                for p in partitions:
                    self.cp_lengths[p.id] = self._compute_critical_path(p, partitions, dag)

            # 检查改善
            current_obj = self._compute_objective(partitions)
            if current_obj < self.stats['best_objective']:
                improvement = self.stats['best_objective'] - current_obj
                self.stats['best_objective'] = current_obj
                no_improvement_count = 0
                logger.debug(f"第{iteration}轮：目标函数改善 {improvement:.6f} -> {current_obj:.6f}")
            else:
                no_improvement_count += 1

            # 详细进度日志
            if iteration % 50 == 0:
                logger.info(
                    f"迭代 {iteration}: 目标函数={current_obj:.6f}, "
                    f"已执行 {self.stats['moves_executed']} 次迁移, "
                    f"堆大小={len(gain_heap)}"
                )

            # 终止条件：连续无改善
            if no_improvement_count >= self.no_improvement_threshold:
                logger.info(f"连续{no_improvement_count}轮无改善，提前终止")
                break

        self.stats['iterations'] = iteration
        self.stats['final_objective'] = self._compute_objective(partitions)

        improvement = self.stats['initial_objective'] - self.stats['final_objective']
        logger.info(
            f"迭代优化完成: {iteration}轮, "
            f"{self.stats['moves_executed']}次迁移, "
            f"目标函数改善 {improvement:.6f} "
            f"({self.stats['initial_objective']:.6f} -> {self.stats['final_objective']:.6f})"
        )

        return partitions

    # ==================== 初始化 ====================

    def _initialize(self, partitions: List[Partition], dag: ExecutionDAG):
        """初始化：计算节点权重、分区映射、关键路径、边界节点、邻接表"""
        logger.info("初始化优化器...")

        # 计算节点权重
        for node_id, node in dag.nodes.items():
            exec_time = node.performance.get('execution_time', 0.0)
            # 考虑聚类信息
            if hasattr(node, 'attributes') and 'cluster_info' in node.attributes:
                instance_count = node.attributes['cluster_info']['instance_count']
            else:
                instance_count = 1
            self.node_weights[node_id] = exec_time * instance_count

        # 预处理邻接表（优化边遍历效率）
        for src, tgt, etype in dag.edges:
            self.predecessors[tgt].append((src, etype))
            self.successors[src].append((tgt, etype))

        # 建立节点到分区的映射
        for partition in partitions:
            for node_id in partition.nodes:
                self.partition_map[node_id] = partition.id

        # 保存初始最大负载比率（用于动态阈值）
        avg_load = sum(p.total_compute for p in partitions) / len(partitions)
        if avg_load > 0:
            self.initial_max_load_ratio = max(p.total_compute / avg_load for p in partitions)
        else:
            self.initial_max_load_ratio = 1.0

        # 计算每个分区的关键路径
        for partition in partitions:
            self.cp_lengths[partition.id] = self._compute_critical_path(partition, partitions, dag)

        # 识别边界节点
        for partition in partitions:
            self.boundary_nodes[partition.id] = self._find_boundary_nodes(partition, dag)

        logger.info(
            f"初始化完成: CP长度={list(self.cp_lengths.values())}, "
            f"边界节点数={[len(bn) for bn in self.boundary_nodes.values()]}, "
            f"初始最大负载比={self.initial_max_load_ratio:.2f}"
        )

    def _copy_partition(self, partition: Partition) -> Partition:
        """深拷贝分区"""
        new_partition = Partition(partition.id, partition.center)
        new_partition.nodes = partition.nodes.copy()
        new_partition.total_compute = partition.total_compute
        new_partition.total_memory = partition.total_memory
        return new_partition

    # ==================== 关键路径计算 ====================

    def _compute_critical_path(self, partition: Partition, all_partitions: List[Partition],
                               dag: ExecutionDAG) -> float:
        """
        计算分区的关键路径长度

        修复：
        1. 正确处理跨分区前驱节点的影响
        2. 使用预处理的邻接表提升效率
        """
        if not partition.nodes:
            return 0.0

        partition_nodes = partition.nodes

        # 初始化距离：考虑跨分区输入
        dist = {}
        for node_id in partition_nodes:
            # 检查跨分区前驱（使用邻接表）
            max_external_input = 0.0
            for src, etype in self.predecessors[node_id]:
                if src not in partition_nodes:
                    # 跨分区前驱，考虑其执行时间 + 通信开销
                    external_cost = (self.node_weights.get(src, 0.0) +
                                   self._get_communication_cost(src, node_id, dag))
                    max_external_input = max(max_external_input, external_cost)
            dist[node_id] = max_external_input

        # 提取分区内边
        partition_edges = []
        for node_id in partition_nodes:
            for tgt, etype in self.successors[node_id]:
                if tgt in partition_nodes:
                    partition_edges.append((node_id, tgt, etype))

        # 拓扑排序（只考虑分区内的边）
        topo_order = self._topological_sort(partition_nodes, partition_edges)
        if not topo_order:
            # 无法拓扑排序，返回所有节点权重之和作为估计
            logger.warning(f"分区{partition.id}无法拓扑排序")
            return sum(self.node_weights.get(n, 0) for n in partition_nodes)

        # 前向传播：只考虑分区内的依赖（使用邻接表）
        for node_id in topo_order:
            for src, etype in self.predecessors[node_id]:
                if src in partition_nodes:
                    # 分区内前驱
                    dist[node_id] = max(
                        dist[node_id],
                        dist[src] + self.node_weights.get(src, 0.0)
                    )

        # 找出口节点（使用邻接表）
        exit_nodes = []
        for node_id in partition_nodes:
            has_successor_in_partition = any(
                tgt in partition_nodes for tgt, _ in self.successors[node_id]
            )
            if not has_successor_in_partition:
                exit_nodes.append(node_id)

        if not exit_nodes:
            exit_nodes = list(partition_nodes)

        # 关键路径长度 = 出口节点的最大完成时间
        cp_length = max(
            dist[node_id] + self.node_weights.get(node_id, 0.0)
            for node_id in exit_nodes
        )

        return cp_length

    def _topological_sort(self, nodes: Set[int], edges: List[Tuple]) -> List[int]:
        """拓扑排序（只考虑给定的节点和边）"""
        # 计算入度
        in_degree = {node_id: 0 for node_id in nodes}
        adj_list = defaultdict(list)

        for src, tgt, _ in edges:
            if src in nodes and tgt in nodes:
                adj_list[src].append(tgt)
                in_degree[tgt] += 1

        # BFS
        queue = deque([node_id for node_id in nodes if in_degree[node_id] == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 检查是否所有节点都被访问（无环）
        if len(result) != len(nodes):
            logger.warning(f"拓扑排序失败：访问了{len(result)}/{len(nodes)}个节点")
            return []

        return result

    def _get_communication_cost(self, src_id: int, tgt_id: int, dag: ExecutionDAG) -> float:
        """计算跨分区通信开销"""
        # 简化模型：基于源节点权重的比例
        src_weight = self.node_weights.get(src_id, 0.0)
        return src_weight * self.communication_overhead

    # ==================== 边界节点识别 ====================

    def _find_boundary_nodes(self, partition: Partition, dag: ExecutionDAG) -> Set[int]:
        """
        识别边界节点（使用邻接表优化）

        边界节点 = 有跨分区边连接的节点
        """
        boundary = set()

        for node_id in partition.nodes:
            # 检查出边（使用邻接表）
            for tgt, _ in self.successors[node_id]:
                if tgt not in partition.nodes:
                    boundary.add(node_id)
                    break

            # 检查入边（使用邻接表）
            if node_id not in boundary:
                for src, _ in self.predecessors[node_id]:
                    if src not in partition.nodes:
                        boundary.add(node_id)
                        break

        return boundary

    def _update_boundary_incremental(self, node_id: int, from_pid: int, to_pid: int,
                                    partitions: List[Partition], dag: ExecutionDAG):
        """
        增量更新边界节点

        只检查迁移节点及其邻居
        """
        from_partition = self._get_partition_by_id(partitions, from_pid)
        to_partition = self._get_partition_by_id(partitions, to_pid)

        if not from_partition or not to_partition:
            return

        # 收集需要检查的节点（使用邻接表）
        nodes_to_check = {node_id}

        # 添加迁移节点的邻居
        for tgt, _ in self.successors[node_id]:
            nodes_to_check.add(tgt)
        for src, _ in self.predecessors[node_id]:
            nodes_to_check.add(src)

        # 重新检查这些节点是否是边界节点
        for check_node in nodes_to_check:
            # 确定该节点所属分区
            node_pid = self.partition_map.get(check_node)
            if node_pid is None:
                continue

            partition = self._get_partition_by_id(partitions, node_pid)
            if not partition:
                continue

            # 检查是否是边界节点（使用邻接表）
            is_boundary = False

            # 检查出边
            for tgt, _ in self.successors[check_node]:
                if tgt not in partition.nodes:
                    is_boundary = True
                    break

            # 检查入边
            if not is_boundary:
                for src, _ in self.predecessors[check_node]:
                    if src not in partition.nodes:
                        is_boundary = True
                        break

            # 更新边界集合
            if is_boundary:
                self.boundary_nodes[node_pid].add(check_node)
            else:
                self.boundary_nodes[node_pid].discard(check_node)

    # ==================== 收益计算 ====================

    def _compute_gain(self, node_id: int, from_pid: int, to_pid: int,
                     partitions: List[Partition], dag: ExecutionDAG) -> float:
        """
        计算迁移收益

        Gain = J_before - J_after
        """
        # 当前目标函数值
        j_before = self._compute_objective(partitions)

        # 模拟迁移后的目标函数值（异常安全）
        try:
            j_after = self._simulate_move_objective(node_id, from_pid, to_pid, partitions, dag)
        except Exception as e:
            logger.error(f"计算收益时出错: {e}")
            return -float('inf')  # 返回负无穷，该迁移不会被选中

        gain = j_before - j_after
        return gain

    def _compute_objective(self, partitions: List[Partition]) -> float:
        """
        计算目标函数（修复：量纲归一化）

        J = α · CV(loads) + β · normalized_max_cp
        """
        # 负载列表
        loads = [p.total_compute for p in partitions]

        # CV(loads) = std / mean（已经是无量纲的）
        mean_load = sum(loads) / len(loads) if loads else 1.0
        if mean_load > 0:
            variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
            std_load = variance ** 0.5
            cv_load = std_load / mean_load
        else:
            cv_load = 0.0

        # max(CPs) - 归一化到1附近
        if self.cp_lengths:
            max_cp = max(self.cp_lengths.values())
            mean_cp = sum(self.cp_lengths.values()) / len(self.cp_lengths)
            normalized_cp = max_cp / mean_cp if mean_cp > 0 else 0.0
        else:
            normalized_cp = 0.0

        # 综合目标
        objective = self.alpha * cv_load + self.beta * normalized_cp

        return objective

    def _simulate_move_objective(self, node_id: int, from_pid: int, to_pid: int,
                                 partitions: List[Partition], dag: ExecutionDAG) -> float:
        """
        模拟迁移后的目标函数值

        修复：
        1. 异常安全（try-finally）
        2. 使用字典索引而非列表索引
        3. 健壮性检查（.get()方法）
        """
        from_partition = self._get_partition_by_id(partitions, from_pid)
        to_partition = self._get_partition_by_id(partitions, to_pid)

        if not from_partition or not to_partition:
            return float('inf')

        node_weight = self.node_weights.get(node_id, 0.0)

        # 1. 模拟负载变化
        loads = []
        for p in partitions:
            if p.id == from_pid:
                loads.append(p.total_compute - node_weight)
            elif p.id == to_pid:
                loads.append(p.total_compute + node_weight)
            else:
                loads.append(p.total_compute)

        # CV(loads)
        mean_load = sum(loads) / len(loads) if loads else 1.0
        if mean_load > 0:
            variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
            std_load = variance ** 0.5
            cv_load = std_load / mean_load
        else:
            cv_load = 0.0

        # 2. 估算关键路径变化（异常安全 + 健壮性）
        old_pid = self.partition_map.get(node_id)
        if old_pid is None:
            return float('inf')

        try:
            # 临时修改分区进行模拟
            from_partition.nodes.remove(node_id)
            to_partition.nodes.add(node_id)
            self.partition_map[node_id] = to_pid

            # 重算CP
            new_cp_from = self._compute_critical_path(from_partition, partitions, dag)
            new_cp_to = self._compute_critical_path(to_partition, partitions, dag)

        finally:
            # 恢复分区（确保执行）
            from_partition.nodes.add(node_id)
            to_partition.nodes.remove(node_id)
            self.partition_map[node_id] = old_pid

        # max(CPs) - 使用字典而非列表索引
        cp_values = dict(self.cp_lengths)
        cp_values[from_pid] = new_cp_from
        cp_values[to_pid] = new_cp_to

        max_cp = max(cp_values.values())
        mean_cp = sum(cp_values.values()) / len(cp_values)
        normalized_cp = max_cp / mean_cp if mean_cp > 0 else 0.0

        # 综合目标
        objective = self.alpha * cv_load + self.beta * normalized_cp

        return objective

    def _get_partition_by_id(self, partitions: List[Partition], pid: int) -> Optional[Partition]:
        """根据ID获取分区"""
        for p in partitions:
            if p.id == pid:
                return p
        return None

    # ==================== 迁移验证 ====================

    def _is_valid_move(self, node_id: int, from_pid: int, to_pid: int,
                      partitions: List[Partition], dag: ExecutionDAG) -> bool:
        """
        验证迁移是否有效

        三个硬约束：
        1. 不移动中心节点
        2. 只允许迁移到相邻分区
        3. 负载比不超限（基于初始比率的动态阈值）
        """
        # 验证节点仍在from分区
        if self.partition_map.get(node_id) != from_pid:
            return False

        # 约束1：不移动中心节点
        from_partition = self._get_partition_by_id(partitions, from_pid)
        if not from_partition or node_id == from_partition.center:
            return False

        # 约束2：只允许迁移到相邻分区
        adjacent_partitions = self._get_adjacent_partitions(node_id, partitions, dag)
        if to_pid not in adjacent_partitions:
            return False

        # 约束3：负载比不超限（使用初始比率的动态阈值）
        to_partition = self._get_partition_by_id(partitions, to_pid)
        if not to_partition:
            return False

        node_weight = self.node_weights.get(node_id, 0.0)
        avg_load = sum(p.total_compute for p in partitions) / len(partitions)

        if avg_load > 0:
            would_be_load = to_partition.total_compute + node_weight
            load_ratio = would_be_load / avg_load

            # 动态阈值：max(2.0, initial_max_ratio * 1.2)
            max_allowed_ratio = max(2.0, self.initial_max_load_ratio * 1.2)

            if load_ratio > max_allowed_ratio:
                return False

        return True

    def _get_adjacent_partitions(self, node_id: int, partitions: List[Partition],
                                dag: ExecutionDAG) -> Set[int]:
        """获取节点的相邻分区（使用邻接表）"""
        adjacent = set()

        # 检查后继节点
        for tgt, _ in self.successors[node_id]:
            tgt_pid = self.partition_map.get(tgt)
            if tgt_pid is not None:
                adjacent.add(tgt_pid)

        # 检查前驱节点
        for src, _ in self.predecessors[node_id]:
            src_pid = self.partition_map.get(src)
            if src_pid is not None:
                adjacent.add(src_pid)

        # 移除当前分区
        current_pid = self.partition_map.get(node_id)
        if current_pid in adjacent:
            adjacent.remove(current_pid)

        return adjacent

    # ==================== 迁移执行 ====================

    def _execute_move(self, node_id: int, from_pid: int, to_pid: int,
                     partitions: List[Partition], dag: ExecutionDAG):
        """执行迁移"""
        from_partition = self._get_partition_by_id(partitions, from_pid)
        to_partition = self._get_partition_by_id(partitions, to_pid)

        if not from_partition or not to_partition:
            return

        node_weight = self.node_weights.get(node_id, 0.0)

        # 更新分区
        from_partition.nodes.remove(node_id)
        to_partition.nodes.add(node_id)

        # 更新负载
        from_partition.total_compute -= node_weight
        to_partition.total_compute += node_weight

        # 更新映射
        self.partition_map[node_id] = to_pid

        # 增量更新边界节点
        self._update_boundary_incremental(node_id, from_pid, to_pid, partitions, dag)

        # 更新关键路径
        self.cp_lengths[from_pid] = self._compute_critical_path(from_partition, partitions, dag)
        self.cp_lengths[to_pid] = self._compute_critical_path(to_partition, partitions, dag)

    # ==================== 堆管理 ====================

    def _build_initial_heap(self, partitions: List[Partition], dag: ExecutionDAG) -> List:
        """建立初始GainHeap"""
        heap = []

        for partition in partitions:
            for node_id in self.boundary_nodes[partition.id]:
                # 跳过中心节点
                if node_id == partition.center:
                    continue

                # 计算迁移到相邻分区的收益
                adjacent = self._get_adjacent_partitions(node_id, partitions, dag)
                for to_pid in adjacent:
                    gain = self._compute_gain(node_id, partition.id, to_pid, partitions, dag)
                    if gain > 0:
                        # 使用负收益以实现最大堆
                        heapq.heappush(heap, (-gain, node_id, partition.id, to_pid))

        return heap

    def _get_affected_nodes(self, moved_node: int, from_pid: int, to_pid: int,
                           partitions: List[Partition], dag: ExecutionDAG) -> Set[int]:
        """获取受迁移影响的节点（使用邻接表）"""
        affected = set()

        # 移动节点的邻居
        for tgt, _ in self.successors[moved_node]:
            affected.add(tgt)
        for src, _ in self.predecessors[moved_node]:
            affected.add(src)

        # 受影响分区的边界节点
        affected.update(self.boundary_nodes[from_pid])
        affected.update(self.boundary_nodes[to_pid])

        # 移除已迁移的节点本身
        affected.discard(moved_node)

        return affected

    def _update_node_gains(self, node_id: int, partitions: List[Partition],
                          dag: ExecutionDAG, heap: List):
        """更新节点的收益并加入堆"""
        current_pid = self.partition_map.get(node_id)
        if current_pid is None:
            return

        current_partition = self._get_partition_by_id(partitions, current_pid)
        if not current_partition or node_id == current_partition.center:
            return

        # 计算迁移到相邻分区的收益
        adjacent = self._get_adjacent_partitions(node_id, partitions, dag)
        for to_pid in adjacent:
            gain = self._compute_gain(node_id, current_pid, to_pid, partitions, dag)
            if gain > 0:
                heapq.heappush(heap, (-gain, node_id, current_pid, to_pid))

    # ==================== 统计信息 ====================

    def get_stats(self) -> Dict:
        """获取优化统计信息"""
        improvement = self.stats['initial_objective'] - self.stats['final_objective']
        improvement_ratio = (improvement / self.stats['initial_objective'] * 100
                            if self.stats['initial_objective'] > 0 else 0)

        return {
            'iterations': self.stats['iterations'],
            'moves_executed': self.stats['moves_executed'],
            'initial_objective': self.stats['initial_objective'],
            'final_objective': self.stats['final_objective'],
            'improvement': improvement,
            'improvement_ratio': improvement_ratio
        }
