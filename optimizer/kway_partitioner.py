"""
K-way图划分器 - K-way Graph Partitioner
基于动态贪心 + 增量扩展策略的图划分算法
"""

import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque

try:
    from ..core.dag_builder import ExecutionDAG, DAGNode
except ImportError:
    import sys
    sys.path.append('/mnt/sda/gxy/codedag_clean')
    from core.dag_builder import ExecutionDAG, DAGNode

logger = logging.getLogger(__name__)


class Partition:
    """分区数据结构"""

    def __init__(self, partition_id: int, center_node_id: int):
        self.id = partition_id
        self.center = center_node_id
        self.nodes: Set[int] = {center_node_id}
        self.total_compute: float = 0.0
        self.total_memory: float = 0.0


class KWayPartitioner:
    """K-way图划分器"""

    def __init__(self, k: int = 4, lambda_weight: float = 0.5,
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        Args:
            k: 分区数量
            lambda_weight: 瓶颈选择的距离权重
            alpha: 性能权重
            beta: 距离惩罚权重
            gamma: 负载惩罚权重
        """
        # 参数验证
        assert k > 0, "k must be positive"
        assert 0 <= lambda_weight <= 1, "lambda_weight must be in [0,1]"
        assert alpha >= 0 and beta >= 0 and gamma >= 0, "weights must be non-negative"

        # 权重和检查
        weight_sum = alpha + beta + gamma
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"权重之和={weight_sum:.3f}，建议接近1.0")

        self.k = k
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 预计算数据
        self.diameter: int = 0
        self.node_weights: Dict[int, float] = {}
        self.max_weight: float = 0.0
        self.eccentricity: Dict[int, int] = {}
        self.center_distances: Dict[int, Dict[int, int]] = defaultdict(dict)

    def partition(self, dag: ExecutionDAG) -> List[Partition]:
        """
        执行K-way划分

        流程：预处理 -> 瓶颈选择 -> 初始化 -> 贪心分配 -> 孤立节点处理
        """
        logger.info(f"开始K-way划分 (k={self.k})...")

        if len(dag.nodes) == 0:
            logger.warning("DAG为空")
            return []

        # 阶段0: 预处理
        self._preprocess(dag)

        # 阶段1: 选择K个瓶颈节点
        bottlenecks = self._select_bottlenecks(dag)
        logger.info(f"选择了 {len(bottlenecks)} 个瓶颈节点: {bottlenecks}")

        # 阶段2: 初始化分区和候选集
        partitions, candidates, assigned = self._initialize_partitions(dag, bottlenecks)

        # 阶段3: 动态贪心分配
        self._greedy_assignment(dag, partitions, candidates, assigned)

        # 阶段4: 处理孤立节点
        self._handle_isolated_nodes(dag, partitions, assigned)

        logger.info(f"K-way划分完成: {[len(p.nodes) for p in partitions]}")
        return partitions

    # ==================== 阶段0: 预处理 ====================

    def _preprocess(self, dag: ExecutionDAG):
        """预处理：计算图直径、节点权重"""
        logger.info("预处理: 计算图特征...")

        # 计算离心率和直径
        self._compute_diameter(dag)

        # 计算节点权重
        self._compute_node_weights(dag)

        logger.debug(f"图直径: {self.diameter}, 最大权重: {self.max_weight:.2f}")

        # Debug: 检查边界情况
        if self.diameter == 0:
            logger.warning(f"⚠️  图直径为0！这通常表示图中节点几乎没有连通性。节点数:{len(dag.nodes)}, 边数:{len(dag.edges)}")
        if self.max_weight == 0:
            logger.warning(f"⚠️  最大权重为0！这通常表示所有节点的性能数据（时间/内存）都为0。")

    def _compute_diameter(self, dag: ExecutionDAG):
        """计算图直径（基于离心率）"""
        for node_id in dag.nodes:
            # BFS计算从node_id到所有其他节点的最短路径
            distances = self._bfs_distances(dag, node_id)
            # 离心率 = 到最远节点的距离
            self.eccentricity[node_id] = max(distances.values()) if distances else 0

        # 直径 = 所有节点离心率的最大值
        self.diameter = max(self.eccentricity.values()) if self.eccentricity else 1

    def _bfs_distances(self, dag: ExecutionDAG, start_node: int) -> Dict[int, int]:
        """BFS计算从start_node到所有节点的最短距离"""
        distances = {start_node: 0}
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            current_dist = distances[current]

            # 遍历所有出边
            for src, tgt, _ in dag.edges:
                if src == current and tgt not in distances:
                    distances[tgt] = current_dist + 1
                    queue.append(tgt)

        return distances

    def _compute_node_weights(self, dag: ExecutionDAG):
        """计算节点权重（考虑instance_count）"""
        for node_id, node in dag.nodes.items():
            # 基础权重：执行时间 + 内存
            exec_time = node.performance.get('execution_time', 0.0)
            memory = node.performance.get('memory_usage', 0)

            # 修正：合理的权重计算
            # 假设memory单位是字节，转换为MB级别与时间相当
            # 如果memory已经是合理单位，直接相加
            base_weight = exec_time + memory / 1e6  # 字节转MB

            # 考虑聚类信息
            if hasattr(node, 'attributes') and 'cluster_info' in node.attributes:
                instance_count = node.attributes['cluster_info']['instance_count']
            else:
                instance_count = 1

            self.node_weights[node_id] = base_weight * instance_count

        self.max_weight = max(self.node_weights.values()) if self.node_weights else 1.0

    # ==================== 阶段1: 选择瓶颈节点 ====================

    def _select_bottlenecks(self, dag: ExecutionDAG) -> List[int]:
        """选择K个瓶颈节点"""
        selected = []
        total_weight = sum(self.node_weights.values())

        for i in range(min(self.k, len(dag.nodes))):
            best_node = None
            best_score = -float('inf')

            for node_id in dag.nodes:
                if node_id in selected:
                    continue

                # 计算距离项
                if not selected:
                    # 第一个节点：使用离心率
                    dist = self.eccentricity[node_id]
                else:
                    # 后续节点：到已选节点的最小距离
                    dist = min(
                        self._compute_distance(dag, node_id, sel_node)
                        for sel_node in selected
                    )

                # 计算综合得分
                dist_score = dist / self.diameter if self.diameter > 0 else 0
                perf_score = self.node_weights[node_id] / total_weight if total_weight > 0 else 0
                score = self.lambda_weight * dist_score + (1 - self.lambda_weight) * perf_score

                if score > best_score:
                    best_score = score
                    best_node = node_id

            selected.append(best_node)

        # 预计算中心距离
        self._precompute_center_distances(dag, selected)

        return selected

    def _compute_distance(self, dag: ExecutionDAG, node1: int, node2: int) -> int:
        """计算两个节点间的最短距离"""
        if node1 == node2:
            return 0

        distances = self._bfs_distances(dag, node1)
        return distances.get(node2, self.diameter)

    def _precompute_center_distances(self, dag: ExecutionDAG, centers: List[int]):
        """预计算所有节点到中心节点的距离"""
        for center in centers:
            distances = self._bfs_distances(dag, center)
            for node_id in dag.nodes:
                self.center_distances[node_id][center] = distances.get(node_id, self.diameter)

    # ==================== 阶段2: 初始化 ====================

    def _initialize_partitions(self, dag: ExecutionDAG, bottlenecks: List[int]
                              ) -> Tuple[List[Partition], Set[int], Set[int]]:
        """初始化分区和候选集"""
        partitions = []
        assigned = set(bottlenecks)
        candidates = set()

        # 创建分区
        for i, center in enumerate(bottlenecks):
            partition = Partition(i, center)
            partition.total_compute = self.node_weights[center]
            partition.total_memory = self.node_weights[center]
            partitions.append(partition)

            # 添加中心节点的后继到候选集
            for src, tgt, _ in dag.edges:
                if src == center and tgt not in assigned:
                    candidates.add(tgt)

        logger.debug(f"初始化: {len(partitions)} 个分区, {len(candidates)} 个候选节点")
        return partitions, candidates, assigned

    # ==================== 阶段3: 动态贪心分配 ====================

    def _greedy_assignment(self, dag: ExecutionDAG, partitions: List[Partition],
                          candidates: Set[int], assigned: Set[int]):
        """动态贪心分配"""
        iteration = 0
        total_nodes = len(dag.nodes)

        while candidates:
            iteration += 1

            # 全局搜索最优分配
            best_node = None
            best_partition = None
            best_score = -float('inf')

            for node_id in candidates:
                for partition in partitions:
                    score = self._compute_assignment_score(node_id, partition, partitions)

                    if score > best_score:
                        best_score = score
                        best_node = node_id
                        best_partition = partition

            if best_node is None:
                break

            # 执行分配
            best_partition.nodes.add(best_node)
            best_partition.total_compute += self.node_weights[best_node]
            best_partition.total_memory += self.node_weights[best_node]

            assigned.add(best_node)
            candidates.remove(best_node)

            # 更新候选集
            for src, tgt, _ in dag.edges:
                if src == best_node and tgt not in assigned:
                    candidates.add(tgt)

            # 详细进度日志
            if iteration % 100 == 0:
                progress = len(assigned) / total_nodes * 100
                logger.info(
                    f"迭代 {iteration}: 已分配 {len(assigned)}/{total_nodes} "
                    f"({progress:.1f}%), 候选 {len(candidates)}"
                )

    def _compute_assignment_score(self, node_id: int, partition: Partition,
                                  all_partitions: List[Partition]) -> float:
        """计算节点分配到分区的得分"""
        # 1. 性能得分（归一化）
        perf_score = self.node_weights[node_id] / self.max_weight if self.max_weight > 0 else 0

        # 2. 距离惩罚（归一化）
        dist = self.center_distances[node_id][partition.center]
        dist_penalty = dist / self.diameter if self.diameter > 0 else 0

        # 3. 负载惩罚
        avg_load = sum(p.total_compute for p in all_partitions) / len(all_partitions)
        would_be_load = partition.total_compute + self.node_weights[node_id]
        load_ratio = would_be_load / avg_load if avg_load > 0 else 1.0

        if load_ratio > 1.0:
            load_penalty = (load_ratio - 1.0) ** 2
        else:
            load_penalty = 0.0

        # 综合得分
        score = self.alpha * perf_score - self.beta * dist_penalty - self.gamma * load_penalty

        return score

    # ==================== 阶段4: 孤立节点处理 ====================

    def _handle_isolated_nodes(self, dag: ExecutionDAG, partitions: List[Partition],
                               assigned: Set[int]):
        """处理孤立节点（未被分配的节点）"""
        isolated = set(dag.nodes.keys()) - assigned

        if isolated:
            logger.info(f"处理 {len(isolated)} 个孤立节点")

            for node_id in isolated:
                # 找到负载最轻的分区
                min_partition = min(partitions, key=lambda p: p.total_compute)

                # 分配
                min_partition.nodes.add(node_id)
                min_partition.total_compute += self.node_weights[node_id]
                min_partition.total_memory += self.node_weights[node_id]

    # ==================== 统计信息 ====================

    def get_stats(self, partitions: List[Partition]) -> Dict:
        """获取划分统计信息"""
        if not partitions:
            return {}

        loads = [p.total_compute for p in partitions]
        total_compute = sum(loads)
        avg_compute = total_compute / len(partitions)

        # 修正：按设计文档的定义计算负载均衡度
        max_load = max(loads)
        min_load = min(loads)

        if avg_compute > 0:
            load_balance = 1 - (max_load - min_load) / avg_compute
        else:
            load_balance = 1.0

        return {
            'partition_count': len(partitions),
            'partition_sizes': [len(p.nodes) for p in partitions],
            'partition_loads': loads,
            'avg_load': avg_compute,
            'max_load': max_load,
            'min_load': min_load,
            'load_balance': load_balance,  # 值域[0,1]，越接近1越均衡
            'total_nodes': sum(len(p.nodes) for p in partitions)
        }

    def compute_edge_cut(self, partitions: List[Partition], dag: ExecutionDAG) -> Dict:
        """计算边切割统计"""
        # 建立节点到分区的映射
        node_to_partition = {}
        for p in partitions:
            for node_id in p.nodes:
                node_to_partition[node_id] = p.id

        # 统计边
        total_edges = len(dag.edges)
        cut_edges = 0

        for src, tgt, _ in dag.edges:
            if node_to_partition.get(src) != node_to_partition.get(tgt):
                cut_edges += 1

        cut_ratio = cut_edges / total_edges if total_edges > 0 else 0

        return {
            'total_edges': total_edges,
            'cut_edges': cut_edges,
            'internal_edges': total_edges - cut_edges,
            'cut_ratio': cut_ratio
        }
