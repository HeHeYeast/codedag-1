"""
图粗化聚类器 - Graph Coarsening
核心：识别重复的子图模式，计算结构Hash，合并同构子图
"""

import logging
import hashlib
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque
from copy import deepcopy

try:
    from ..core.dag_builder import ExecutionDAG, DAGNode
except ImportError:
    import sys
    sys.path.append('/mnt/sda/gxy/codedag_clean')
    from core.dag_builder import ExecutionDAG, DAGNode

logger = logging.getLogger(__name__)


class Subgraph:
    """子图数据结构"""

    def __init__(self, anchor_id: int):
        self.anchor_id = anchor_id
        self.nodes: Set[int] = set()
        self.internal_edges: List[Tuple] = []
        self.input_edges: List[Tuple] = []
        self.output_edges: List[Tuple] = []
        self.hash: str = ""

    def add_node(self, node_id: int):
        """添加节点到子图"""
        self.nodes.add(node_id)


class GraphCoarsening:
    """图粗化聚类器"""

    def __init__(self, max_depth: int = 5, anchor_patterns: List[str] = None):
        """
        Args:
            max_depth: 子图提取最大深度
            anchor_patterns: 锚点模式列表
        """
        self.max_depth = max_depth
        self.anchor_patterns = anchor_patterns or []

        self.anchors: Dict[str, List[int]] = defaultdict(list)
        self.subgraphs: List[Subgraph] = []
        self.hash_groups: Dict[str, List[Subgraph]] = defaultdict(list)
        self.node_mapping: Dict[int, int] = {}

    def coarsen(self, dag: ExecutionDAG) -> ExecutionDAG:
        """执行图粗化"""
        logger.info("开始图粗化聚类...")

        self._identify_anchors(dag)
        if not self.anchors:
            logger.info("未发现重复锚点，跳过聚类")
            return deepcopy(dag)

        self._extract_subgraphs(dag)
        self._group_by_hash(dag)
        coarsened_dag = self._merge_subgraphs(dag)

        logger.info(f"粗化完成: {len(dag.nodes)} -> {len(coarsened_dag.nodes)} 节点")
        return coarsened_dag

    # ==================== 锚点识别 ====================

    def _identify_anchors(self, dag: ExecutionDAG):
        """识别重复出现的锚点节点"""
        name_groups = defaultdict(list)
        for node_id, node in dag.nodes.items():
            if node.node_type == "function_call":
                normalized = self._normalize_name(node.name)
                name_groups[normalized].append(node_id)

        for name, node_ids in name_groups.items():
            if len(node_ids) >= 2:
                self.anchors[name] = node_ids

        # 用户指定pattern：只保留重复的
        if self.anchor_patterns:
            pattern_nodes = defaultdict(list)
            for pattern in self.anchor_patterns:
                for node_id, node in dag.nodes.items():
                    if pattern in node.name and node.node_type == "function_call":
                        normalized = self._normalize_name(node.name)
                        pattern_nodes[normalized].append(node_id)

            for normalized, node_ids in pattern_nodes.items():
                if len(node_ids) >= 2:
                    self.anchors[normalized] = node_ids

        logger.info(f"识别到 {len(self.anchors)} 个锚点模式")
        for name, node_ids in self.anchors.items():
            logger.debug(f"  锚点 '{name}': {len(node_ids)} 个实例")

    def _normalize_name(self, name: str) -> str:
        """去除实例编号"""
        return re.sub(r'#\d+$', '', name)

    # ==================== 子图提取 ====================

    def _extract_subgraphs(self, dag: ExecutionDAG):
        """从锚点提取完整子图"""
        all_anchor_ids = set()
        for anchor_list in self.anchors.values():
            all_anchor_ids.update(anchor_list)

        for anchor_id in all_anchor_ids:
            subgraph = Subgraph(anchor_id)
            self._bfs_extract_subgraph(dag, anchor_id, subgraph, all_anchor_ids)
            self._classify_subgraph_edges(dag, subgraph)

            # 过滤太小的子图
            if len(subgraph.nodes) < 2:
                logger.debug(f"跳过单节点子图 (anchor={anchor_id})")
                continue

            self.subgraphs.append(subgraph)

    def _bfs_extract_subgraph(self, dag: ExecutionDAG, anchor_id: int,
                             subgraph: Subgraph, all_anchors: Set[int]):
        """BFS提取子图节点"""
        queue = deque([(anchor_id, 0)])
        visited = {anchor_id}
        subgraph.add_node(anchor_id)

        while queue:
            current_id, depth = queue.popleft()

            if depth >= self.max_depth:
                continue

            for src, tgt, edge_type in dag.edges:
                if src == current_id and tgt not in visited:
                    if tgt in all_anchors and tgt != anchor_id:
                        continue

                    visited.add(tgt)
                    subgraph.add_node(tgt)
                    queue.append((tgt, depth + 1))

    def _classify_subgraph_edges(self, dag: ExecutionDAG, subgraph: Subgraph):
        """分类边"""
        for src, tgt, etype in dag.edges:
            src_in = src in subgraph.nodes
            tgt_in = tgt in subgraph.nodes

            if src_in and tgt_in:
                subgraph.internal_edges.append((src, tgt, etype))
            elif not src_in and tgt_in:
                subgraph.input_edges.append((src, tgt, etype))
            elif src_in and not tgt_in:
                subgraph.output_edges.append((src, tgt, etype))

    # ==================== Hash计算与分组 ====================

    def _group_by_hash(self, dag: ExecutionDAG):
        """计算子图Hash并分组"""
        for subgraph in self.subgraphs:
            subgraph.hash = self._compute_subgraph_hash(dag, subgraph)
            self.hash_groups[subgraph.hash].append(subgraph)

        self.hash_groups = {
            h: sgs for h, sgs in self.hash_groups.items()
            if len(sgs) >= 2
        }

        logger.info(f"发现 {len(self.hash_groups)} 个同构模式组")
        for h, sgs in self.hash_groups.items():
            logger.debug(f"  模式 {h[:8]}: {len(sgs)} 个实例")

    def _compute_subgraph_hash(self, dag: ExecutionDAG, subgraph: Subgraph) -> str:
        """计算子图结构Hash"""
        signature = []
        queue = deque([(subgraph.anchor_id, 0)])
        visited = {subgraph.anchor_id}
        node_order = {subgraph.anchor_id: 0}
        order_counter = 1

        while queue:
            node_id, depth = queue.popleft()
            node = dag.nodes[node_id]

            norm_name = self._normalize_name(node.name)
            signature.append(f"{depth}:{node.node_type}:{norm_name}")

            children = []
            for src, tgt, etype in subgraph.internal_edges:
                if src == node_id:
                    tgt_node = dag.nodes[tgt]
                    tgt_name = self._normalize_name(tgt_node.name)
                    children.append((etype, tgt_name, tgt))

            children.sort()

            for etype, tgt_name, tgt_id in children:
                if tgt_id in visited:
                    signature.append(f"{depth}:BACK:{etype}:{node_order[tgt_id]}")
                else:
                    signature.append(f"{depth}:EDGE:{etype}")
                    visited.add(tgt_id)
                    node_order[tgt_id] = order_counter
                    order_counter += 1
                    queue.append((tgt_id, depth + 1))

        sig_str = "|".join(signature)
        return hashlib.sha256(sig_str.encode()).hexdigest()

    # ==================== 合并同构子图 ====================

    def _merge_subgraphs(self, dag: ExecutionDAG) -> ExecutionDAG:
        """合并同构子图"""
        new_dag = ExecutionDAG()

        self._build_node_mappings(dag)

        merged_nodes = self._create_merged_nodes(dag)
        for node in merged_nodes:
            new_dag.nodes[node.node_id] = node

        clustered_ids = set(self.node_mapping.keys()) | set(self.node_mapping.values())
        for node_id, node in dag.nodes.items():
            if node_id not in clustered_ids:
                new_dag.nodes[node_id] = deepcopy(node)

        self._rebuild_edges(dag, new_dag)
        self._validate_cluster_consistency(new_dag)

        return new_dag

    def _build_node_mappings(self, dag: ExecutionDAG):
        """建立节点映射关系，检测重叠"""
        mapped_instances = set()

        for hash_value, subgraphs in self.hash_groups.items():
            representative = subgraphs[0]

            for instance in subgraphs[1:]:
                mapping = self._match_subgraph_nodes(dag, representative, instance)

                # 检测节点重叠
                overlap = mapped_instances & set(mapping.keys())
                if overlap:
                    logger.error(f"检测到节点重叠: {overlap}")

                mapped_instances.update(mapping.keys())

                # 验证映射完整性（90%阈值）
                mapping_ratio = len(mapping) / len(instance.nodes) if instance.nodes else 0
                if mapping_ratio < 0.9:
                    logger.warning(
                        f"节点映射率较低: {len(mapping)}/{len(instance.nodes)} "
                        f"({mapping_ratio:.1%}) (hash: {hash_value[:8]})"
                    )

                self.node_mapping.update(mapping)

    def _match_subgraph_nodes(self, dag: ExecutionDAG,
                             repr_sg: Subgraph, inst_sg: Subgraph) -> Dict[int, int]:
        """匹配同构子图的节点"""
        mapping = {}

        repr_queue = deque([repr_sg.anchor_id])
        inst_queue = deque([inst_sg.anchor_id])
        repr_visited = {repr_sg.anchor_id}
        inst_visited = {inst_sg.anchor_id}

        mapping[inst_sg.anchor_id] = repr_sg.anchor_id

        while repr_queue and inst_queue:
            repr_id = repr_queue.popleft()
            inst_id = inst_queue.popleft()

            repr_children = self._get_sorted_children(dag, repr_id, repr_sg.internal_edges)
            inst_children = self._get_sorted_children(dag, inst_id, inst_sg.internal_edges)

            # 健壮性检查：子节点数量应该相同
            if len(repr_children) != len(inst_children):
                logger.warning(
                    f"子节点数量不匹配: repr={len(repr_children)}, "
                    f"inst={len(inst_children)} (节点: {repr_id} vs {inst_id})"
                )

            for r_child, i_child in zip(repr_children, inst_children):
                if r_child not in repr_visited and i_child not in inst_visited:
                    mapping[i_child] = r_child
                    repr_visited.add(r_child)
                    inst_visited.add(i_child)
                    repr_queue.append(r_child)
                    inst_queue.append(i_child)

        return mapping

    def _get_sorted_children(self, dag: ExecutionDAG, node_id: int,
                            internal_edges: List[Tuple]) -> List[int]:
        """获取排序后的子节点"""
        children = []
        for src, tgt, etype in internal_edges:
            if src == node_id:
                tgt_node = dag.nodes[tgt]
                tgt_name = self._normalize_name(tgt_node.name)
                children.append((etype, tgt_name, tgt))

        children.sort()
        return [c[2] for c in children]

    def _create_merged_nodes(self, dag: ExecutionDAG) -> List[DAGNode]:
        """创建合并后的节点"""
        merged_nodes = []

        repr_to_instances = defaultdict(list)
        for inst_id, repr_id in self.node_mapping.items():
            repr_to_instances[repr_id].append(inst_id)

        for repr_id, inst_ids in repr_to_instances.items():
            all_ids = [repr_id] + inst_ids
            repr_node = dag.nodes[repr_id]

            merged = DAGNode(
                node_id=repr_id,
                name=self._normalize_name(repr_node.name),
                node_type=repr_node.node_type
            )

            merged.performance = self._aggregate_performance(dag, all_ids)

            merged.attributes = deepcopy(repr_node.attributes)
            merged.attributes['cluster_info'] = {
                'is_clustered': True,
                'instance_count': len(all_ids),
                'original_ids': all_ids,
                'representative_id': repr_id
            }

            merged_nodes.append(merged)

        return merged_nodes

    def _aggregate_performance(self, dag: ExecutionDAG, node_ids: List[int]) -> Dict:
        """聚合性能数据"""
        exec_times = []
        memories = []

        for nid in node_ids:
            node = dag.nodes[nid]
            exec_times.append(node.performance.get('execution_time', 0.0))
            memories.append(node.performance.get('memory_usage', 0))

        exec_stats = self._compute_stats(exec_times)

        return {
            'execution_time': sum(exec_times),
            'memory_usage': max(memories) if memories else 0,
            'call_count': len(node_ids),
            'performance_stats': {
                'execution_time': {
                    'total': sum(exec_times),
                    'mean': exec_stats['mean'],
                    'std': exec_stats['std'],
                    'min': exec_stats['min'],
                    'max': exec_stats['max'],
                    'instances': exec_times
                },
                'memory_usage': {
                    'max': max(memories) if memories else 0,
                    'total': sum(memories),
                    'instances': memories
                }
            }
        }

    def _compute_stats(self, values: List[float]) -> Dict:
        """纯Python实现统计计算"""
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5

        return {
            'mean': mean,
            'std': std,
            'min': min(values),
            'max': max(values)
        }

    def _rebuild_edges(self, old_dag: ExecutionDAG, new_dag: ExecutionDAG):
        """重建边"""
        internal_edges = set()

        # 内部边：代表子图的节点ID不在node_mapping中
        # 所以 get(src, src) 返回原ID，正确添加代表子图的内部边
        for hash_value, subgraphs in self.hash_groups.items():
            repr_sg = subgraphs[0]
            for src, tgt, etype in repr_sg.internal_edges:
                new_src = self.node_mapping.get(src, src)
                new_tgt = self.node_mapping.get(tgt, tgt)
                internal_edges.add((new_src, new_tgt, etype))

        new_dag.edges.extend(list(internal_edges))

        # 外部边：收集并记录multiplicity
        external_edge_counts = defaultdict(int)
        for src, tgt, etype in old_dag.edges:
            new_src = self.node_mapping.get(src, src)
            new_tgt = self.node_mapping.get(tgt, tgt)
            edge_key = (new_src, new_tgt, etype)

            if edge_key not in internal_edges:
                external_edge_counts[edge_key] += 1

        for edge_key, count in external_edge_counts.items():
            new_dag.edges.append(edge_key)
            if count > 1:
                logger.debug(f"外部边 {edge_key[:2]} 合并了 {count} 条重复边")

        # 验证边完整性
        for src, tgt, etype in new_dag.edges:
            if src not in new_dag.nodes:
                logger.error(f"悬空边: 源节点 {src} 不存在")
            if tgt not in new_dag.nodes:
                logger.error(f"悬空边: 目标节点 {tgt} 不存在")

    def _validate_cluster_consistency(self, new_dag: ExecutionDAG):
        """验证聚类一致性：同一hash组的所有节点instance_count相同"""
        for hash_value, subgraphs in self.hash_groups.items():
            expected_count = len(subgraphs)

            # 检查这个模式的所有节点
            for node_id in subgraphs[0].nodes:
                if node_id in new_dag.nodes:
                    node = new_dag.nodes[node_id]
                    if hasattr(node, 'attributes') and 'cluster_info' in node.attributes:
                        actual = node.attributes['cluster_info']['instance_count']
                        if actual != expected_count:
                            logger.error(
                                f"Hash {hash_value[:8]}: 节点{node_id} "
                                f"instance_count={actual}, 期望={expected_count}"
                            )

    def get_stats(self) -> Dict:
        """获取聚类统计信息"""
        total_groups = len(self.hash_groups)
        pattern_details = []
        total_nodes_saved = 0

        for h, subgraphs in self.hash_groups.items():
            # 实际计算节省的节点数
            nodes_saved = sum(len(sg.nodes) for sg in subgraphs[1:])
            total_nodes_saved += nodes_saved

            pattern_details.append({
                'hash': h[:8],
                'instance_count': len(subgraphs),
                'nodes_per_instance': len(subgraphs[0].nodes),
                'nodes_saved': nodes_saved
            })

        return {
            'pattern_count': total_groups,
            'total_instances': sum(len(sgs) for sgs in self.hash_groups.values()),
            'nodes_saved': total_nodes_saved,
            'pattern_details': pattern_details
        }
