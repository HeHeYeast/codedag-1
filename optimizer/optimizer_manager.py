"""
优化管理器 - Optimizer Manager
统一管理整个优化流程：粗化 -> K路分割 -> 迭代优化
"""

import logging
from typing import Dict, List, Optional

try:
    from ..core.dag_builder import ExecutionDAG
    from .graph_coarsening import GraphCoarsening
    from .kway_partitioner import KWayPartitioner, Partition
    from .iterative_optimizer import IterativeOptimizer
except ImportError:
    import sys
    sys.path.append('/mnt/sda/gxy/codedag_clean')
    from core.dag_builder import ExecutionDAG
    from optimizer.graph_coarsening import GraphCoarsening
    from optimizer.kway_partitioner import KWayPartitioner, Partition
    from optimizer.iterative_optimizer import IterativeOptimizer

logger = logging.getLogger(__name__)


class OptimizerManager:
    """
    优化管理器

    提供简单接口：接收DAG -> 运行优化流程 -> 返回分区结果
    """

    def __init__(self,
                 # 粗化参数
                 coarsen_max_depth: int = 5,
                 coarsen_anchor_patterns: Optional[List[str]] = None,
                 # K路分割参数
                 k: int = 4,
                 lambda_weight: float = 0.5,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 # 迭代优化参数
                 iter_alpha: float = 0.4,
                 iter_beta: float = 0.6,
                 max_iterations: int = 200,
                 no_improvement_threshold: int = 20,
                 communication_overhead: float = 0.0001):
        """
        Args:
            coarsen_max_depth: 粗化子图提取最大深度
            coarsen_anchor_patterns: 粗化锚点模式列表
            k: 分区数量
            lambda_weight: 瓶颈选择距离权重
            alpha: kway性能权重
            beta: kway距离惩罚权重
            gamma: kway负载惩罚权重
            iter_alpha: 迭代优化负载均衡权重
            iter_beta: 迭代优化关键路径权重
            max_iterations: 最大迭代次数
            no_improvement_threshold: 连续无改善阈值
            communication_overhead: 跨分区通信开销系数
        """
        # 初始化各个优化器
        self.coarsener = GraphCoarsening(
            max_depth=coarsen_max_depth,
            anchor_patterns=coarsen_anchor_patterns or []
        )

        self.partitioner = KWayPartitioner(
            k=k,
            lambda_weight=lambda_weight,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        self.iterator = IterativeOptimizer(
            alpha=iter_alpha,
            beta=iter_beta,
            max_iterations=max_iterations,
            no_improvement_threshold=no_improvement_threshold,
            communication_overhead=communication_overhead
        )

        # 存储最近一次的结果（用于导出）
        self.last_result: Optional[Dict] = None

    def optimize(self, dag: ExecutionDAG, export_dir: Optional[str] = None) -> Dict:
        """
        执行完整的优化流程

        Args:
            dag: 原始ExecutionDAG对象（来自tracer）
            export_dir: 可选的导出目录，如果指定则自动导出每个阶段的结果

        Returns:
            优化结果字典，包含：
            - coarsened_dag: 粗化后的DAG
            - kway_partitions: K路分割结果
            - optimized_partitions: 优化后的分区列表
            - statistics: 各阶段统计信息
        """
        logger.info("=" * 60)
        logger.info("开始优化流程")
        logger.info(f"输入DAG: {len(dag.nodes)}个节点, {len(dag.edges)}条边")
        logger.info("=" * 60)

        # 阶段1：图粗化
        logger.info("\n阶段1: 图粗化聚类")
        logger.info("-" * 60)
        coarsened_dag = self.coarsener.coarsen(dag)
        coarsening_stats = self.coarsener.get_stats()
        logger.info(f"粗化完成: {len(dag.nodes)} -> {len(coarsened_dag.nodes)} 节点")
        logger.info(f"粗化统计: {coarsening_stats}")

        # 导出粗化结果
        if export_dir:
            self._export_coarsening_result(coarsened_dag, coarsening_stats, export_dir)

        # 阶段2：K路分割
        logger.info("\n阶段2: K路分割")
        logger.info("-" * 60)
        kway_partitions = self.partitioner.partition(coarsened_dag)
        partitioning_stats = self.partitioner.get_stats(kway_partitions)
        edge_cut_stats = self.partitioner.compute_edge_cut(kway_partitions, coarsened_dag)
        logger.info(f"K路分割完成: {partitioning_stats}")
        logger.info(f"边切割: {edge_cut_stats}")

        # 导出K路分割结果
        if export_dir:
            self._export_kway_result(kway_partitions, coarsened_dag, partitioning_stats, edge_cut_stats, export_dir)

        # 阶段3：迭代优化
        logger.info("\n阶段3: 迭代优化")
        logger.info("-" * 60)
        optimized_partitions = self.iterator.optimize(kway_partitions, coarsened_dag)
        iteration_stats = self.iterator.get_stats()
        logger.info(f"迭代优化完成: {iteration_stats}")

        # 导出优化结果
        if export_dir:
            self._export_optimized_result(optimized_partitions, coarsened_dag, iteration_stats, export_dir)

        # 汇总结果
        result = {
            'coarsened_dag': coarsened_dag,
            'kway_partitions': kway_partitions,
            'optimized_partitions': optimized_partitions,
            'statistics': {
                'original_nodes': len(dag.nodes),
                'original_edges': len(dag.edges),
                'coarsened_nodes': len(coarsened_dag.nodes),
                'coarsened_edges': len(coarsened_dag.edges),
                'coarsening': coarsening_stats,
                'partitioning': partitioning_stats,
                'edge_cut': edge_cut_stats,
                'iteration': iteration_stats
            }
        }

        # 保存最近结果
        self.last_result = result

        logger.info("\n" + "=" * 60)
        logger.info("优化流程完成")
        logger.info("=" * 60)

        return result

    def get_partition_summary(self, result: Dict) -> Dict:
        """
        生成分区摘要（用于tracer保存）

        Args:
            result: optimize()的返回结果

        Returns:
            分区摘要字典
        """
        partitions = result['partitions']
        stats = result['statistics']

        # 生成每个分区的详细信息
        partition_details = []
        for partition in partitions:
            partition_info = {
                'partition_id': partition.id,
                'center_node': partition.center,
                'node_count': len(partition.nodes),
                'node_ids': list(partition.nodes),
                'total_compute': partition.total_compute,
                'total_memory': partition.total_memory
            }
            partition_details.append(partition_info)

        # 计算负载均衡度
        loads = [p.total_compute for p in partitions]
        avg_load = sum(loads) / len(loads) if loads else 0
        max_load = max(loads) if loads else 0
        min_load = min(loads) if loads else 0
        load_balance = 1 - (max_load - min_load) / avg_load if avg_load > 0 else 1.0

        summary = {
            'partition_count': len(partitions),
            'partitions': partition_details,
            'load_balance': load_balance,
            'avg_load': avg_load,
            'max_load': max_load,
            'min_load': min_load,
            'edge_cut_ratio': stats['edge_cut'].get('cut_ratio', 0),
            'total_cut_edges': stats['edge_cut'].get('cut_edges', 0),
            'compression_ratio': (
                (1 - stats['coarsened_nodes'] / stats['original_nodes']) * 100
                if stats['original_nodes'] > 0 else 0
            ),
            'optimization_improvement': stats['iteration'].get('improvement_ratio', 0)
        }

        return summary

    def _export_coarsening_result(self, coarsened_dag, coarsening_stats, export_dir):
        """导出粗化结果（JSON + 可视化）"""
        import json
        from pathlib import Path

        output_dir = Path(export_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 导出JSON
        json_data = {
            'coarsening_statistics': coarsening_stats,
            'nodes': []
        }

        for node_id, node in coarsened_dag.nodes.items():
            node_info = {
                'id': node_id,
                'name': node.name,
                'type': node.node_type,
                'performance': node.performance
            }
            if hasattr(node, 'attributes') and 'cluster_info' in node.attributes:
                node_info['cluster_info'] = node.attributes['cluster_info']
            json_data['nodes'].append(node_info)

        json_file = output_dir / 'coarsening_result.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # 导出可视化
        dot_file = output_dir / 'coarsening_graph.dot'
        self._export_dag_dot(coarsened_dag, dot_file)

        logger.info(f"✓ 粗化结果已导出: {json_file}, {dot_file}")

    def _export_kway_result(self, partitions, coarsened_dag, partitioning_stats, edge_cut_stats, export_dir):
        """导出K路分割结果"""
        import json
        from pathlib import Path

        output_dir = Path(export_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 构建节点到分区的映射
        node_partition_map = {}
        for partition in partitions:
            for node_id in partition.nodes:
                node_partition_map[node_id] = partition.id

        # 导出JSON
        json_data = {
            'partitioning_statistics': partitioning_stats,
            'edge_cut_statistics': edge_cut_stats,
            'partition_count': len(partitions),
            'nodes': []
        }

        for node_id, node in coarsened_dag.nodes.items():
            node_info = {
                'id': node_id,
                'name': node.name,
                'type': node.node_type,
                'partition_id': node_partition_map.get(node_id, -1),
                'performance': node.performance
            }
            if hasattr(node, 'attributes') and 'cluster_info' in node.attributes:
                node_info['cluster_info'] = node.attributes['cluster_info']
            json_data['nodes'].append(node_info)

        json_file = output_dir / 'kway_partition.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ K路分割结果已导出: {json_file}")

    def _export_optimized_result(self, partitions, coarsened_dag, iteration_stats, export_dir):
        """导出优化后的分区结果"""
        import json
        from pathlib import Path

        output_dir = Path(export_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 构建节点到分区的映射
        node_partition_map = {}
        for partition in partitions:
            for node_id in partition.nodes:
                node_partition_map[node_id] = partition.id

        # 导出JSON
        json_data = {
            'iteration_statistics': iteration_stats,
            'partition_count': len(partitions),
            'nodes': []
        }

        for node_id, node in coarsened_dag.nodes.items():
            node_info = {
                'id': node_id,
                'name': node.name,
                'type': node.node_type,
                'partition_id': node_partition_map.get(node_id, -1),
                'performance': node.performance
            }
            if hasattr(node, 'attributes') and 'cluster_info' in node.attributes:
                node_info['cluster_info'] = node.attributes['cluster_info']
            json_data['nodes'].append(node_info)

        json_file = output_dir / 'optimized_partition.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ 优化后分区结果已导出: {json_file}")

        # 同时导出迁移计划
        self._export_migration_plan(partitions, coarsened_dag, export_dir)

    def _export_dag_dot(self, dag, output_path):
        """导出DAG可视化（DOT格式）"""
        dot_lines = ['digraph CoarsenedDAG {']
        dot_lines.append('  rankdir=TB;')
        dot_lines.append('  node [shape=box, style=filled];')

        # 添加节点
        for node_id, node in dag.nodes.items():
            label = node.name
            color = 'lightblue'

            # 聚类节点特殊标记
            if hasattr(node, 'attributes') and 'cluster_info' in node.attributes:
                instance_count = node.attributes['cluster_info']['instance_count']
                label += f'\\n×{instance_count}'
                color = 'lightgreen'

            dot_lines.append(f'  node{node_id} [label="{label}", fillcolor={color}];')

        # 添加边
        for src, tgt, etype in dag.edges:
            dot_lines.append(f'  node{src} -> node{tgt} [label="{etype}"];')

        dot_lines.append('}')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dot_lines))

        logger.info(f"✓ 可视化已导出: {output_path}")

    def _export_migration_plan(self, partitions, coarsened_dag, export_dir, gpu_count: int = 1):
        """
        导出迁移计划 (migration_plan.json)

        专门用于迁移模块，包含 context 字段和设备映射
        """
        import json
        import re
        from pathlib import Path

        output_dir = Path(export_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 构建节点到分区的映射
        node_partition_map = {}
        for partition in partitions:
            for node_id in partition.nodes:
                node_partition_map[node_id] = partition.id

        # 构建分区到设备的映射 (partition_id % gpu_count -> cuda:X)
        partition_device_map = {}
        for partition in partitions:
            if gpu_count > 0:
                device_idx = partition.id % gpu_count
                partition_device_map[partition.id] = f"cuda:{device_idx}"
            else:
                partition_device_map[partition.id] = "cpu"

        # 归一化 context key 的辅助函数
        def normalize_context(context: str) -> str:
            """去除调用计数 #数字"""
            if not context:
                return ""
            return re.sub(r'#\d+', '', context)

        # 构建 context -> device 映射
        context_device_map = {}
        nodes_info = []

        for node_id, node in coarsened_dag.nodes.items():
            partition_id = node_partition_map.get(node_id, -1)
            device = partition_device_map.get(partition_id, "cpu")

            # 获取 context (优先使用 node.context，否则用 name)
            raw_context = getattr(node, 'context', '') or node.name
            normalized_context = normalize_context(raw_context)

            # 添加到映射
            if normalized_context:
                context_device_map[normalized_context] = device

            # 节点详细信息
            node_info = {
                'id': node_id,
                'name': node.name,
                'type': node.node_type,
                'context': raw_context,
                'normalized_context': normalized_context,
                'partition_id': partition_id,
                'device': device,
                'performance': node.performance
            }

            if hasattr(node, 'attributes') and 'cluster_info' in node.attributes:
                node_info['cluster_info'] = node.attributes['cluster_info']

            nodes_info.append(node_info)

        # 构建迁移计划
        migration_plan = {
            'version': '1.0',
            'gpu_count': gpu_count,
            'partition_count': len(partitions),
            'partition_device_map': partition_device_map,
            'context_device_map': context_device_map,
            'nodes': nodes_info
        }

        json_file = output_dir / 'migration_plan.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(migration_plan, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ 迁移计划已导出: {json_file}")

    def get_migration_plan(self, partitions=None, dag=None, gpu_count: int = 1) -> Dict:
        """
        获取迁移计划（内存中直接返回，不写文件）

        用于在线模式：Tracer -> Optimizer -> Migration 直接传递

        Args:
            partitions: 分区列表，默认使用 last_result
            dag: DAG 对象，默认使用 last_result
            gpu_count: GPU 数量

        Returns:
            迁移计划字典
        """
        import re

        if partitions is None or dag is None:
            if self.last_result is None:
                raise ValueError("没有可用的优化结果，请先调用 optimize()")
            partitions = self.last_result['optimized_partitions']
            dag = self.last_result['coarsened_dag']

        # 构建节点到分区的映射
        node_partition_map = {}
        for partition in partitions:
            for node_id in partition.nodes:
                node_partition_map[node_id] = partition.id

        # 构建分区到设备的映射
        partition_device_map = {}
        for partition in partitions:
            if gpu_count > 0:
                device_idx = partition.id % gpu_count
                partition_device_map[partition.id] = f"cuda:{device_idx}"
            else:
                partition_device_map[partition.id] = "cpu"

        # 归一化 context key
        def normalize_context(context: str) -> str:
            if not context:
                return ""
            return re.sub(r'#\d+', '', context)

        # 构建 context -> device 映射
        context_device_map = {}
        for node_id, node in dag.nodes.items():
            partition_id = node_partition_map.get(node_id, -1)
            device = partition_device_map.get(partition_id, "cpu")

            raw_context = getattr(node, 'context', '') or node.name
            normalized_context = normalize_context(raw_context)

            if normalized_context:
                context_device_map[normalized_context] = device

        return {
            'version': '1.0',
            'gpu_count': gpu_count,
            'partition_count': len(partitions),
            'partition_device_map': partition_device_map,
            'context_device_map': context_device_map
        }
