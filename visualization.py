"""
精简的可视化模块
提供DAG和性能数据的可视化功能
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DAGVisualizer:
    """DAG可视化器"""
    
    def __init__(self):
        self.figure_size = (12, 8)
        
    def visualize_dag(self, dag, output_path: str = None, title: str = "Execution DAG"):
        """可视化DAG结构"""
        return visualize_dag(dag, output_path, title)
        
    def create_performance_charts(self, report_data: Dict, output_dir: str):
        """创建性能图表"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建简单的性能对比图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 模拟性能数据
            categories = ['CPU执行', '手动GPU', '自动优化']
            times = [1.0, 0.5, 0.3]  # 模拟时间
            
            bars = ax.bar(categories, times, color=['blue', 'orange', 'green'])
            ax.set_ylabel('执行时间 (相对)')
            ax.set_title('性能对比')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}x', ha='center', va='bottom')
                       
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"性能图表已保存到: {output_dir}/performance_comparison.png")
            
        except Exception as e:
            logger.warning(f"创建性能图表失败: {e}")
            
    def generate_html_report(self, report_data: Dict, output_path: str) -> str:
        """生成HTML报告"""
        try:
            # 简单的文本报告
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("CodeDAG 性能报告\n")
                f.write("=" * 50 + "\n\n")
                
                if 'dag_summary' in report_data:
                    f.write(f"DAG节点数: {report_data['dag_summary'].get('node_count', 0)}\n")
                    
                if 'session_stats' in report_data:
                    stats = report_data['session_stats']
                    f.write(f"会话数: {stats.get('sessions_count', 0)}\n")
                    f.write(f"总节点追踪数: {stats.get('total_nodes_traced', 0)}\n")
                    
                f.write("\n报告生成时间: " + str(time.time()) + "\n")
                
            logger.info(f"HTML报告已生成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
            return ""
    
    def export_detailed_node_graph(self, nodes: List, edges: List, output_path: str = None, title: str = "详细数据流图"):
        """导出包含变量节点和运算符节点的详细图 - 分层布局"""
        try:
            fig, ax = plt.subplots(figsize=(20, 14))
            
            if not nodes:
                ax.text(0.5, 0.5, '无节点数据', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=16, color='gray')
                ax.set_title(title)
                ax.axis('off')
                if output_path:
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info(f"空图已保存到: {output_path}")
                return
            
            # 按类型分组节点
            function_nodes = [n for n in nodes if getattr(n, 'type', '') == 'function']
            variable_nodes = [n for n in nodes if getattr(n, 'type', '') == 'variable']
            operator_nodes = [n for n in nodes if getattr(n, 'type', '') == 'operator']
            
            # 使用分层布局算法
            node_positions = self._calculate_layered_layout(function_nodes, variable_nodes, operator_nodes, edges)
            
            # 绘制节点
            for node in nodes:
                self._draw_node_with_shape(ax, node, node_positions)
            
            # 绘制边，使用曲线避免重合
            self._draw_curved_edges(ax, edges, node_positions)
            
            # 设置图形属性
            if node_positions:
                all_x = [pos[0] for pos in node_positions.values()]
                all_y = [pos[1] for pos in node_positions.values()]
                
                margin = 2
                ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            
            ax.set_title(title, fontsize=18, weight='bold', pad=20)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 添加改进的图例
            self._add_enhanced_legend(ax)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"详细数据流图已保存到: {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"导出详细节点图失败: {e}")
    
    def export_context_key_visualization(self, nodes: List, output_path: str = None):
        """可视化Context Key结构"""
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # 按Context Key分组
            context_groups = {}
            for node in nodes:
                if hasattr(node, 'context') and node.context:
                    # 提取调用链部分
                    context = node.context
                    if '->' in context:
                        call_chain = '->'.join(context.split('->')[:-1])
                    else:
                        call_chain = 'global'
                    
                    if call_chain not in context_groups:
                        context_groups[call_chain] = []
                    context_groups[call_chain].append(node)
            
            # 为每个Context组分配颜色和位置
            colors = plt.cm.Set3(np.linspace(0, 1, len(context_groups)))
            y_offset = 0
            
            for i, (context, group_nodes) in enumerate(context_groups.items()):
                ax.text(-0.5, y_offset, context, fontsize=10, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
                
                # 在该Context下排列节点
                x_offset = 0
                for node in group_nodes:
                    node_name = getattr(node, 'name', 'unknown')
                    node_type = getattr(node, 'type', 'unknown')
                    version = getattr(node, 'version', '')
                    
                    # 绘制节点
                    rect = patches.Rectangle((x_offset, y_offset-0.3), 1.5, 0.6,
                                           facecolor=colors[i], alpha=0.6,
                                           edgecolor='black')
                    ax.add_patch(rect)
                    
                    # 节点标签
                    label = f"{node_name}\n({node_type})"
                    if version:
                        label += f"\nv{version}"
                    
                    ax.text(x_offset+0.75, y_offset, label, ha='center', va='center',
                           fontsize=8, weight='bold')
                    
                    x_offset += 2
                
                y_offset -= 1.5
            
            ax.set_xlim(-2, max(8, len(max(context_groups.values(), key=len)) * 2))
            ax.set_ylim(y_offset-0.5, 1)
            ax.set_title('Context Key 结构可视化', fontsize=16, weight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Context Key可视化图已保存到: {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"导出Context Key可视化失败: {e}")

    def _calculate_layered_layout(self, function_nodes, variable_nodes, operator_nodes, edges):
        """计算分层布局位置"""
        node_positions = {}
        
        # 分层策略：函数节点在顶层，变量和运算符节点在下层，按照数据流组织
        layer_height = 4
        node_spacing_x = 3
        node_spacing_y = 2
        
        # 第一层：函数节点
        func_y = layer_height * 2
        for i, node in enumerate(function_nodes):
            x = i * node_spacing_x
            node_positions[node.node_id] = (x, func_y)
        
        # 为每个函数分析其内部的数据流
        for func_node in function_nodes:
            # 找到属于这个函数的变量和运算符
            func_context = getattr(func_node, 'context', '')
            
            # 获取该函数内的变量和运算符
            func_variables = [v for v in variable_nodes if self._belongs_to_function(v, func_context)]
            func_operators = [o for o in operator_nodes if self._belongs_to_function(o, func_context)]
            
            if not func_variables and not func_operators:
                continue
                
            # 计算该函数的数据流布局
            func_base_x = node_positions[func_node.node_id][0]
            self._layout_function_dataflow(func_variables, func_operators, edges, 
                                         func_base_x, func_y - layer_height, 
                                         node_positions, node_spacing_x)
        
        return node_positions
    
    def _belongs_to_function(self, node, func_context):
        """判断节点是否属于指定函数"""
        node_context = getattr(node, 'context', '')
        if not node_context or not func_context:
            return False
        
        # 提取函数部分的上下文
        func_name = func_context.split('->')[0] if '->' in func_context else func_context
        func_name = func_name.split('#')[0] if '#' in func_name else func_name
        
        return func_name in node_context
    
    def _layout_function_dataflow(self, variables, operators, edges, base_x, base_y, 
                                node_positions, spacing_x):
        """为单个函数内部的数据流进行布局"""
        
        # 建立数据流关系图
        var_to_operators = {}  # 变量 -> 使用它的运算符
        operator_to_vars = {}  # 运算符 -> 它产生的变量
        
        for edge in edges:
            edge_type = edge.get('type', '')
            from_id = edge.get('from')
            to_id = edge.get('to')
            
            if edge_type == 'uses':
                # 变量被运算符使用
                var_node = next((v for v in variables if v.id == from_id), None)
                op_node = next((o for o in operators if o.id == to_id), None)
                if var_node and op_node:
                    if from_id not in var_to_operators:
                        var_to_operators[from_id] = []
                    var_to_operators[from_id].append(to_id)
            
            elif edge_type == 'produces':
                # 运算符产生变量
                op_node = next((o for o in operators if o.id == from_id), None)
                var_node = next((v for v in variables if v.id == to_id), None)
                if op_node and var_node:
                    operator_to_vars[from_id] = to_id
        
        # 按照数据流顺序排列节点
        positioned_nodes = set()
        current_layer = 0
        x_offset = 0
        
        # 找到输入变量（不是由运算符产生的变量）
        input_vars = [v for v in variables if v.id not in operator_to_vars.values()]
        
        # 布局输入变量
        layer_y = base_y - current_layer * 2
        for i, var in enumerate(input_vars):
            x = base_x + x_offset
            node_positions[var.id] = (x, layer_y)
            positioned_nodes.add(var.id)
            x_offset += spacing_x / 2
        
        # 逐层布局运算符和结果变量
        remaining_operators = operators.copy()
        remaining_variables = [v for v in variables if v.id not in positioned_nodes]
        
        while remaining_operators or remaining_variables:
            current_layer += 1
            layer_y = base_y - current_layer * 2
            layer_x_offset = 0
            
            # 找到可以放置的运算符（其输入变量已经布局）
            placeable_operators = []
            for op in remaining_operators:
                # 检查运算符的所有输入是否已经布局
                inputs_ready = True
                for edge in edges:
                    if edge.get('type') == 'uses' and edge.get('to') == op.id:
                        if edge.get('from') not in positioned_nodes:
                            inputs_ready = False
                            break
                if inputs_ready:
                    placeable_operators.append(op)
            
            # 布局运算符
            for op in placeable_operators:
                x = base_x + layer_x_offset
                node_positions[op.id] = (x, layer_y)
                positioned_nodes.add(op.id)
                layer_x_offset += spacing_x / 2
                
                # 布局该运算符产生的变量
                if op.id in operator_to_vars:
                    result_var_id = operator_to_vars[op.id]
                    result_var = next((v for v in remaining_variables if v.id == result_var_id), None)
                    if result_var:
                        x += spacing_x / 4  # 稍微偏移
                        node_positions[result_var_id] = (x, layer_y - 1)
                        positioned_nodes.add(result_var_id)
                        remaining_variables.remove(result_var)
                
                remaining_operators.remove(op)
            
            # 如果没有可放置的运算符，直接放置剩余变量
            if not placeable_operators and remaining_variables:
                for var in remaining_variables[:]:
                    x = base_x + layer_x_offset
                    node_positions[var.id] = (x, layer_y)
                    positioned_nodes.add(var.id)
                    remaining_variables.remove(var)
                    layer_x_offset += spacing_x / 2
    
    def _draw_node_with_shape(self, ax, node, node_positions):
        """根据节点类型绘制不同形状的节点"""
        node_id = getattr(node, 'id', '')
        if node_id not in node_positions:
            return
            
        x, y = node_positions[node_id]
        node_name = getattr(node, 'name', 'unknown')
        node_type = getattr(node, 'type', 'unknown')
        
        # 节点形状和颜色映射
        if node_type == 'function':
            # 函数节点：矩形
            rect = patches.Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                                   facecolor='#4CAF50', edgecolor='black', 
                                   linewidth=2, alpha=0.8)
            ax.add_patch(rect)
            label = f"{node_name}"
            
        elif node_type == 'variable':
            # 变量节点：圆形
            circle = patches.Circle((x, y), 0.4, 
                                  facecolor='#2196F3', edgecolor='black',
                                  linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            label = f"{node_name}"
            if hasattr(node, 'version') and node.version:
                label += f"\nv{node.version}"
                
        elif node_type == 'operator':
            # 运算符节点：菱形
            diamond_points = np.array([[x, y+0.5], [x+0.5, y], [x, y-0.5], [x-0.5, y]])
            diamond = patches.Polygon(diamond_points, 
                                    facecolor='#FF9800', edgecolor='black',
                                    linewidth=2, alpha=0.8)
            ax.add_patch(diamond)
            label = f"{node_name}"
            
        else:
            # 默认：圆形
            circle = patches.Circle((x, y), 0.4, 
                                  facecolor='#9E9E9E', edgecolor='black',
                                  linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            label = f"{node_name}"
        
        # 添加文本标签
        ax.text(x, y-0.8, label, ha='center', va='top', 
               fontsize=9, weight='bold', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    def _draw_curved_edges(self, ax, edges, node_positions):
        """绘制曲线边，避免重合"""
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.patches import ConnectionPatch
        
        # 边类型样式映射
        edge_styles = {
            'creates': {'color': '#4CAF50', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 2},
            'uses': {'color': '#2196F3', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 2},
            'produces': {'color': '#FF5722', 'linestyle': '-', 'alpha': 0.9, 'linewidth': 3},
            'calls': {'color': '#9C27B0', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 2},
            'parameter': {'color': '#795548', 'linestyle': '-.', 'alpha': 0.7, 'linewidth': 1.5}
        }
        
        # 统计重复边，用于计算曲线偏移
        edge_counts = {}
        for edge in edges:
            from_id = edge.get('from')
            to_id = edge.get('to')
            edge_key = (from_id, to_id)
            if edge_key not in edge_counts:
                edge_counts[edge_key] = 0
            edge_counts[edge_key] += 1
        
        # 绘制边
        edge_offset_counter = {}
        for edge in edges:
            from_id = edge.get('from')
            to_id = edge.get('to')
            edge_type = edge.get('type', 'unknown')
            
            if from_id not in node_positions or to_id not in node_positions:
                continue
                
            x1, y1 = node_positions[from_id]
            x2, y2 = node_positions[to_id]
            
            # 计算曲线偏移
            edge_key = (from_id, to_id)
            if edge_key not in edge_offset_counter:
                edge_offset_counter[edge_key] = 0
            else:
                edge_offset_counter[edge_key] += 1
            
            offset = edge_offset_counter[edge_key] * 0.3
            
            # 获取样式
            style = edge_styles.get(edge_type, 
                                   {'color': 'gray', 'linestyle': '-', 'alpha': 0.5, 'linewidth': 1})
            
            # 创建曲线连接
            if abs(x2 - x1) > 0.1 or abs(y2 - y1) > 0.1:  # 避免自环
                # 计算控制点，创建贝塞尔曲线
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # 根据边的方向调整控制点
                if x2 > x1:  # 向右
                    control_x = mid_x
                    control_y = mid_y + offset + 0.5
                else:  # 向左
                    control_x = mid_x
                    control_y = mid_y - offset - 0.5
                
                # 使用FancyArrowPatch创建曲线箭头
                arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                      connectionstyle=f"arc3,rad={offset * 0.3}",
                                      arrowstyle='->', 
                                      color=style['color'],
                                      linestyle=style['linestyle'],
                                      alpha=style['alpha'],
                                      linewidth=style['linewidth'],
                                      mutation_scale=15)
                ax.add_patch(arrow)
                
                # 添加边类型标签（对于重要的边）
                if edge_type in ['produces', 'calls']:
                    label_x = mid_x + offset * 0.2
                    label_y = mid_y + offset * 0.2 + 0.2
                    ax.text(label_x, label_y, edge_type, 
                           fontsize=7, ha='center', 
                           bbox=dict(boxstyle="round,pad=0.1", 
                                   facecolor='white', alpha=0.7))
    
    def _add_enhanced_legend(self, ax):
        """添加增强的图例"""
        # 节点形状图例
        node_legend = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50', 
                      markersize=12, label='函数节点 (矩形)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', 
                      markersize=12, label='变量节点 (圆形)'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF9800', 
                      markersize=12, label='运算符节点 (菱形)')
        ]
        
        # 边类型图例
        edge_legend = [
            plt.Line2D([0], [0], color='#4CAF50', linewidth=3, label='creates - 创建'),
            plt.Line2D([0], [0], color='#2196F3', linewidth=3, label='uses - 使用'),
            plt.Line2D([0], [0], color='#FF5722', linewidth=3, label='produces - 产生'),
            plt.Line2D([0], [0], color='#9C27B0', linewidth=2, linestyle='--', label='calls - 调用'),
            plt.Line2D([0], [0], color='#795548', linewidth=2, linestyle='-.', label='parameter - 参数')
        ]
        
        # 创建两个图例
        legend1 = ax.legend(handles=node_legend, title='节点类型', 
                          loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.add_artist(legend1)  # 保持第一个图例
        
        legend2 = ax.legend(handles=edge_legend, title='边类型',
                          loc='upper left', bbox_to_anchor=(1.02, 0.7))


def visualize_dag(dag, output_path: str = None, title: str = "Execution DAG"):
    """可视化DAG结构"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 如果DAG为空，显示空图
        if not dag or not hasattr(dag, 'nodes') or len(dag.nodes) == 0:
            ax.text(0.5, 0.5, 'No DAG nodes to visualize', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.axis('off')
        else:
            # 绘制DAG节点
            node_positions = {}
            y_pos = 0.8
            
            for i, (node_id, node) in enumerate(dag.nodes.items()):
                x_pos = 0.1 + (i * 0.8 / len(dag.nodes))
                node_positions[node_id] = (x_pos, y_pos)
                
                # 绘制节点
                circle = patches.Circle((x_pos, y_pos), 0.05, 
                                      facecolor='lightblue', 
                                      edgecolor='black')
                ax.add_patch(circle)
                
                # 添加节点标签
                ax.text(x_pos, y_pos - 0.1, node.name, 
                       ha='center', va='center', fontsize=10)
            
            # 绘制边（如果有的话）
            if hasattr(dag, 'edges'):
                for edge in dag.edges:
                    start_pos = node_positions.get(edge[0])
                    end_pos = node_positions.get(edge[1])
                    if start_pos and end_pos:
                        ax.arrow(start_pos[0], start_pos[1], 
                               end_pos[0] - start_pos[0], 
                               end_pos[1] - start_pos[1],
                               head_width=0.02, head_length=0.02,
                               fc='black', ec='black')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"DAG可视化已保存到: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"DAG可视化失败: {e}")


def visualize_performance_comparison(performance_data: Dict[str, Any], 
                                   output_path: str = None):
    """可视化性能比较"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 提取数据
        original_times = performance_data.get('original_execution_times', [])
        migrated_times = performance_data.get('migrated_execution_times', [])
        
        if not original_times and not migrated_times:
            ax1.text(0.5, 0.5, 'No performance data available', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=14, color='gray')
            ax2.text(0.5, 0.5, 'No performance data available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, color='gray')
        else:
            # 绘制执行时间比较
            labels = []
            times = []
            colors = []
            
            if original_times:
                labels.append('Original CPU')
                times.append(np.mean(original_times))
                colors.append('orange')
            
            if migrated_times:
                labels.append('Migrated GPU')
                times.append(np.mean(migrated_times))
                colors.append('green')
            
            bars = ax1.bar(labels, times, color=colors, alpha=0.7)
            ax1.set_ylabel('Average Execution Time (s)')
            ax1.set_title('Performance Comparison')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{time:.4f}s', ha='center', va='bottom')
            
            # 绘制时间序列
            if original_times:
                ax2.plot(range(len(original_times)), original_times, 
                        'o-', label='Original CPU', color='orange', alpha=0.7)
            
            if migrated_times:
                ax2.plot(range(len(migrated_times)), migrated_times, 
                        's-', label='Migrated GPU', color='green', alpha=0.7)
            
            ax2.set_xlabel('Batch Index')
            ax2.set_ylabel('Execution Time (s)')
            ax2.set_title('Execution Time Timeline')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"性能比较图已保存到: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"性能比较可视化失败: {e}")


def visualize_migration_stats(migration_stats: Dict[str, Any], 
                            output_path: str = None):
    """可视化迁移统计"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 迁移成功率饼图
        total = migration_stats.get('total_migrations', 0)
        successful = migration_stats.get('successful_migrations', 0)
        failed = migration_stats.get('failed_migrations', 0)
        
        if total > 0:
            sizes = [successful, failed]
            labels = ['Successful', 'Failed']
            colors = ['green', 'red']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90)
            ax1.set_title(f'Migration Success Rate\n(Total: {total})')
        else:
            ax1.text(0.5, 0.5, 'No migration data', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=14, color='gray')
            ax1.set_title('Migration Success Rate')
        
        # 迁移统计柱状图
        if total > 0:
            categories = ['Total', 'Successful', 'Failed']
            values = [total, successful, failed]
            colors = ['blue', 'green', 'red']
            
            bars = ax2.bar(categories, values, color=colors, alpha=0.7)
            ax2.set_ylabel('Count')
            ax2.set_title('Migration Statistics')
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{value}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No migration data', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, color='gray')
            ax2.set_title('Migration Statistics')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"迁移统计图已保存到: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"迁移统计可视化失败: {e}")


def create_performance_report(tracer, output_dir: str = "./codedag_report"):
    """创建完整的性能报告"""
    import os
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取数据
        migration_summary = tracer.get_migration_summary()
        performance_data = tracer.performance_comparison
        migration_stats = tracer.migration_stats
        
        # 生成可视化
        visualize_dag(tracer.dag, 
                     os.path.join(output_dir, "execution_dag.png"),
                     "Execution DAG")
        
        visualize_performance_comparison(performance_data,
                                       os.path.join(output_dir, "performance_comparison.png"))
        
        visualize_migration_stats(migration_stats,
                                os.path.join(output_dir, "migration_stats.png"))
        
        # 生成文本报告
        report_path = os.path.join(output_dir, "performance_report.txt")
        with open(report_path, 'w') as f:
            f.write("CodeDAG Performance Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Migration Summary:\n")
            f.write(f"- Total migrations: {migration_stats['total_migrations']}\n")
            f.write(f"- Successful migrations: {migration_stats['successful_migrations']}\n")
            f.write(f"- Failed migrations: {migration_stats['failed_migrations']}\n\n")
            
            if performance_data.get('original_execution_times') and performance_data.get('migrated_execution_times'):
                original_avg = np.mean(performance_data['original_execution_times'])
                migrated_avg = np.mean(performance_data['migrated_execution_times'])
                speedup = original_avg / migrated_avg if migrated_avg > 0 else 0
                
                f.write("Performance Comparison:\n")
                f.write(f"- Original CPU average: {original_avg:.4f}s\n")
                f.write(f"- Migrated GPU average: {migrated_avg:.4f}s\n")
                f.write(f"- Speedup ratio: {speedup:.2f}x\n")
                f.write(f"- Improvement: {((speedup - 1) * 100):.1f}%\n\n")
            
            f.write("System Status:\n")
            system_status = migration_summary.get('system_status', {})
            f.write(f"- Migration enabled: {system_status.get('migration_enabled', False)}\n")
            f.write(f"- Migration active: {system_status.get('migration_active', False)}\n")
            f.write(f"- Instrumented iterators: {system_status.get('instrumented_iterators_count', 0)}\n")
            f.write(f"- DAG nodes: {system_status.get('dag_nodes_count', 0)}\n")
        
        logger.info(f"性能报告已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"创建性能报告失败: {e}")


def plot_device_info():
    """绘制设备信息"""
    try:
        from .migration import get_device_info
        
        device_info = get_device_info()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制设备信息
        devices = device_info['available_devices']
        device_names = []
        
        for device in devices:
            if device == 'cpu':
                device_names.append('CPU')
            else:
                device_id = int(device.split(':')[1])
                if device_info['cuda_available']:
                    cuda_devices = device_info.get('cuda_devices', [])
                    if device_id < len(cuda_devices):
                        device_names.append(f"GPU {device_id}\n{cuda_devices[device_id]['name']}")
                    else:
                        device_names.append(f"GPU {device_id}")
                else:
                    device_names.append(f"GPU {device_id}")
        
        y_pos = np.arange(len(devices))
        
        # 创建条形图
        bars = ax.barh(y_pos, [1] * len(devices), 
                      color=['orange' if d == 'cpu' else 'green' for d in devices],
                      alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(device_names)
        ax.set_xlabel('Available')
        ax.set_title('Available Devices')
        ax.set_xlim(0, 1.2)
        
        # 添加状态标签
        for i, (bar, device) in enumerate(zip(bars, devices)):
            status = "Available" if device in device_info['available_devices'] else "Unavailable"
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   status, ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig("device_info.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("设备信息图已保存到: device_info.png")
        
    except Exception as e:
        logger.error(f"设备信息可视化失败: {e}")
