"""
基于Graphviz的数据流图可视化器
提供自上而下的层次化布局，支持SVG矢量输出和HTML节点属性显示
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NodeStyle:
    """节点样式配置"""
    shape: str
    fillcolor: str
    style: str
    fontcolor: str = "black"
    fontsize: str = "10"
    fontname: str = "Arial"

class GraphvizDataflowVisualizer:
    """基于Graphviz的数据流图可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.node_styles = {
            'function': NodeStyle(
                shape='box',
                fillcolor='lightgreen',
                style='filled,rounded',
                fontsize='12',
                fontcolor='darkgreen'
            ),
            'variable': NodeStyle(
                shape='ellipse',
                fillcolor='lightblue',
                style='filled',
                fontsize='10',
                fontcolor='darkblue'
            ),
            'operator': NodeStyle(
                shape='diamond',
                fillcolor='orange',
                style='filled',
                fontsize='10',
                fontcolor='darkorange'
            )
        }
        
        self.edge_styles = {
            'creates': {'color': 'green', 'style': 'solid', 'penwidth': '2'},
            'uses': {'color': 'blue', 'style': 'solid', 'penwidth': '2'},
            'produces': {'color': 'red', 'style': 'bold', 'penwidth': '3'},
            'calls': {'color': 'purple', 'style': 'dashed', 'penwidth': '2'},
            'parameter': {'color': 'brown', 'style': 'dotted', 'penwidth': '1'}
        }
    
    def _check_graphviz_available(self) -> bool:
        """检查Graphviz是否可用"""
        try:
            import graphviz
            # 测试是否能创建简单图
            test_dot = graphviz.Digraph()
            test_dot.node('test', 'test')
            return True
        except ImportError:
            logger.warning("Graphviz Python包未安装，请安装: pip install graphviz")
            return False
        except Exception as e:
            logger.warning(f"Graphviz不可用: {e}")
            return False
    
    def _escape_html(self, text: str) -> str:
        """转义HTML特殊字符"""
        if not isinstance(text, str):
            text = str(text)
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
    
    def _create_node_label_html(self, node: Any) -> str:
        """创建包含完整属性信息的HTML标签"""
        node_id = getattr(node, 'node_id', 'unknown')
        node_name = self._escape_html(getattr(node, 'name', 'unknown'))
        node_type = getattr(node, 'node_type', 'unknown')
        
        # 构建HTML表格
        html_parts = [
            '<',
            '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">',
            f'<TR><TD COLSPAN="2" BGCOLOR="white"><B>{node_name}</B></TD></TR>',
            f'<TR><TD>ID</TD><TD>{node_id}</TD></TR>',
            f'<TR><TD>Type</TD><TD>{node_type}</TD></TR>'
        ]
        
        # 添加特定类型的属性
        if node_type == 'function':
            # 函数节点属性（从performance字典获取）
            if hasattr(node, 'performance') and 'execution_time_ms' in node.performance:
                time_str = f"{node.performance['execution_time_ms']:.3f}ms"
                html_parts.append(f'<TR><TD>Time</TD><TD>{time_str}</TD></TR>')

            if hasattr(node, 'performance') and 'peak_memory_mb' in node.performance:
                mem_str = f"{node.performance['peak_memory_mb']:.2f}MB"
                html_parts.append(f'<TR><TD>Memory</TD><TD>{mem_str}</TD></TR>')
            
            if hasattr(node, 'context'):
                context_raw = str(node.context)
                if len(context_raw) > 20:
                    context_raw = context_raw[:17] + "..."
                context_str = self._escape_html(context_raw)
                html_parts.append(f'<TR><TD>Context</TD><TD>{context_str}</TD></TR>')
        
        elif node_type == 'variable':
            # 变量节点属性
            if hasattr(node, 'version'):
                html_parts.append(f'<TR><TD>Version</TD><TD>v{node.version}</TD></TR>')

            # 从attributes['variable_snapshot']获取变量信息
            if hasattr(node, 'attributes') and 'variable_snapshot' in node.attributes:
                snapshot = node.attributes['variable_snapshot']
                if hasattr(snapshot, 'shape') and snapshot.shape:
                    shape_str = str(snapshot.shape)
                    html_parts.append(f'<TR><TD>Shape</TD><TD>{shape_str}</TD></TR>')

                if hasattr(snapshot, 'dtype') and snapshot.dtype:
                    dtype_str = self._escape_html(str(snapshot.dtype))
                    html_parts.append(f'<TR><TD>DType</TD><TD>{dtype_str}</TD></TR>')

                if hasattr(snapshot, 'device') and snapshot.device:
                    device_str = self._escape_html(str(snapshot.device))
                    html_parts.append(f'<TR><TD>Device</TD><TD>{device_str}</TD></TR>')

                if hasattr(snapshot, 'size_mb') and snapshot.size_mb:
                    size_str = f"{snapshot.size_mb:.3f}MB"
                    html_parts.append(f'<TR><TD>Size</TD><TD>{size_str}</TD></TR>')
        
        elif node_type == 'operator':
            # 运算符节点属性（从performance字典获取）
            if hasattr(node, 'performance') and 'execution_time_ms' in node.performance:
                time_str = f"{node.performance['execution_time_ms']:.3f}ms"
                html_parts.append(f'<TR><TD>Time</TD><TD>{time_str}</TD></TR>')
            
            if hasattr(node, 'context'):
                context_raw = str(node.context)
                if len(context_raw) > 20:
                    context_raw = context_raw[:17] + "..."
                context_str = self._escape_html(context_raw)
                html_parts.append(f'<TR><TD>Context</TD><TD>{context_str}</TD></TR>')
        
        html_parts.extend(['</TABLE>', '>'])
        return ''.join(html_parts)
    
    def _organize_nodes_by_layer(self, nodes: List[Any], edges: List[Dict]) -> Dict[int, List[Any]]:
        """按照数据流依赖关系组织节点分层"""
        # 构建依赖关系图
        dependencies = {}  # node_id -> set of dependencies
        dependents = {}    # node_id -> set of dependents
        
        for node in nodes:
            node_id = getattr(node, 'node_id', str(node))
            dependencies[node_id] = set()
            dependents[node_id] = set()
        
        # 分析边关系（统一使用元组格式）
        for edge in edges:
            from_id, to_id, edge_type = edge
            
            if from_id in dependencies and to_id in dependencies:
                # 数据流依赖：uses, produces 表示数据依赖
                if edge_type in ['uses', 'produces']:
                    dependencies[to_id].add(from_id)
                    dependents[from_id].add(to_id)
                # 调用关系：calls 表示执行依赖
                elif edge_type == 'calls':
                    dependencies[to_id].add(from_id)
                    dependents[from_id].add(to_id)
        
        # 拓扑排序分层
        layers = {}
        node_to_layer = {}
        
        # 第0层：没有依赖的节点
        remaining_nodes = set(dependencies.keys())
        current_layer = 0
        
        while remaining_nodes:
            # 找到当前可以放置的节点（所有依赖都已分层）
            current_layer_nodes = []
            for node_id in list(remaining_nodes):
                deps = dependencies[node_id]
                if all(dep in node_to_layer for dep in deps):
                    # 计算应该放在的层级
                    if deps:
                        required_layer = max(node_to_layer[dep] for dep in deps) + 1
                    else:
                        required_layer = 0
                    
                    current_layer_nodes.append((node_id, required_layer))
            
            if not current_layer_nodes:
                # 处理循环依赖，强制放置剩余节点
                for node_id in remaining_nodes:
                    current_layer_nodes.append((node_id, current_layer))
                break
            
            # 按层级分组
            for node_id, layer in current_layer_nodes:
                if layer not in layers:
                    layers[layer] = []
                
                # 找到对应的节点对象
                node_obj = next((n for n in nodes if getattr(n, 'node_id', str(n)) == node_id), None)
                if node_obj:
                    layers[layer].append(node_obj)
                
                node_to_layer[node_id] = layer
                remaining_nodes.remove(node_id)
            
            current_layer += 1
        
        return layers
    
    def _create_graphviz_graph(self, nodes: List[Any], edges: List[Dict], title: str = "Data Flow Graph") -> 'graphviz.Digraph':
        """创建Graphviz图对象"""
        try:
            import graphviz
        except ImportError:
            raise ImportError("需要安装graphviz: pip install graphviz")
        
        # 创建有向图
        dot = graphviz.Digraph(comment=title)
        
        # 设置图属性 - 自上而下的层次化布局
        dot.attr(rankdir='TB')  # Top to Bottom
        dot.attr(splines='spline')  # 使用样条线而非直角，支持边标签
        dot.attr(nodesep='0.8')  # 节点间距
        dot.attr(ranksep='1.2')  # 层间距
        dot.attr(bgcolor='white')
        dot.attr(fontname='Arial')
        dot.attr(fontsize='14')
        dot.attr(labelloc='t')  # 标题位置
        dot.attr(label=title)
        dot.attr(concentrate='true')  # 合并重复边
        
        # 按层组织节点
        layers = self._organize_nodes_by_layer(nodes, edges)
        
        # 创建节点
        for node in nodes:
            node_id = str(getattr(node, 'node_id', id(node)))
            node_type = getattr(node, 'node_type', 'unknown')
            
            # 获取样式
            style = self.node_styles.get(node_type, self.node_styles['variable'])
            
            # 创建HTML标签
            label = self._create_node_label_html(node)
            
            # 添加节点 - 修复HTML标签支持
            dot.node(
                node_id,
                label=label,
                shape=style.shape,
                style=style.style,
                fillcolor=style.fillcolor,
                fontname=style.fontname,
                fontsize=style.fontsize,
                fontcolor=style.fontcolor
            )
        
        # 使用subgraph约束同层节点
        for layer_num in sorted(layers.keys()):
            layer_nodes = layers[layer_num]
            if len(layer_nodes) > 1:
                with dot.subgraph() as subgraph:
                    subgraph.attr(rank='same')
                    for node in layer_nodes:
                        node_id = str(getattr(node, 'node_id', id(node)))
                        subgraph.node(node_id)
        
        # 创建边（统一使用元组格式）
        for edge in edges:
            from_id, to_id, edge_type = edge
            from_id = str(from_id)
            to_id = str(to_id)
            
            if from_id and to_id:
                # 获取边样式
                edge_style = self.edge_styles.get(edge_type, {'color': 'gray', 'style': 'solid', 'penwidth': '1'})
                
                # 添加边标签
                edge_label = edge_type
                
                dot.edge(
                    from_id,
                    to_id,
                    label=edge_label,
                    color=edge_style['color'],
                    style=edge_style['style'],
                    penwidth=edge_style['penwidth'],
                    fontsize='8',
                    fontcolor=edge_style['color']
                )
        
        return dot
    
    def generate_dataflow_svg(self, nodes: List[Any], edges: List[Dict], 
                            output_path: str = "/mnt/sda/gxy/codedag_clean/dataflow.svg",
                            title: str = "数据流图可视化") -> bool:
        """生成数据流图的SVG可视化
        
        Args:
            nodes: 节点列表
            edges: 边列表  
            output_path: 输出SVG文件路径
            title: 图标题
            
        Returns:
            bool: 是否成功生成
        """
        if not self._check_graphviz_available():
            logger.error("Graphviz不可用，无法生成SVG")
            return False
        
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 创建Graphviz图
            dot_graph = self._create_graphviz_graph(nodes, edges, title)
            
            # 生成SVG
            svg_path_without_ext = output_path.replace('.svg', '')
            dot_graph.render(svg_path_without_ext, format='svg', cleanup=True)
            
            logger.info(f"数据流图SVG已生成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成SVG失败: {e}")
            return False
    
    def generate_legend_svg(self, output_path: str = "/mnt/sda/gxy/codedag_clean/dataflow_legend.svg") -> bool:
        """生成图例的SVG
        
        Args:
            output_path: 输出SVG文件路径
            
        Returns:
            bool: 是否成功生成
        """
        if not self._check_graphviz_available():
            return False
        
        try:
            import graphviz
            
            # 创建图例图
            legend = graphviz.Digraph(comment="图例")
            legend.attr(rankdir='TB')
            legend.attr(bgcolor='white')
            legend.attr(label='数据流图图例')
            legend.attr(labelloc='t')
            
            # 节点类型示例
            legend.node('func_example', 
                       label='<函数节点<BR/>Function Node>',
                       shape='box', style='filled,rounded', 
                       fillcolor='lightgreen', fontcolor='darkgreen')
            
            legend.node('var_example',
                       label='<变量节点<BR/>Variable Node>',
                       shape='ellipse', style='filled',
                       fillcolor='lightblue', fontcolor='darkblue')
            
            legend.node('op_example',
                       label='<运算符节点<BR/>Operator Node>',
                       shape='diamond', style='filled',
                       fillcolor='orange', fontcolor='darkorange')
            
            # 边类型示例
            legend.edge('func_example', 'var_example', label='creates', color='green', penwidth='2')
            legend.edge('var_example', 'op_example', label='uses', color='blue', penwidth='2') 
            legend.edge('op_example', 'var_example', label='produces', color='red', penwidth='3', style='bold')
            
            # 生成图例SVG
            legend_path_without_ext = output_path.replace('.svg', '')
            legend.render(legend_path_without_ext, format='svg', cleanup=True)
            
            logger.info(f"图例SVG已生成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成图例失败: {e}")
            return False

def create_dataflow_visualization(tracer, output_path: str = "/mnt/sda/gxy/codedag_clean/dataflow.svg") -> bool:
    """便捷函数：从追踪器创建数据流可视化
    
    Args:
        tracer: 增强追踪器对象
        output_path: 输出SVG路径
        
    Returns:
        bool: 是否成功
    """
    try:
        # 获取节点和边数据（使用统一的DAG结构）
        if hasattr(tracer, 'dag_builder') and hasattr(tracer.dag_builder, 'dag'):
            nodes = list(tracer.dag_builder.dag.nodes.values())
            edges = tracer.dag_builder.dag.edges
        else:
            logger.error("追踪器没有dag_builder.dag属性")
            return False
        
        # 创建可视化器
        visualizer = GraphvizDataflowVisualizer()
        
        # 生成SVG
        success = visualizer.generate_dataflow_svg(nodes, edges, output_path)
        
        # 同时生成图例
        if success:
            legend_path = output_path.replace('.svg', '_legend.svg')
            visualizer.generate_legend_svg(legend_path)
        
        return success
        
    except Exception as e:
        logger.error(f"创建数据流可视化失败: {e}")
        return False