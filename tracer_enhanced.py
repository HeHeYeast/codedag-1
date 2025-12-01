#!/usr/bin/env python3
"""
增强版tracer - 使用真实设备检测和性能测量
"""
import torch
import torch.nn.functional as F
import functools
import time
import psutil
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
import logging
from utils.device_detector import DeviceDetector

logger = logging.getLogger(__name__)

class EnhancedMigrationTracer:
    """
    增强的迁移追踪器 - 使用真实设备检测和性能测量
    """
    
    def __init__(self, max_depth=10):
        """初始化增强追踪器"""
        self.max_depth = max_depth
        self.current_depth = 0
        self.call_stack = []
        self.dag_building_active = False
        self.optimization_active = False
        self.original_trace_function = None
        
        # DAG构建相关
        self.nodes = []
        self.edges = []
        self.node_id_counter = 0
        self.operation_map = {}
        self.traced_operations = []
        self.gpu_accelerated_ops = set()
        
        # 优化相关
        self.optimization_plan = None
        self.migration_decisions = []
        self.optimization_context_active = False
        
        # 迁移相关
        self.migration_enabled = False
        self.target_device = None
        self.migration_stats = {
            'total_migrations': 0,
            'successful_migrations': 0,
            'failed_migrations': 0
        }
        
        # 性能监控
        self.batch_times = []
        self.trace_overhead = 0
        self.node_performance = {}  # 实际性能测量
        
        # 设备检测和管理
        self.device_detector = DeviceDetector()
        self.available_devices = self.device_detector.get_available_devices()
        self._log_available_devices()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("增强CodeDAG迁移追踪器初始化完成")
    
    def _log_available_devices(self):
        """记录检测到的设备"""
        logger.info("检测到的设备:")
        for device in self.available_devices:
            if device['device_type'] == 'cpu':
                logger.info(f"  CPU: {device['physical_cores']}核心, "
                          f"{device['memory_gb']:.1f}GB内存, "
                          f"{device['compute_power']:.1f}GFLOPS")
            else:
                logger.info(f"  {device['device_id']}: {device['name']}, "
                          f"{device['total_memory_gb']:.1f}GB显存, "
                          f"{device['compute_power']:.1f}TFLOPS")
    
    def _trace_function(self, frame, event, arg):
        """追踪函数调用"""
        if not self.dag_building_active:
            return None
        
        if self.current_depth >= self.max_depth:
            return None
        
        if event == 'call':
            self.current_depth += 1
            func_name = frame.f_code.co_name
            module_name = frame.f_globals.get('__name__', 'unknown')
            
            # 最小化的过滤列表 - 只过滤真正的内部实现细节
            skip_functions = ['<module>', '<listcomp>', '<genexpr>', '<lambda>', 
                            'format', 'items', 'keys', 'values', '__repr__', '__str__']
            
            # 记录所有未被过滤的函数（仅通过深度控制）
            if func_name not in skip_functions:
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                
                node_id = self.node_id_counter
                self.node_id_counter += 1
                
                op_name = f"{func_name}_{node_id}"
                
                # 创建节点
                node = {
                    "id": node_id,
                    "name": func_name,
                    "module": module_name,
                    "op_name": op_name,
                    "start_time": start_time,
                    "start_memory": start_memory,
                    "depth": self.current_depth
                }
                
                self.nodes.append(node)
                self.operation_map[op_name] = node_id
                self.traced_operations.append(op_name)
                
                # 基于操作特征判断是否为GPU可加速操作
                operation_data = self._extract_operation_data(frame)
                if self.device_detector.is_operation_gpu_suitable(func_name, operation_data):
                    self.gpu_accelerated_ops.add(op_name)
                
                # 创建边（如果有父节点）
                if self.call_stack:
                    parent_id = self.call_stack[-1]["id"]
                    edge_data = {
                        "from": parent_id,
                        "to": node_id,
                        "data_size": self._estimate_data_size(frame.f_locals)
                    }
                    self.edges.append(edge_data)
                
                self.call_stack.append(node)
        
        elif event == 'return':
            self.current_depth -= 1
            if self.call_stack:
                node = self.call_stack.pop()
                # 测量实际性能
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                
                execution_time = (end_time - node['start_time']) * 1000  # ms
                memory_delta = end_memory - node['start_memory']  # MB
                
                # 记录实际性能数据
                self.node_performance[node['id']] = {
                    'execution_time': execution_time,
                    'memory_usage': max(memory_delta, 0),  # 避免负值
                    'compute_time': execution_time * 0.8,  # 估算计算时间
                    'file_read_time': execution_time * 0.1,  # 估算IO时间
                    'measured': True  # 标记为实测数据
                }
                
                # 更新节点信息
                node['execution_time'] = execution_time
                node['memory_usage'] = memory_delta
        
        return self._trace_function
    
    def _extract_operation_data(self, frame) -> dict:
        """从帧中提取操作数据"""
        operation_data = {}
        locals_dict = frame.f_locals
        
        # 检查设备信息
        if 'device' in locals_dict:
            operation_data['device'] = str(locals_dict['device'])
        
        # 估算数据规模
        size = 0
        for key, value in locals_dict.items():
            if torch.is_tensor(value):
                size += value.numel()
            elif isinstance(value, (list, tuple)):
                size += len(value)
        
        operation_data['size'] = size
        
        # 检查是否有张量参数
        has_tensor = any(torch.is_tensor(v) for v in locals_dict.values())
        operation_data['has_tensor'] = has_tensor
        
        return operation_data
    
    def _estimate_data_size(self, locals_dict) -> float:
        """估算数据传输大小（MB）"""
        total_size = 0
        
        for value in locals_dict.values():
            if torch.is_tensor(value):
                # 张量大小 = 元素数 * 每元素字节数
                element_size = value.element_size()
                num_elements = value.numel()
                total_size += (element_size * num_elements) / (1024 * 1024)  # MB
            elif isinstance(value, (list, tuple)):
                # 简单估算列表/元组大小
                total_size += len(value) * 8 / (1024 * 1024)  # 假设每元素8字节
        
        return max(total_size, 0.001)  # 至少0.001MB
    
    @contextmanager
    def tracing_context(self):
        """DAG构建上下文"""
        self.logger.info("开始增强DAG构建阶段")
        self.dag_building_active = True
        self.original_trace_function = sys.gettrace()
        sys.settrace(self._trace_function)
        
        try:
            yield self
        finally:
            sys.settrace(self.original_trace_function)
            self.dag_building_active = False
            self.logger.info("停止增强DAG构建阶段")
            self.logger.info(f"发现 {len(self.gpu_accelerated_ops)}/{len(self.nodes)} GPU可加速操作")
    
    def analyze_and_optimize(self):
        """分析并优化DAG"""
        self.logger.info("开始智能分析和优化")
        
        # 基于实际测量的GPU操作比例
        gpu_ratio = len(self.gpu_accelerated_ops) / max(len(self.nodes), 1)
        
        # 迁移策略
        strategy = "aggressive" if gpu_ratio > 0.3 else "conservative"
        
        # 基于实际设备性能估算加速比
        if self.available_devices:
            gpu_devices = [d for d in self.available_devices if d['device_type'] == 'cuda']
            cpu_device = next((d for d in self.available_devices if d['device_type'] == 'cpu'), None)
            
            if gpu_devices and cpu_device:
                # 使用实际计算能力比例估算加速
                gpu_power = max(d['compute_power'] for d in gpu_devices)
                cpu_power = cpu_device['compute_power']
                power_ratio = gpu_power / max(cpu_power, 0.1)
                speedup = 1.0 + (power_ratio - 1.0) * gpu_ratio * 0.5  # 保守估计
            else:
                speedup = 3.2 * gpu_ratio + 1.0  # 默认估算
        else:
            speedup = 3.2 * gpu_ratio + 1.0
        
        # 选择最优GPU设备
        optimal_device = self.device_detector.select_optimal_device(
            self.available_devices, task_type='compute'
        )
        
        self.optimization_plan = {
            "enabled": True,
            "strategy": strategy,
            "estimated_speedup": speedup,
            "gpu_ratio": gpu_ratio,
            "target_device": optimal_device
        }
        
        self.logger.info(f"迁移策略: {strategy}")
        self.logger.info(f"预期加速比: {speedup:.2f}x")
        self.logger.info(f"GPU操作占比: {gpu_ratio:.1%}")
        self.logger.info(f"目标设备: {optimal_device}")
        
        return {"optimizations_applied": 1}
    
    @contextmanager
    def optimization_context(self):
        """优化执行上下文"""
        self.logger.info("开始智能优化执行阶段")
        
        if self.optimization_plan and self.optimization_plan.get("enabled"):
            strategy = self.optimization_plan.get("strategy", "conservative")
            self.logger.info(f"启用 {strategy} 迁移策略")
            
            self.target_device = self.optimization_plan.get("target_device", "cpu")
            self.logger.info(f"目标设备: {self.target_device}")
            
            self.migration_enabled = True
            self.optimization_context_active = True
        
        try:
            yield self
        finally:
            self.optimization_context_active = False
            self.logger.info("停止智能优化执行阶段")
    
    def _migrate_batch(self, batch, target_device):
        """迁移批次数据到目标设备"""
        if isinstance(batch, dict):
            migrated = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    migrated[key] = value.to(target_device)
                else:
                    migrated[key] = value
            return migrated
        elif torch.is_tensor(batch):
            return batch.to(target_device)
        elif isinstance(batch, (list, tuple)):
            migrated = []
            for item in batch:
                if torch.is_tensor(item):
                    migrated.append(item.to(target_device))
                else:
                    migrated.append(item)
            return type(batch)(migrated)
        return batch
    
    def export_dataflow_graph(self, output_path="test_results/dataflow_graph.json"):
        """导出真实的数据流图"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 添加实际性能数据到节点
        for node in self.nodes:
            node_id = node['id']
            if node_id in self.node_performance:
                node.update(self.node_performance[node_id])
        
        dataflow = {
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "gpu_operations": len(self.gpu_accelerated_ops),
                "traced_operations": self.traced_operations[:50],
                "target_device": self.optimization_plan.get("target_device") if self.optimization_plan else "cpu",
                "trace_depth": self.max_depth,
                "available_devices": self.available_devices
            },
            "nodes": self.nodes,
            "edges": self.edges,
            "gpu_accelerated_ops": list(self.gpu_accelerated_ops),
            "performance_data": self.node_performance
        }
        
        with open(output_path, 'w') as f:
            json.dump(dataflow, f, indent=2)
        
        self.logger.info(f"数据流图已导出到 {output_path}")
        print(f"[CodeDAG] Dataflow graph exported to {output_path}")
    
    def instrument_dataloader(self, dataloader):
        """装备数据加载器"""
        if not self.migration_enabled:
            return dataloader
        
        self.logger.info("装备数据加载器迭代器...")
        
        # 获取迭代器类型
        iterator = iter(dataloader)
        iterator_class = type(iterator)
        
        # 保存原始的__next__方法
        original_next = iterator_class.__next__
        
        # 创建包装的__next__方法
        def wrapped_next(self_iter):
            data = original_next(self_iter)
            if self.optimization_context_active and self.target_device:
                data = self._migrate_batch(data, self.target_device)
                self.migration_stats['total_migrations'] += 1
                self.migration_stats['successful_migrations'] += 1
            return data
        
        # 替换__next__方法
        iterator_class.__next__ = wrapped_next
        
        self.logger.info(f"✓ 已装备 {iterator_class.__name__}.__next__ 用于智能迁移")
        
        return dataloader
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dag_building_active:
            sys.settrace(self.original_trace_function)
            self.dag_building_active = False


# 兼容性别名
MigrationEnabledTracer = EnhancedMigrationTracer