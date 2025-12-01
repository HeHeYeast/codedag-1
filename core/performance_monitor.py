"""
性能监控器 - 专门负责性能数据收集和分析
集成内存监控、设备监控等功能
"""

import time
import logging
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """性能指标数据结构"""
    
    def __init__(self):
        self.execution_time = 0.0
        self.memory_usage = 0  # bytes
        self.cpu_usage = 0.0   # percentage
        self.gpu_memory = 0    # bytes
        self.device = 'cpu'
        self.call_count = 0
        
    def update(self, execution_time: float = 0.0, memory_usage: int = 0, 
               cpu_usage: float = 0.0, gpu_memory: int = 0, device: str = 'cpu'):
        """更新性能指标"""
        self.execution_time += execution_time
        self.memory_usage = max(self.memory_usage, memory_usage)
        self.cpu_usage = max(self.cpu_usage, cpu_usage)
        self.gpu_memory = max(self.gpu_memory, gpu_memory)
        self.device = device
        self.call_count += 1
        
    def get_avg_execution_time(self) -> float:
        """获取平均执行时间"""
        return self.execution_time / max(self.call_count, 1)


class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)  # 保存最近1000个样本
        
    def start_monitoring(self):
        """开始系统监控"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.debug("系统监控已启动")
        
    def stop_monitoring(self):
        """停止系统监控"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        logger.debug("系统监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.warning(f"系统监控错误: {e}")
                
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used': psutil.virtual_memory().used,
            'memory_available': psutil.virtual_memory().available
        }
        
        # GPU指标（如果可用）
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_reserved = torch.cuda.memory_reserved()
                metrics.update({
                    'gpu_memory_allocated': gpu_memory_allocated,
                    'gpu_memory_reserved': gpu_memory_reserved,
                    'gpu_utilization': self._get_gpu_utilization()
                })
            except Exception as e:
                logger.debug(f"GPU指标收集失败: {e}")
                
        return metrics
        
    def _get_gpu_utilization(self) -> float:
        """获取GPU利用率（简化版）"""
        # 这里可以集成nvidia-ml-py或其他GPU监控库
        return 0.0
        
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """获取当前系统指标"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}
            
        cpu_values = [m['cpu_percent'] for m in self.metrics_history]
        memory_values = [m['memory_percent'] for m in self.metrics_history]
        
        return {
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg': sum(memory_values) / len(memory_values),
            'memory_max': max(memory_values),
            'sample_count': len(self.metrics_history)
        }


class PerformanceMonitor:
    """
    性能监控器主类
    整合各种性能监控功能
    """
    
    def __init__(self, enable_system_monitor: bool = True):
        self.enable_system_monitor = enable_system_monitor
        self.system_monitor = SystemMonitor() if enable_system_monitor else None
        
        # 函数级性能指标
        self.function_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
        # 会话数据
        self.session_active = False
        self.session_start_time = 0.0
        
        # 回调函数
        self.metric_callbacks: List[Callable] = []
        
    def start_session(self):
        """开始性能监控会话"""
        self.session_active = True
        self.session_start_time = time.time()
        
        if self.system_monitor:
            self.system_monitor.start_monitoring()
            
        logger.debug("性能监控会话已开始")
        
    def stop_session(self):
        """停止性能监控会话"""
        self.session_active = False
        
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
            
        logger.debug("性能监控会话已停止")
        
    def record_function_performance(self, func_name: str, execution_time: float,
                                  memory_usage: int = 0, device: str = 'cpu'):
        """记录函数性能"""
        if not self.session_active:
            return
            
        metrics = self.function_metrics[func_name]
        
        # 获取当前系统指标
        current_system = self.system_monitor.get_current_metrics() if self.system_monitor else {}
        cpu_usage = current_system.get('cpu_percent', 0.0)
        gpu_memory = current_system.get('gpu_memory_allocated', 0)
        
        metrics.update(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            gpu_memory=gpu_memory,
            device=device
        )
        
        # 调用回调函数
        for callback in self.metric_callbacks:
            try:
                callback(func_name, metrics)
            except Exception as e:
                logger.warning(f"性能监控回调错误: {e}")
                
    def add_metric_callback(self, callback: Callable):
        """添加性能指标回调"""
        self.metric_callbacks.append(callback)
        
    def get_function_metrics(self) -> Dict[str, PerformanceMetrics]:
        """获取函数性能指标"""
        return dict(self.function_metrics)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'session_duration': time.time() - self.session_start_time if self.session_active else 0,
            'function_count': len(self.function_metrics),
            'total_execution_time': sum(m.execution_time for m in self.function_metrics.values()),
            'functions': {}
        }
        
        # 函数级摘要
        for func_name, metrics in self.function_metrics.items():
            summary['functions'][func_name] = {
                'call_count': metrics.call_count,
                'total_time': metrics.execution_time,
                'avg_time': metrics.get_avg_execution_time(),
                'memory_usage': metrics.memory_usage,
                'device': metrics.device
            }
            
        # 系统级摘要
        if self.system_monitor:
            summary['system'] = self.system_monitor.get_metrics_summary()
            
        return summary
        
    def reset_metrics(self):
        """重置所有性能指标"""
        self.function_metrics.clear()
        if self.system_monitor:
            self.system_monitor.metrics_history.clear()
        logger.debug("性能指标已重置")
        
    def export_metrics(self, format: str = 'dict') -> Any:
        """导出性能指标"""
        if format == 'dict':
            return self.get_performance_summary()
        elif format == 'json':
            import json
            return json.dumps(self.get_performance_summary(), indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
            
    def __enter__(self):
        """上下文管理器入口"""
        self.start_session()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_session()