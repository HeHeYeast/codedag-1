"""
基础追踪器 - 核心追踪功能的基类
提供最小化的执行追踪和DAG构建能力
"""

import sys
import time
import logging
import functools
from typing import Dict, Set, Callable, Optional, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class BaseTracer:
    """
    基础追踪器类
    提供核心的函数追踪和上下文管理功能
    """
    
    def __init__(self, max_depth=3, enabled=True):
        self.max_depth = max_depth
        self.enabled = enabled
        self.is_tracing = False
        
        # 追踪状态
        self.call_stack = []
        self.current_depth = 0
        
        # 性能统计
        self.function_stats = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        })
        
        # 会话管理
        self.session_active = False
        self.session_data = {}
        
    def start_tracing_session(self):
        """开始新的追踪会话"""
        if not self.enabled:
            return
            
        self.session_active = True
        self.session_data = {}
        self._reset_stats()
        self.enable_tracing()  # 自动启用追踪
        logger.debug("追踪会话已开始")
        
    def stop_tracing_session(self):
        """停止当前追踪会话"""
        self.disable_tracing()  # 自动禁用追踪
        self.session_active = False
        logger.debug("追踪会话已停止")
        
    def _reset_stats(self):
        """重置统计数据"""
        self.function_stats.clear()
        self.call_stack.clear()
        self.current_depth = 0
        
    def _should_trace_call(self, frame) -> bool:
        """判断是否应该追踪此函数调用"""
        if not self.enabled or not self.session_active:
            return False
            
        if self.current_depth >= self.max_depth:
            return False
            
        return True
        
    def _trace_function_call(self, frame, event, arg):
        """追踪函数调用的核心逻辑"""
        if event == 'call':
            if self._should_trace_call(frame):
                self._on_function_enter(frame)
        elif event == 'return':
            if self.call_stack and self.current_depth > 0:
                self._on_function_exit(frame, arg)
                
        return self._trace_function_call
        
    def _on_function_enter(self, frame):
        """函数进入时的处理"""
        func_name = self._get_function_name(frame)
        start_time = time.perf_counter()
        
        call_info = {
            'function': func_name,
            'start_time': start_time,
            'depth': self.current_depth
        }
        
        self.call_stack.append(call_info)
        self.current_depth += 1
        
        logger.debug(f"进入函数: {func_name} (深度: {self.current_depth})")
        
    def _on_function_exit(self, frame, return_value):
        """函数退出时的处理"""
        if not self.call_stack:
            return
            
        call_info = self.call_stack.pop()
        self.current_depth -= 1
        
        end_time = time.perf_counter()
        execution_time = end_time - call_info['start_time']
        
        func_name = call_info['function']
        stats = self.function_stats[func_name]
        stats['call_count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['call_count']
        
        logger.debug(f"退出函数: {func_name}, 耗时: {execution_time:.6f}s")
        
    def _get_function_name(self, frame) -> str:
        """获取函数名称"""
        code = frame.f_code
        return f"{code.co_filename}:{code.co_name}:{code.co_firstlineno}"
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计数据"""
        return dict(self.function_stats)
        
    def enable_tracing(self):
        """启用追踪"""
        if not self.enabled:
            return
            
        self.is_tracing = True
        sys.settrace(self._trace_function_call)
        logger.debug("追踪已启用")
        
    def disable_tracing(self):
        """禁用追踪"""
        self.is_tracing = False
        sys.settrace(None)
        logger.debug("追踪已禁用")
        
    def __enter__(self):
        """上下文管理器入口"""
        self.start_tracing_session()
        self.enable_tracing()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disable_tracing()
        self.stop_tracing_session()