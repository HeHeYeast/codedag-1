"""
内存分析器模块 - 追踪内存分配、使用和峰值
"""
import torch
import psutil
import tracemalloc
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    process_memory_mb: float  # 进程内存
    gpu_memory_mb: float  # GPU内存
    allocated_variables: Dict[str, float]  # 变量名 -> 大小(MB)
    
@dataclass
class VariableInfo:
    """变量信息"""
    name: str
    version: int
    shape: Tuple
    dtype: str
    size_mb: float
    device: str
    created_at: float
    context: str  # 调用上下文
    
@dataclass
class FunctionMemoryProfile:
    """函数内存概况"""
    function_name: str
    context: str
    call_count: int
    current_memory_mb: float
    peak_memory_mb: float
    allocated_memory_mb: float  # 该函数分配的内存
    freed_memory_mb: float  # 该函数释放的内存
    child_memory_mb: float  # 子函数使用的内存
    start_time: float
    end_time: float

class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self, track_gpu: bool = True):
        """
        初始化内存分析器
        
        Args:
            track_gpu: 是否追踪GPU内存
        """
        self.track_gpu = track_gpu and torch.cuda.is_available()
        
        # 内存追踪
        self.memory_snapshots: List[MemorySnapshot] = []
        self.function_profiles: Dict[str, FunctionMemoryProfile] = {}
        self.variable_registry: Dict[str, List[VariableInfo]] = defaultdict(list)
        
        # 当前状态
        self.current_context_stack: List[str] = []
        self.call_count_stack: List[Dict[str, int]] = [defaultdict(int)]
        self.memory_stack: List[float] = []
        
        # 峰值记录
        self.peak_memory_mb = 0
        self.peak_gpu_memory_mb = 0
        
        # Python内存追踪
        self.tracemalloc_enabled = False
        
    def start_tracking(self):
        """开始内存追踪"""
        if not self.tracemalloc_enabled:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            logger.info("内存追踪已启动")
    
    def stop_tracking(self):
        """停止内存追踪"""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
            logger.info("内存追踪已停止")
    
    def enter_function(self, func_name: str, module: str = "") -> str:
        """
        进入函数时调用
        
        Args:
            func_name: 函数名
            module: 模块名
            
        Returns:
            context: 调用上下文字符串
        """
        # 更新调用计数
        self.call_count_stack[-1][func_name] += 1
        call_count = self.call_count_stack[-1][func_name]
        
        # 构建上下文路径
        if self.current_context_stack:
            parent_context = self.current_context_stack[-1]
            context = f"{parent_context}->{func_name}#{call_count}"
        else:
            context = f"{func_name}#{call_count}"
        
        self.current_context_stack.append(context)
        self.call_count_stack.append(defaultdict(int))
        
        # 记录进入时的内存
        current_memory = self._get_current_memory()
        self.memory_stack.append(current_memory)
        
        # 初始化函数profile
        if context not in self.function_profiles:
            self.function_profiles[context] = FunctionMemoryProfile(
                function_name=func_name,
                context=context,
                call_count=call_count,
                current_memory_mb=current_memory,
                peak_memory_mb=current_memory,
                allocated_memory_mb=0,
                freed_memory_mb=0,
                child_memory_mb=0,
                start_time=time.time(),
                end_time=0
            )
        
        return context
    
    def exit_function(self, context: str):
        """
        退出函数时调用
        
        Args:
            context: 调用上下文
        """
        if not self.current_context_stack:
            return
        
        # 计算内存变化
        exit_memory = self._get_current_memory()
        enter_memory = self.memory_stack.pop() if self.memory_stack else exit_memory
        
        # 更新函数profile
        if context in self.function_profiles:
            profile = self.function_profiles[context]
            profile.end_time = time.time()
            profile.current_memory_mb = exit_memory
            
            # 计算内存分配/释放
            memory_delta = exit_memory - enter_memory
            if memory_delta > 0:
                profile.allocated_memory_mb += memory_delta
            else:
                profile.freed_memory_mb += abs(memory_delta)
            
            # 更新峰值
            profile.peak_memory_mb = max(profile.peak_memory_mb, exit_memory)
        
        # 更新全局峰值
        self.peak_memory_mb = max(self.peak_memory_mb, exit_memory)
        if self.track_gpu:
            gpu_memory = self._get_gpu_memory()
            self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, gpu_memory)
        
        # 弹出上下文
        self.current_context_stack.pop()
        self.call_count_stack.pop()
    
    def track_variable(self, var_name: str, value: Any, operation: str = "create"):
        """
        追踪变量
        
        Args:
            var_name: 变量名
            value: 变量值
            operation: 操作类型 (create/update/delete)
        """
        if value is None:
            return
        
        # 获取当前上下文
        context = self.current_context_stack[-1] if self.current_context_stack else "global"
        
        # 计算变量信息
        var_info = self._analyze_variable(var_name, value, context)
        
        if var_info:
            # 检查是否是版本更新
            if var_name in self.variable_registry and self.variable_registry[var_name]:
                last_info = self.variable_registry[var_name][-1]
                if last_info.shape != var_info.shape or last_info.dtype != var_info.dtype:
                    var_info.version = last_info.version + 1
            
            self.variable_registry[var_name].append(var_info)
    
    def _analyze_variable(self, name: str, value: Any, context: str) -> Optional[VariableInfo]:
        """分析变量信息"""
        try:
            if torch.is_tensor(value):
                shape = tuple(value.shape)
                dtype = str(value.dtype)
                device = str(value.device)
                size_mb = value.element_size() * value.numel() / (1024 * 1024)
                
                return VariableInfo(
                    name=name,
                    version=1,
                    shape=shape,
                    dtype=dtype,
                    size_mb=size_mb,
                    device=device,
                    created_at=time.time(),
                    context=context
                )
            elif isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
                # 处理张量列表
                total_size = sum(
                    v.element_size() * v.numel() / (1024 * 1024)
                    for v in value if torch.is_tensor(v)
                )
                return VariableInfo(
                    name=name,
                    version=1,
                    shape=(len(value),),
                    dtype="tensor_list",
                    size_mb=total_size,
                    device="mixed",
                    created_at=time.time(),
                    context=context
                )
        except Exception as e:
            logger.debug(f"无法分析变量 {name}: {e}")
        
        return None
    
    def _get_current_memory(self) -> float:
        """获取当前内存使用（MB）"""
        # 使用tracemalloc获取更精确的内存变化
        if self.tracemalloc_enabled:
            current, peak = tracemalloc.get_traced_memory()
            # 返回当前内存使用量（MB）
            return current / (1024 * 1024)
        else:
            # 回退到进程内存
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
    
    def _get_gpu_memory(self) -> float:
        """获取GPU内存使用（MB）"""
        if not self.track_gpu:
            return 0
        
        try:
            # 获取所有GPU的内存使用
            total_memory = 0
            for i in range(torch.cuda.device_count()):
                total_memory += torch.cuda.memory_allocated(i) / (1024 * 1024)
            return total_memory
        except:
            return 0
    
    def take_snapshot(self) -> MemorySnapshot:
        """获取内存快照"""
        # 收集当前分配的变量
        allocated_vars = {}
        for var_name, versions in self.variable_registry.items():
            if versions:
                latest = versions[-1]
                allocated_vars[f"{var_name}_v{latest.version}"] = latest.size_mb
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_memory_mb=self._get_current_memory(),
            gpu_memory_mb=self._get_gpu_memory(),
            allocated_variables=allocated_vars
        )
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def get_memory_report(self) -> Dict:
        """生成内存报告"""
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "total_variables_tracked": sum(len(v) for v in self.variable_registry.values()),
            "function_profiles": {
                ctx: {
                    "function": prof.function_name,
                    "call_count": prof.call_count,
                    "peak_memory_mb": prof.peak_memory_mb,
                    "allocated_mb": prof.allocated_memory_mb,
                    "freed_mb": prof.freed_memory_mb,
                    "duration_ms": (prof.end_time - prof.start_time) * 1000 if prof.end_time else 0
                }
                for ctx, prof in self.function_profiles.items()
            },
            "variable_summary": {
                var_name: {
                    "versions": len(versions),
                    "total_size_mb": sum(v.size_mb for v in versions),
                    "latest_shape": versions[-1].shape if versions else None
                }
                for var_name, versions in self.variable_registry.items()
                if versions
            }
        }
    
    def generate_flame_graph_data(self) -> Dict:
        """生成火焰图数据"""
        flame_data = {
            "name": "root",
            "value": self.peak_memory_mb,
            "children": []
        }
        
        # 构建层级结构
        context_tree = {}
        for context, profile in self.function_profiles.items():
            parts = context.split("->")
            current_level = context_tree
            
            for part in parts:
                if part not in current_level:
                    current_level[part] = {
                        "profile": None,
                        "children": {}
                    }
                current_level = current_level[part]["children"]
            
            # 设置profile
            parent = context_tree
            for part in parts[:-1]:
                parent = parent[part]["children"]
            if parts[-1] in parent:
                parent[parts[-1]]["profile"] = profile
        
        # 转换为火焰图格式
        def build_flame_node(name: str, node_data: Dict) -> Dict:
            profile = node_data.get("profile")
            children = []
            
            for child_name, child_data in node_data.get("children", {}).items():
                children.append(build_flame_node(child_name, child_data))
            
            return {
                "name": name,
                "value": profile.peak_memory_mb if profile else sum(c["value"] for c in children),
                "children": children
            }
        
        for root_name, root_data in context_tree.items():
            flame_data["children"].append(build_flame_node(root_name, root_data))
        
        return flame_data