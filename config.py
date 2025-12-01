"""
CodeDAG配置管理模块
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class TracingConfig:
    """追踪配置"""
    max_depth: int = 12
    track_memory: bool = True
    track_gpu: bool = True
    skip_functions: list = None
    
    def __post_init__(self):
        if self.skip_functions is None:
            self.skip_functions = [
                '<module>', '<listcomp>', '<genexpr>', '<lambda>',
                'format', 'items', 'keys', 'values', '__repr__', '__str__'
            ]

@dataclass
class OptimizationConfig:
    """优化配置"""
    mode: str = "balanced"  # "performance", "memory", "balanced"
    memory_threshold_mb: float = 50.0
    performance_threshold_ms: float = 5.0
    enable_migration: bool = True
    target_device: str = "cuda:0"

@dataclass
class VisualizationConfig:
    """可视化配置"""
    generate_dag: bool = True
    generate_flame_graph: bool = True
    output_format: str = "png"  # "png", "svg", "html"
    width: int = 12
    height: int = 8

@dataclass
class CodeDAGConfig:
    """CodeDAG全局配置"""
    tracing: TracingConfig = None
    optimization: OptimizationConfig = None
    visualization: VisualizationConfig = None
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # 输出配置
    output_dir: str = "results"
    save_intermediate: bool = False
    
    def __post_init__(self):
        if self.tracing is None:
            self.tracing = TracingConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "CodeDAGConfig":
        """从字典创建配置"""
        tracing_dict = config_dict.get("tracing", {})
        optimization_dict = config_dict.get("optimization", {})
        visualization_dict = config_dict.get("visualization", {})
        
        return cls(
            tracing=TracingConfig(**tracing_dict),
            optimization=OptimizationConfig(**optimization_dict),
            visualization=VisualizationConfig(**visualization_dict),
            log_level=config_dict.get("log_level", "INFO"),
            log_file=config_dict.get("log_file"),
            output_dir=config_dict.get("output_dir", "results"),
            save_intermediate=config_dict.get("save_intermediate", False)
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "tracing": {
                "max_depth": self.tracing.max_depth,
                "track_memory": self.tracing.track_memory,
                "track_gpu": self.tracing.track_gpu,
                "skip_functions": self.tracing.skip_functions
            },
            "optimization": {
                "mode": self.optimization.mode,
                "memory_threshold_mb": self.optimization.memory_threshold_mb,
                "performance_threshold_ms": self.optimization.performance_threshold_ms,
                "enable_migration": self.optimization.enable_migration,
                "target_device": self.optimization.target_device
            },
            "visualization": {
                "generate_dag": self.visualization.generate_dag,
                "generate_flame_graph": self.visualization.generate_flame_graph,
                "output_format": self.visualization.output_format,
                "width": self.visualization.width,
                "height": self.visualization.height
            },
            "log_level": self.log_level,
            "log_file": self.log_file,
            "output_dir": self.output_dir,
            "save_intermediate": self.save_intermediate
        }

# 默认配置实例
default_config = CodeDAGConfig()