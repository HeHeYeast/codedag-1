"""
CodeDAG异常定义模块
"""

class CodeDAGError(Exception):
    """CodeDAG基础异常类"""
    pass

class TracingError(CodeDAGError):
    """追踪相关错误"""
    pass

class OptimizationError(CodeDAGError):
    """优化相关错误"""
    pass

class MigrationError(CodeDAGError):
    """迁移相关错误"""
    pass

class VisualizationError(CodeDAGError):
    """可视化相关错误"""
    pass

class ConfigurationError(CodeDAGError):
    """配置相关错误"""
    pass