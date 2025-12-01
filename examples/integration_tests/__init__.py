"""
CodeDAG 集成测试模块

包含完整的测试用例，验证：
- 代码追踪 (EnhancedTracer)
- 图优化 (OptimizerManager)
- 代码迁移 (PipelineMigrator)
- 可视化 (GraphvizDataflowVisualizer)

测试样例:
1. 简单计算 - NumPy 矩阵运算
2. 复杂计算 - OpenCV 图像处理管道
3. PyTorch Dataset - 深度学习数据加载
4. Generator Pipeline - Python 生成器 / TensorFlow
"""

from .test_harness import CodeDAGTester, DependencyChecker, TestResult

__all__ = ['CodeDAGTester', 'DependencyChecker', 'TestResult']
