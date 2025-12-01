"""
CodeDAG 通用测试运行器 (Test Harness)
封装 API 文档中提到的核心流程，用于对任意函数进行全流程验证。
"""

import sys
import os
import time
import json
import logging
import shutil
import subprocess
from typing import Callable, Any, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

# 添加项目根目录到路径 (使用 resolve() 获取绝对路径)
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 引入 CodeDAG 模块
from core.enhanced_tracer import EnhancedTracer
from optimizer.optimizer_manager import OptimizerManager
from migration.api import PipelineMigrator

# Graphviz 可视化模块延迟导入，避免未安装时启动失败
# 在运行时通过 DependencyChecker 检查后再导入
_graphviz_visualizer = None

def _get_dataflow_visualizer():
    """延迟加载 graphviz 可视化模块"""
    global _graphviz_visualizer
    if _graphviz_visualizer is None:
        try:
            from visualization.graphviz_visualizer import create_dataflow_visualization
            _graphviz_visualizer = create_dataflow_visualization
        except ImportError:
            _graphviz_visualizer = None
    return _graphviz_visualizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CodeDAG_Test")


@dataclass
class TestResult:
    """测试结果数据类"""
    case_name: str
    success: bool

    # 追踪阶段
    trace_time: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    function_nodes: int = 0
    variable_nodes: int = 0

    # 优化阶段
    optimize_time: float = 0.0
    coarsening_stats: Dict = field(default_factory=dict)
    partition_stats: Dict = field(default_factory=dict)
    optimization_stats: Dict = field(default_factory=dict)
    partition_count: int = 0

    # 迁移阶段
    migration_time: float = 0.0
    migration_stats: Dict = field(default_factory=dict)
    patched_count: int = 0

    # 错误信息
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class DependencyChecker:
    """依赖检查器"""

    @staticmethod
    def check_torch() -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False

    @staticmethod
    def check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def check_kornia() -> bool:
        try:
            import kornia
            return True
        except ImportError:
            return False

    @staticmethod
    def check_opencv() -> bool:
        try:
            import cv2
            return True
        except ImportError:
            return False

    @staticmethod
    def check_tensorflow() -> bool:
        try:
            import tensorflow
            return True
        except Exception:
            # 捕获所有异常，包括 ImportError 和 protobuf 版本冲突等
            return False

    @staticmethod
    def check_graphviz() -> bool:
        """检查 graphviz Python 库和系统二进制文件"""
        try:
            import graphviz
            # 还需要检查系统是否安装了 dot 命令
            result = subprocess.run(
                ['dot', '-V'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception:
            return False

    @classmethod
    def get_available_features(cls) -> Dict[str, bool]:
        return {
            'torch': cls.check_torch(),
            'cuda': cls.check_cuda(),
            'kornia': cls.check_kornia(),
            'opencv': cls.check_opencv(),
            'tensorflow': cls.check_tensorflow(),
            'graphviz': cls.check_graphviz(),
        }


class CodeDAGTester:
    """CodeDAG 测试运行器"""

    def __init__(self, output_dir: str = "results", clear_existing: bool = True):
        """
        初始化测试器

        Args:
            output_dir: 输出目录
            clear_existing: 是否清除已存在的输出目录
        """
        self.output_dir = Path(__file__).parent / output_dir

        if clear_existing and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 检查依赖
        self.features = DependencyChecker.get_available_features()
        logger.info(f"可用特性: {self.features}")

        # 存储所有测试结果
        self.results: List[TestResult] = []

    def run_test(
        self,
        case_name: str,
        target_func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        max_depth: int = 10,
        track_memory: bool = True,
        k_partitions: int = 2,
        skip_migration: bool = False
    ) -> TestResult:
        """
        运行完整测试流程

        Args:
            case_name: 测试用例名称
            target_func: 目标函数
            args: 位置参数
            kwargs: 关键字参数
            max_depth: 追踪最大深度
            track_memory: 是否追踪内存
            k_partitions: 分区数量
            skip_migration: 是否跳过迁移阶段

        Returns:
            TestResult 对象
        """
        if kwargs is None:
            kwargs = {}

        result = TestResult(case_name=case_name, success=False)

        logger.info(f"\n{'='*60}")
        logger.info(f"开始测试: {case_name}")
        logger.info(f"{'='*60}")

        # 创建用例目录
        case_dir = self.output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        try:
            # -------------------------------------------------
            # 阶段 1: 代码追踪与数据流图构建
            # -------------------------------------------------
            self._run_trace_phase(
                result, case_dir, target_func, args, kwargs,
                max_depth, track_memory
            )

            # -------------------------------------------------
            # 阶段 2: 图优化
            # -------------------------------------------------
            dag, partitions = self._run_optimize_phase(
                result, case_dir, k_partitions
            )

            # -------------------------------------------------
            # 阶段 3: 迁移执行
            # -------------------------------------------------
            if not skip_migration:
                self._run_migration_phase(
                    result, case_dir, target_func, args, kwargs,
                    dag, partitions
                )
            else:
                result.warnings.append("跳过迁移阶段")
                logger.warning("跳过迁移阶段")

            result.success = True
            logger.info(f"测试 {case_name} 完成: 成功")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"测试 {case_name} 失败: {e}", exc_info=True)

        # 保存结果
        self._save_result(result, case_dir)
        self.results.append(result)

        return result

    def _run_trace_phase(
        self,
        result: TestResult,
        case_dir: Path,
        target_func: Callable,
        args: tuple,
        kwargs: dict,
        max_depth: int,
        track_memory: bool
    ):
        """运行追踪阶段"""
        logger.info("[阶段 1] 代码追踪与数据流图构建...")

        # 创建追踪器
        self.tracer = EnhancedTracer(max_depth=max_depth, track_memory=track_memory)

        # 执行追踪
        start_time = time.time()
        with self.tracer.tracing_context():
            self.original_result = target_func(*args, **kwargs)
        result.trace_time = time.time() - start_time

        logger.info(f"  原始执行耗时: {result.trace_time:.4f}s")

        # 获取 DAG
        dag = self.tracer.dag_builder.dag
        result.node_count = len(dag.nodes)
        result.edge_count = len(dag.edges)

        # 统计节点类型
        result.function_nodes = sum(
            1 for n in dag.nodes.values() if n.node_type == 'function_call'
        )
        result.variable_nodes = sum(
            1 for n in dag.nodes.values() if n.node_type == 'variable'
        )

        logger.info(f"  DAG 构建完成: 节点数={result.node_count}, 边数={result.edge_count}")
        logger.info(f"  函数节点: {result.function_nodes}, 变量节点: {result.variable_nodes}")

        # 验证追踪结果
        if result.node_count == 0:
            raise AssertionError("DAG 节点数为 0，追踪失败")

        # 检查性能数据
        func_nodes = [n for n in dag.nodes.values() if n.node_type == 'function_call']
        if func_nodes:
            sample_node = func_nodes[0]
            sample_perf = sample_node.performance
            logger.info(f"  节点性能样本 ({sample_node.name}): {sample_perf}")

            if 'execution_time' not in sample_perf:
                result.warnings.append(f"节点 {sample_node.name} 缺少执行时间数据")

        # 导出数据流图 JSON
        try:
            dataflow = self.tracer.export_dataflow_graph()
            with open(case_dir / "dataflow.json", 'w', encoding='utf-8') as f:
                json.dump(dataflow, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"  数据流图已保存: {case_dir / 'dataflow.json'}")
        except Exception as e:
            result.warnings.append(f"导出数据流图 JSON 失败: {e}")

        # 生成可视化
        if self.features['graphviz']:
            visualizer = _get_dataflow_visualizer()
            if visualizer is not None:
                try:
                    success = visualizer(
                        self.tracer,
                        str(case_dir / "dataflow.svg")
                    )
                    if success:
                        logger.info(f"  可视化已生成: {case_dir / 'dataflow.svg'}")
                    else:
                        result.warnings.append("可视化生成失败")
                except Exception as e:
                    result.warnings.append(f"可视化生成异常: {e}")
            else:
                result.warnings.append("可视化模块导入失败")
        else:
            result.warnings.append("Graphviz 不可用，跳过可视化")

    def _run_optimize_phase(
        self,
        result: TestResult,
        case_dir: Path,
        k_partitions: int
    ):
        """运行优化阶段"""
        logger.info("[阶段 2] 图优化...")

        dag = self.tracer.dag_builder.dag

        # 动态调整分区数
        actual_k = min(k_partitions, max(1, result.node_count // 2))
        if actual_k != k_partitions:
            logger.info(f"  调整分区数: {k_partitions} -> {actual_k} (节点数不足)")

        # 创建优化器
        optimizer = OptimizerManager(
            k=actual_k,
            coarsen_max_depth=5,
            max_iterations=50
        )

        # 执行优化
        start_time = time.time()
        opt_result = optimizer.optimize(dag, export_dir=str(case_dir))
        result.optimize_time = time.time() - start_time

        logger.info(f"  优化耗时: {result.optimize_time:.4f}s")

        # 记录统计信息 (统计信息嵌套在 'statistics' 字典中)
        stats = opt_result.get('statistics', {})
        result.coarsening_stats = stats.get('coarsening', {})
        result.partition_stats = stats.get('partitioning', {})
        result.optimization_stats = stats.get('iteration', {})

        partitions = opt_result['optimized_partitions']
        result.partition_count = len(partitions)

        logger.info(f"  粗化统计: {result.coarsening_stats}")
        logger.info(f"  分区统计: {result.partition_stats}")
        logger.info(f"  优化统计: {result.optimization_stats}")

        # 验证分区结果
        if len(partitions) == 0:
            raise AssertionError("分区结果为空")

        # 保存优化器引用以便后续使用
        self.optimizer = optimizer
        self.opt_result = opt_result

        return opt_result['coarsened_dag'], partitions

    def _run_migration_phase(
        self,
        result: TestResult,
        case_dir: Path,
        target_func: Callable,
        args: tuple,
        kwargs: dict,
        dag,
        partitions
    ):
        """
        运行迁移阶段

        注意: 使用 try...finally 确保即使执行失败也能恢复环境，
        避免 Monkey Patch 污染后续测试用例。
        """
        logger.info("[阶段 3] 迁移执行...")

        # 确定目标设备
        if self.features['cuda']:
            device = "cuda:0"
            logger.info("  使用 CUDA 设备")
        else:
            device = "cpu"
            logger.warning("  CUDA 不可用，使用 CPU 模拟迁移流程")

        # 生成迁移计划
        migration_plan = self.optimizer.get_migration_plan(
            partitions=partitions,
            dag=dag,
            gpu_count=1
        )

        # 保存迁移计划
        with open(case_dir / "migration_plan.json", 'w', encoding='utf-8') as f:
            json.dump(migration_plan, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"  迁移计划已保存: {case_dir / 'migration_plan.json'}")

        # 创建迁移器
        migrator = PipelineMigrator.from_plan(migration_plan, default_device=device)

        start_time = time.time()

        # 使用 try...finally 确保清理
        try:
            # 1. 应用 Patch
            apply_stats = migrator.apply()
            result.patched_count = apply_stats.get('patched', 0)
            logger.info(f"  Patch 统计: {apply_stats}")

            # 验证迁移已应用
            if not migrator.is_applied():
                raise RuntimeError("迁移应用失败")

            # 2. 执行迁移后的函数
            logger.info("  执行迁移后的函数...")
            try:
                migrated_result = target_func(*args, **kwargs)
                logger.info("  迁移后执行成功")

                # 可选：检查结果是否在 GPU 上
                if hasattr(migrated_result, 'device'):
                    logger.info(f"  结果设备: {migrated_result.device}")

            except Exception as e:
                # 迁移后执行失败意味着 CodeDAG 的 Fallback 机制也失败了
                # 或者生成的 Patch 有严重错误，应视为测试失败
                error_msg = f"迁移后函数执行异常: {str(e)}"
                result.errors.append(error_msg)
                logger.error(f"  {error_msg}")
                # 重新抛出异常，让外层 try...except 捕获并标记测试失败
                raise

        finally:
            # 3. 必须确保恢复环境！
            if migrator.is_applied():
                restore_count = migrator.restore()
                logger.info(f"  恢复了 {restore_count} 个函数")
            else:
                logger.warning("  Migrator 状态异常或已恢复")

        result.migration_time = time.time() - start_time
        result.migration_stats = migrator.get_stats()

        # 验证已恢复
        if migrator.is_applied():
            raise RuntimeError("迁移器未能正确恢复原始状态")

        logger.info(f"  迁移阶段耗时: {result.migration_time:.4f}s")

    def _save_result(self, result: TestResult, case_dir: Path):
        """保存测试结果"""
        result_path = case_dir / "test_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"  测试结果已保存: {result_path}")

    def generate_summary(self) -> Dict:
        """生成测试汇总报告"""
        summary = {
            'total_tests': len(self.results),
            'passed': sum(1 for r in self.results if r.success),
            'failed': sum(1 for r in self.results if not r.success),
            'features': self.features,
            'results': [r.to_dict() for r in self.results]
        }

        # 保存汇总
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        # 打印汇总
        logger.info(f"\n{'='*60}")
        logger.info("测试汇总")
        logger.info(f"{'='*60}")
        logger.info(f"总测试数: {summary['total_tests']}")
        logger.info(f"通过: {summary['passed']}")
        logger.info(f"失败: {summary['failed']}")

        for r in self.results:
            status = "✓" if r.success else "✗"
            logger.info(f"  {status} {r.case_name}")
            if r.errors:
                for err in r.errors:
                    logger.info(f"      错误: {err}")

        return summary


if __name__ == "__main__":
    # 简单测试
    def simple_test():
        import numpy as np
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        return np.dot(a, b)

    tester = CodeDAGTester()
    result = tester.run_test("simple_test", simple_test)
    tester.generate_summary()
