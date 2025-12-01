#!/usr/bin/env python
"""
CodeDAG 集成测试运行器

运行所有测试样例并生成汇总报告
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from test_harness import CodeDAGTester, DependencyChecker, TestResult


def run_all_tests(verbose: bool = True, skip_slow: bool = False):
    """
    运行所有测试

    Args:
        verbose: 是否显示详细输出
        skip_slow: 是否跳过耗时测试
    """
    start_time = time.time()

    print("=" * 70)
    print("CodeDAG 集成测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 检查依赖
    features = DependencyChecker.get_available_features()
    print("\n依赖检查:")
    for name, available in features.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")

    # 创建结果目录
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    all_results = []

    # ========== 测试样例 1: 简单计算 ==========
    print("\n" + "=" * 70)
    print("测试样例 1: 简单计算 (NumPy)")
    print("=" * 70)

    try:
        from test_01_simple_calc import (
            test_simple_calculation,
            test_random_operations,
            test_unary_operations
        )

        r = test_simple_calculation()
        if r:
            all_results.append(r)

        r = test_random_operations()
        if r:
            all_results.append(r)

        if not skip_slow:
            r = test_unary_operations()
            if r:
                all_results.append(r)

    except Exception as e:
        print(f"测试样例 1 失败: {e}")
        all_results.append(TestResult(
            case_name="01_Simple_Calc",
            success=False,
            errors=[str(e)]
        ))

    # ========== 测试样例 2: OpenCV 图像处理 ==========
    print("\n" + "=" * 70)
    print("测试样例 2: OpenCV 图像处理")
    print("=" * 70)

    if features['opencv']:
        try:
            from test_02_opencv_pipeline import (
                test_preprocessing_pipeline,
                test_augmentation_pipeline,
                test_edge_detection_pipeline,
                test_single_image_operations
            )

            r = test_preprocessing_pipeline()
            if r:
                all_results.append(r)

            r = test_augmentation_pipeline()
            if r:
                all_results.append(r)

            if not skip_slow:
                r = test_edge_detection_pipeline()
                if r:
                    all_results.append(r)

            r = test_single_image_operations()
            if r:
                all_results.append(r)

        except Exception as e:
            print(f"测试样例 2 失败: {e}")
            all_results.append(TestResult(
                case_name="02_OpenCV",
                success=False,
                errors=[str(e)]
            ))
    else:
        print("跳过: OpenCV 不可用")

    # ========== 测试样例 3: PyTorch Dataset ==========
    print("\n" + "=" * 70)
    print("测试样例 3: PyTorch Dataset")
    print("=" * 70)

    if features['torch']:
        try:
            from test_03_pytorch_dataset import (
                test_mock_dataset,
                test_opencv_dataset,
                test_torch_dataloader,
                test_data_augmentation_dataset
            )

            r = test_mock_dataset()
            if r:
                all_results.append(r)

            if features['opencv']:
                r = test_opencv_dataset()
                if r:
                    all_results.append(r)

            r = test_torch_dataloader()
            if r:
                all_results.append(r)

            if not skip_slow:
                r = test_data_augmentation_dataset()
                if r:
                    all_results.append(r)

        except Exception as e:
            print(f"测试样例 3 失败: {e}")
            all_results.append(TestResult(
                case_name="03_PyTorch",
                success=False,
                errors=[str(e)]
            ))
    else:
        print("跳过: PyTorch 不可用")

    # ========== 测试样例 4: Generator ==========
    print("\n" + "=" * 70)
    print("测试样例 4: Generator Pipeline")
    print("=" * 70)

    try:
        from test_04_generator_pipeline import (
            test_pure_generator,
            test_opencv_generator,
            test_batch_generator,
            test_tensorflow_generator,
            test_complex_pipeline
        )

        r = test_pure_generator()
        if r:
            all_results.append(r)

        if features['opencv']:
            r = test_opencv_generator()
            if r:
                all_results.append(r)

        r = test_batch_generator()
        if r:
            all_results.append(r)

        if features['tensorflow'] and not skip_slow:
            r = test_tensorflow_generator()
            if r:
                all_results.append(r)

        r = test_complex_pipeline()
        if r:
            all_results.append(r)

    except Exception as e:
        print(f"测试样例 4 失败: {e}")
        all_results.append(TestResult(
            case_name="04_Generator",
            success=False,
            errors=[str(e)]
        ))

    # ========== 生成汇总报告 ==========
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("测试汇总报告")
    print("=" * 70)

    passed = sum(1 for r in all_results if r.success)
    failed = len(all_results) - passed

    print(f"\n总测试数: {len(all_results)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总耗时: {total_time:.2f}s")

    print("\n详细结果:")
    print("-" * 70)

    for r in all_results:
        status = "✓ PASS" if r.success else "✗ FAIL"
        print(f"{status}  {r.case_name}")
        print(f"       追踪: {r.trace_time:.3f}s, 节点={r.node_count}, 边={r.edge_count}")
        print(f"       优化: {r.optimize_time:.3f}s, 分区={r.partition_count}")
        print(f"       迁移: {r.migration_time:.3f}s, Patch={r.patched_count}")

        if r.errors:
            for err in r.errors[:2]:
                print(f"       错误: {err[:80]}...")

        if r.warnings and verbose:
            for w in r.warnings[:2]:
                print(f"       警告: {w[:80]}...")

    # 保存汇总到文件
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time': total_time,
        'features': features,
        'total_tests': len(all_results),
        'passed': passed,
        'failed': failed,
        'results': [r.to_dict() for r in all_results]
    }

    summary_path = results_dir / "test_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n汇总报告已保存: {summary_path}")

    # 返回是否全部通过
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="CodeDAG 集成测试运行器")
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细输出'
    )
    parser.add_argument(
        '--skip-slow',
        action='store_true',
        help='跳过耗时测试'
    )
    parser.add_argument(
        '--test',
        type=int,
        choices=[1, 2, 3, 4],
        help='只运行指定的测试样例'
    )

    args = parser.parse_args()

    if args.test:
        # 运行单个测试
        print(f"只运行测试样例 {args.test}")
        if args.test == 1:
            from test_01_simple_calc import (
                test_simple_calculation,
                test_random_operations,
                test_unary_operations
            )
            test_simple_calculation()
            test_random_operations()
            test_unary_operations()
        elif args.test == 2:
            from test_02_opencv_pipeline import (
                test_preprocessing_pipeline,
                test_single_image_operations
            )
            test_preprocessing_pipeline()
            test_single_image_operations()
        elif args.test == 3:
            from test_03_pytorch_dataset import (
                test_mock_dataset,
                test_torch_dataloader
            )
            test_mock_dataset()
            test_torch_dataloader()
        elif args.test == 4:
            from test_04_generator_pipeline import (
                test_pure_generator,
                test_complex_pipeline
            )
            test_pure_generator()
            test_complex_pipeline()
    else:
        # 运行所有测试
        success = run_all_tests(verbose=args.verbose, skip_slow=args.skip_slow)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
