"""
测试样例 1: 简单计算 (NumPy 矩阵运算)

侧重点:
- 基础功能验证
- 测试 numpy 到 torch 的自动转换策略
- 验证 np.dot, np.add 等矩阵运算的追踪和迁移
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from test_harness import CodeDAGTester


def simple_matrix_computation(matrix_a, matrix_b):
    """
    简单的线性代数计算流程

    CodeDAG 应该能识别 numpy.dot, numpy.add 等操作，
    并在迁移时将它们转换为 torch.matmul, torch.add 等。
    """
    # 矩阵乘法 (np.dot -> torch.matmul)
    c = np.dot(matrix_a, matrix_b)

    # 简单的加法 (运算符，可能不被追踪为函数)
    d = c + 1.0

    # 显式使用 np.add (np.add -> torch.add)
    e = np.add(d, matrix_a)

    # 归一化 (np.max -> torch.max)
    max_val = np.max(e)
    f = e / max_val

    # 统计运算 (np.mean, np.sum -> torch.mean, torch.sum)
    mean_val = np.mean(f)
    sum_val = np.sum(f)

    # 形状操作 (np.reshape -> torch.reshape)
    g = np.reshape(f, (-1,))

    return {
        'matmul_result': c,
        'normalized': f,
        'mean': mean_val,
        'sum': sum_val,
        'flattened_shape': g.shape
    }


def test_simple_calculation():
    """运行简单计算测试"""
    # 准备测试数据
    np.random.seed(42)
    matrix_a = np.random.rand(100, 100).astype(np.float32)
    matrix_b = np.random.rand(100, 100).astype(np.float32)

    # 创建测试器
    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    # 运行测试
    result = tester.run_test(
        case_name="01_Simple_Calc",
        target_func=simple_matrix_computation,
        args=(matrix_a, matrix_b),
        max_depth=10,
        track_memory=True,
        k_partitions=2
    )

    return result


def test_random_operations():
    """测试随机数操作"""

    def random_pipeline():
        """随机数生成管道"""
        # np.random.rand
        a = np.random.rand(50, 50)

        # np.random.randn
        b = np.random.randn(50, 50)

        # np.random.normal (使用 Dispatcher)
        c = np.random.normal(0, 1, (50, 50))

        # np.random.uniform (使用 Dispatcher)
        d = np.random.uniform(-1, 1, (50, 50))

        # 组合
        result = a + b + c + d
        return np.mean(result)

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="01b_Random_Ops",
        target_func=random_pipeline,
        max_depth=10,
        k_partitions=2
    )

    return result


def test_unary_operations():
    """测试一元运算"""

    def unary_pipeline(data):
        """一元运算管道"""
        # 数学运算
        a = np.abs(data)
        b = np.sqrt(a)
        c = np.exp(b * 0.01)  # 缩小以避免溢出
        d = np.log(c + 1)

        # 三角函数
        e = np.sin(d)
        f = np.cos(d)

        # 取整
        g = np.floor(e * 100) / 100
        h = np.ceil(f * 100) / 100

        # clip
        result = np.clip(g + h, -1, 1)

        return result

    # 准备数据
    np.random.seed(42)
    data = np.random.rand(50, 50).astype(np.float32) - 0.5  # [-0.5, 0.5]

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="01c_Unary_Ops",
        target_func=unary_pipeline,
        args=(data,),
        max_depth=10,
        k_partitions=2
    )

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("测试样例 1: 简单计算 (NumPy 矩阵运算)")
    print("=" * 60)

    # 运行所有子测试
    results = []

    print("\n[1/3] 矩阵运算测试...")
    results.append(test_simple_calculation())

    print("\n[2/3] 随机数操作测试...")
    results.append(test_random_operations())

    print("\n[3/3] 一元运算测试...")
    results.append(test_unary_operations())

    # 汇总
    print("\n" + "=" * 60)
    print("测试样例 1 汇总")
    print("=" * 60)
    passed = sum(1 for r in results if r.success)
    print(f"通过: {passed}/{len(results)}")
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] {r.case_name}")
