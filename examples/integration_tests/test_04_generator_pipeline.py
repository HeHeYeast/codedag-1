"""
测试样例 4: TensorFlow Dataset / Python Generator

侧重点:
- 验证对通用 Python 生成器的支持
- 测试 Generator 内部的 NumPy/OpenCV 操作
- 可选: 测试 TensorFlow Dataset.from_generator 兼容性

注意事项:
- TensorFlow 的 Graph 模式无法被 sys.settrace 追踪
- tf.data.Dataset 可能在不同线程执行 generator
- 我们重点测试纯 Python generator 中的操作
- TensorFlow 用例标记为"实验性"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from test_harness import CodeDAGTester, DependencyChecker


def check_dependencies():
    """检查依赖"""
    features = DependencyChecker.get_available_features()
    return features


def pure_python_generator():
    """
    纯 Python 生成器

    模拟典型的数据生成管道，包含 NumPy 操作
    这是 CodeDAG 主要支持的场景
    """
    for i in range(10):
        # 生成随机数据
        np.random.seed(i)
        data = np.random.rand(64, 64).astype(np.float32)

        # NumPy 预处理
        data = np.clip(data, 0.1, 0.9)
        data = (data - data.mean()) / (data.std() + 1e-8)

        # 添加噪声
        noise = np.random.normal(0, 0.01, data.shape).astype(np.float32)
        data = data + noise

        yield data


def opencv_data_generator():
    """
    包含 OpenCV 操作的生成器
    """
    import cv2

    for i in range(10):
        # 生成随机图像
        np.random.seed(i)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # OpenCV 处理
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 归一化
        img = img.astype(np.float32) / 255.0

        yield img


def batch_generator(base_generator, batch_size: int = 4):
    """
    批量生成器

    将基础生成器的输出组合成批次
    """
    batch = []
    for item in base_generator():
        batch.append(item)
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []

    # 处理剩余项
    if batch:
        yield np.stack(batch)


def test_pure_generator():
    """测试纯 Python 生成器"""

    def run_generator():
        """消费生成器"""
        results = list(pure_python_generator())
        return {
            'count': len(results),
            'shape': results[0].shape if results else None,
            'mean': np.mean([r.mean() for r in results])
        }

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="04a_PureGenerator",
        target_func=run_generator,
        max_depth=15,
        track_memory=True,
        k_partitions=2
    )

    return result


def test_opencv_generator():
    """测试 OpenCV 生成器"""
    features = check_dependencies()
    if not features['opencv']:
        print("跳过 OpenCV Generator 测试: OpenCV 不可用")
        return None

    def run_opencv_generator():
        results = list(opencv_data_generator())
        return {
            'count': len(results),
            'shape': results[0].shape if results else None,
        }

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="04b_OpenCVGenerator",
        target_func=run_opencv_generator,
        max_depth=15,
        k_partitions=2
    )

    return result


def test_batch_generator():
    """测试批量生成器"""

    def run_batch_generator():
        batches = list(batch_generator(pure_python_generator, batch_size=4))
        return {
            'num_batches': len(batches),
            'batch_shapes': [b.shape for b in batches]
        }

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="04c_BatchGenerator",
        target_func=run_batch_generator,
        max_depth=15,
        k_partitions=2
    )

    return result


def test_tensorflow_generator():
    """
    测试 TensorFlow Dataset Generator (实验性)

    注意:
    - TensorFlow 可能在后台线程运行 generator
    - 如果追踪失败，会自动降级到直接追踪 generator 函数
    """
    features = check_dependencies()
    if not features['tensorflow']:
        print("跳过 TensorFlow Generator 测试: TensorFlow 不可用")
        return None

    import tensorflow as tf

    def data_generator():
        """TensorFlow 兼容的生成器"""
        for i in range(10):
            np.random.seed(i)
            data = np.random.rand(50, 50).astype(np.float32)

            # NumPy 处理
            data = np.clip(data, 0, 1)
            data = data - data.mean()

            yield data

    def run_tf_pipeline():
        """运行 TensorFlow 数据管道"""
        try:
            # 创建 TF Dataset
            ds = tf.data.Dataset.from_generator(
                data_generator,
                output_signature=tf.TensorSpec(shape=(50, 50), dtype=tf.float32)
            )
            ds = ds.batch(2)

            results = []
            for batch in ds.take(3):  # 只取3个 batch
                results.append(batch.numpy())

            return {
                'success': True,
                'num_batches': len(results),
                'batch_shape': results[0].shape if results else None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    # 首先尝试追踪 TF 管道
    print("  尝试追踪 TensorFlow 管道...")
    result = tester.run_test(
        case_name="04d_TensorFlowGenerator",
        target_func=run_tf_pipeline,
        max_depth=20,
        k_partitions=2
    )

    # 如果节点数很少，说明 TF 内部执行可能没被追踪到
    # 降级到直接追踪 generator
    if result.node_count < 5:
        print("  TF 管道追踪结果有限，尝试直接追踪 generator...")

        def run_generator_directly():
            return list(data_generator())

        fallback_result = tester.run_test(
            case_name="04d_TensorFlowGenerator_Fallback",
            target_func=run_generator_directly,
            max_depth=15,
            k_partitions=2
        )
        return fallback_result

    return result


def test_complex_pipeline():
    """测试复杂的生成器管道"""

    def complex_pipeline():
        """
        复杂管道：多阶段处理

        Stage 1: 生成原始数据
        Stage 2: 特征提取
        Stage 3: 数据增强
        Stage 4: 批量组合
        """
        results = []

        for i in range(8):
            # Stage 1: 生成
            np.random.seed(i)
            raw = np.random.rand(32, 32, 3).astype(np.float32)

            # Stage 2: 特征提取
            # 模拟卷积操作（简单均值池化）
            pooled = raw.reshape(16, 2, 16, 2, 3).mean(axis=(1, 3))

            # Stage 3: 增强
            # 随机翻转
            if np.random.random() > 0.5:
                pooled = np.flip(pooled, axis=0)

            # 颜色抖动
            jitter = np.random.uniform(0.9, 1.1, (1, 1, 3))
            pooled = np.clip(pooled * jitter, 0, 1)

            # 添加噪声
            noise = np.random.normal(0, 0.02, pooled.shape)
            pooled = np.clip(pooled + noise, 0, 1)

            # Stage 4: 格式转换
            tensor = pooled.transpose(2, 0, 1)  # HWC -> CHW

            results.append(tensor)

        # 批量组合
        batch = np.stack(results)

        # 统计信息
        return {
            'batch_shape': batch.shape,
            'mean': float(batch.mean()),
            'std': float(batch.std()),
            'min': float(batch.min()),
            'max': float(batch.max())
        }

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="04e_ComplexPipeline",
        target_func=complex_pipeline,
        max_depth=20,
        k_partitions=2
    )

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("测试样例 4: TensorFlow/Generator 数据管道")
    print("=" * 60)

    features = check_dependencies()
    print(f"可用特性: {features}")

    # 运行所有子测试
    results = []

    print("\n[1/5] 纯 Python Generator 测试...")
    r = test_pure_generator()
    if r:
        results.append(r)

    print("\n[2/5] OpenCV Generator 测试...")
    r = test_opencv_generator()
    if r:
        results.append(r)

    print("\n[3/5] 批量 Generator 测试...")
    r = test_batch_generator()
    if r:
        results.append(r)

    print("\n[4/5] TensorFlow Generator 测试 (实验性)...")
    r = test_tensorflow_generator()
    if r:
        results.append(r)

    print("\n[5/5] 复杂管道测试...")
    r = test_complex_pipeline()
    if r:
        results.append(r)

    # 汇总
    print("\n" + "=" * 60)
    print("测试样例 4 汇总")
    print("=" * 60)
    if results:
        passed = sum(1 for r in results if r.success)
        print(f"通过: {passed}/{len(results)}")
        for r in results:
            status = "PASS" if r.success else "FAIL"
            print(f"  [{status}] {r.case_name}")
            print(f"      节点: {r.node_count}, 边: {r.edge_count}, 分区: {r.partition_count}")
    else:
        print("没有测试被执行")
