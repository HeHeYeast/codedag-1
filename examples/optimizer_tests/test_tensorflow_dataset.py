"""
TensorFlow图像处理数据管道优化测试
测试数据管道：resize -> crop -> gauss噪声
batch_size=4
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import tensorflow as tf
import numpy as np

from core.enhanced_tracer import EnhancedTracer
from optimizer import OptimizerManager


def load_image(idx):
    """模拟从磁盘加载图像"""
    image = np.random.rand(128, 128, 3).astype(np.float32)
    return image, idx


def resize_image(image, label):
    """Resize操作"""
    resized = tf.image.resize(image, [96, 96])
    return resized, label


def random_crop_image(image, label):
    """随机裁剪"""
    cropped = tf.image.random_crop(image, size=[64, 64, 3])
    return cropped, label


def add_gaussian_noise(image, label):
    """添加高斯噪声"""
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01, dtype=tf.float32)
    noisy = image + noise
    noisy = tf.clip_by_value(noisy, 0.0, 1.0)
    return noisy, label


def create_dataset(size=16):
    """创建TensorFlow数据集"""
    # 创建索引数据集
    indices = np.arange(size)

    # 创建Dataset
    dataset = tf.data.Dataset.from_tensor_slices(indices)

    # 加载图像
    dataset = dataset.map(
        lambda idx: tf.py_function(
            func=load_image,
            inp=[idx],
            Tout=[tf.float32, tf.int64]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 数据增强管道
    dataset = dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(random_crop_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(add_gaussian_noise, num_parallel_calls=tf.data.AUTOTUNE)

    # 批处理和预取
    dataset = dataset.batch(4)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    output_dir = Path(__file__).parent / 'outputs' / 'tensorflow'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据集（在追踪之前）
    dataset = create_dataset(size=16)

    # 创建tracer
    tracer = EnhancedTracer(max_depth=8, track_memory=False)

    # 使用context manager追踪数据加载
    batch_data = []
    with tracer.tracing_context():
        for batch_images, batch_indices in dataset:
            batch_data.append(batch_images)
            break  # 只加载一个batch

    # 导出tracer数据流图
    import json
    dataflow = tracer.export_dataflow_graph()
    with open(output_dir / 'dataflow_graph.json', 'w') as f:
        json.dump(dataflow, f, indent=2)

    # 导出tracer可视化
    tracer.export_visualization(str(output_dir / 'dataflow_graph.dot'))

    # 执行优化并自动导出每个阶段
    dag = tracer.dag_builder.dag
    optimizer = OptimizerManager(
        k=2,
        coarsen_max_depth=5,
        max_iterations=50
    )

    # 一次调用，自动导出所有阶段
    result = optimizer.optimize(dag, export_dir=str(output_dir))

    print(f"\n✓ TensorFlow测试完成，结果保存至: {output_dir}")
    print(f"  - 数据流图: dataflow_graph.json, dataflow_graph.dot")
    print(f"  - 粗化结果: coarsening_result.json, coarsening_graph.dot")
    print(f"  - K路分割: kway_partition.json")
    print(f"  - 迭代优化: optimized_partition.json")


if __name__ == "__main__":
    main()
