"""
PyTorch图像处理数据管道优化测试
测试数据管道：resize -> crop -> gauss噪声
batch_size=4
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from core.enhanced_tracer import EnhancedTracer
from optimizer import OptimizerManager


class ImageDataset(Dataset):
    """模拟图像数据集"""

    def __init__(self, size=16, image_size=(128, 128)):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 模拟加载图像
        image = self.load_image(idx)

        # 数据增强管道
        image = self.resize(image)
        image = self.random_crop(image)
        image = self.add_gaussian_noise(image)

        # 转换为tensor
        image = self.to_tensor(image)

        return image, idx

    def load_image(self, idx):
        """模拟从磁盘加载图像"""
        image = np.random.rand(*self.image_size, 3).astype(np.float32)
        return image

    def resize(self, image):
        """Resize操作"""
        target_size = (96, 96)
        resized = np.random.rand(*target_size, 3).astype(np.float32)
        return resized

    def random_crop(self, image):
        """随机裁剪"""
        crop_size = (64, 64)
        h, w = image.shape[:2]
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])
        cropped = image[top:top+crop_size[0], left:left+crop_size[1]]
        return cropped

    def add_gaussian_noise(self, image):
        """添加高斯噪声"""
        noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
        noisy = image + noise
        noisy = np.clip(noisy, 0, 1)
        return noisy

    def to_tensor(self, image):
        """转换为PyTorch tensor"""
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor


def main():
    output_dir = Path(__file__).parent / 'outputs' / 'pytorch'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据集和DataLoader（在追踪之前）
    dataset = ImageDataset(size=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # 创建tracer
    tracer = EnhancedTracer(max_depth=8, track_memory=False)

    # 使用context manager追踪数据加载
    batch_data = []
    with tracer.tracing_context():
        for batch_images, batch_indices in dataloader:
            batch_data.append(batch_images)
            break  # 只加载一个batch

    # 导出tracer数据流图
    import json
    dataflow = tracer.export_dataflow_graph()
    with open(output_dir / 'dataflow_graph.json', 'w') as f:
        json.dump(dataflow, f, indent=2, ensure_ascii=False)

    # 导出tracer可视化（SVG格式）
    svg_path = str(output_dir / 'dataflow_graph.svg')
    success = tracer.export_visualization(svg_path)
    if success:
        print(f"✓ 数据流图可视化已生成: {svg_path}")
    else:
        print("✗ 数据流图可视化生成失败")

    # 执行优化并自动导出每个阶段
    dag = tracer.dag_builder.dag
    optimizer = OptimizerManager(
        k=2,
        coarsen_max_depth=5,
        max_iterations=50
    )

    # 一次调用，自动导出所有阶段
    result = optimizer.optimize(dag, export_dir=str(output_dir))

    print(f"\n✓ PyTorch测试完成，结果保存至: {output_dir}")
    print(f"  - 数据流图: dataflow_graph.json, dataflow_graph.dot")
    print(f"  - 粗化结果: coarsening_result.json, coarsening_graph.dot")
    print(f"  - K路分割: kway_partition.json")
    print(f"  - 迭代优化: optimized_partition.json")


if __name__ == "__main__":
    main()
