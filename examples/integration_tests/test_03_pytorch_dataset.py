"""
测试样例 3: PyTorch Dataset (深度学习数据加载)

侧重点:
- 测试类方法的追踪 (__getitem__)
- 测试迭代器（Iterator）模式下的图构建
- 验证 DataLoader 与追踪器的兼容性
- 测试 CPU 预处理操作的迁移

注意事项:
- num_workers 必须为 0，否则数据加载在子进程中，无法被 sys.settrace 追踪
- 追踪的是 __getitem__ 内部的逻辑，而非整个 DataLoader 机制
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from test_harness import CodeDAGTester, DependencyChecker


def check_dependencies():
    """检查必要的依赖"""
    features = DependencyChecker.get_available_features()

    if not features['torch']:
        print("警告: PyTorch 不可用，跳过此测试")
        return False

    if not features['opencv']:
        print("警告: OpenCV 不可用，将使用纯 NumPy 操作")

    return True


class MockImageDataset:
    """
    模拟图像数据集

    使用纯 Python 实现 Dataset 接口，不继承 torch.utils.data.Dataset
    这样追踪器可以更容易地追踪 __getitem__ 调用
    """

    def __init__(self, length: int = 10, image_size: tuple = (256, 256)):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 模拟从磁盘加载图像
        img = self._load_image(idx)

        # CPU 预处理操作 (目标是迁移这些到 GPU)
        img = self._resize(img)
        img = self._normalize(img)
        img = self._to_tensor(img)

        return img, idx

    def _load_image(self, idx):
        """模拟加载图像"""
        # 创建具有可预测内容的图像
        np.random.seed(idx)
        img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        return img

    def _resize(self, img):
        """Resize 操作"""
        # 使用简单的 NumPy 实现
        target_size = (128, 128)
        # 简化: 直接裁剪中心区域
        h, w = img.shape[:2]
        th, tw = target_size
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return img[start_h:start_h+th, start_w:start_w+tw]

    def _normalize(self, img):
        """归一化"""
        img = img.astype(np.float32) / 255.0
        # 标准化 (ImageNet 均值和标准差)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img

    def _to_tensor(self, img):
        """转换为 tensor 格式 (HWC -> CHW)"""
        return img.transpose(2, 0, 1)


class OpenCVImageDataset:
    """使用 OpenCV 的图像数据集"""

    def __init__(self, length: int = 10, image_size: tuple = (256, 256)):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import cv2

        # 模拟加载
        img = self._load_image(idx)

        # OpenCV 预处理
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 归一化和转换
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        return img, idx

    def _load_image(self, idx):
        np.random.seed(idx)
        return np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)


def simple_dataloader(dataset, batch_size: int = 4):
    """
    简单的 DataLoader 实现

    不使用 torch.utils.data.DataLoader，以便追踪器可以追踪所有操作
    """
    batches = []
    batch_data = []
    batch_indices = []

    for i in range(len(dataset)):
        data, idx = dataset[i]
        batch_data.append(data)
        batch_indices.append(idx)

        if len(batch_data) == batch_size:
            batches.append((np.stack(batch_data), batch_indices.copy()))
            batch_data = []
            batch_indices = []

    # 处理最后一个不完整的 batch
    if batch_data:
        batches.append((np.stack(batch_data), batch_indices.copy()))

    return batches


def test_mock_dataset():
    """测试 Mock 图像数据集"""
    if not check_dependencies():
        return None

    def run_dataset():
        dataset = MockImageDataset(length=8, image_size=(256, 256))
        batches = simple_dataloader(dataset, batch_size=2)
        return {
            'num_batches': len(batches),
            'batch_shape': batches[0][0].shape if batches else None,
            'total_samples': sum(len(b[1]) for b in batches)
        }

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="03a_MockDataset",
        target_func=run_dataset,
        max_depth=15,
        track_memory=True,
        k_partitions=2
    )

    return result


def test_opencv_dataset():
    """测试 OpenCV 图像数据集"""
    features = DependencyChecker.get_available_features()
    if not features['torch'] or not features['opencv']:
        print("跳过 OpenCV Dataset 测试: 依赖不满足")
        return None

    def run_opencv_dataset():
        dataset = OpenCVImageDataset(length=8, image_size=(256, 256))
        batches = simple_dataloader(dataset, batch_size=2)
        return {
            'num_batches': len(batches),
            'batch_shape': batches[0][0].shape if batches else None,
        }

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="03b_OpenCVDataset",
        target_func=run_opencv_dataset,
        max_depth=15,
        k_partitions=2
    )

    return result


def test_torch_dataloader():
    """测试 PyTorch DataLoader (num_workers=0)"""
    features = DependencyChecker.get_available_features()
    if not features['torch']:
        print("跳过 PyTorch DataLoader 测试: PyTorch 不可用")
        return None

    import torch
    from torch.utils.data import Dataset, DataLoader

    class TorchImageDataset(Dataset):
        """继承自 torch.utils.data.Dataset"""

        def __init__(self, length=10):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # 生成随机图像
            np.random.seed(idx)
            img = np.random.rand(3, 64, 64).astype(np.float32)

            # NumPy 操作 (可被迁移)
            img = np.clip(img, 0, 1)
            img = (img - 0.5) / 0.5  # 归一化到 [-1, 1]

            return torch.from_numpy(img), idx

    def run_torch_dataloader():
        dataset = TorchImageDataset(length=12)
        # 重要: num_workers=0 使数据加载在主线程进行，便于追踪
        # 如果 num_workers > 0，数据加载在子进程中执行，sys.settrace 无法追踪
        num_workers = 0
        assert num_workers == 0, "num_workers 必须为 0，否则追踪器无法工作"
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=num_workers)

        results = []
        for batch_idx, (images, indices) in enumerate(dataloader):
            results.append({
                'batch_idx': batch_idx,
                'shape': tuple(images.shape),
                'indices': indices.tolist()
            })
            if batch_idx >= 2:  # 只处理前3个 batch
                break

        return results

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="03c_TorchDataLoader",
        target_func=run_torch_dataloader,
        max_depth=20,
        k_partitions=2
    )

    return result


def test_data_augmentation_dataset():
    """测试带数据增强的数据集"""
    features = DependencyChecker.get_available_features()
    if not features['torch']:
        return None

    def run_augmentation():
        """模拟带增强的数据加载"""
        np.random.seed(42)

        def augment(img):
            """数据增强管道"""
            # 随机水平翻转
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1)

            # 随机亮度调整
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)

            # 随机噪声
            noise = np.random.normal(0, 0.01, img.shape)
            img = np.clip(img + noise, 0, 1)

            return img

        results = []
        for i in range(2):
            # 生成图像
            img = np.random.rand(64, 64, 3).astype(np.float32)

            # 应用增强
            img = augment(img)

            # 转换格式
            img = img.transpose(2, 0, 1)  # HWC -> CHW

            results.append(img)

        return np.stack(results)

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="03d_AugmentationDataset",
        target_func=run_augmentation,
        max_depth=15,
        k_partitions=2
    )

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("测试样例 3: PyTorch Dataset (深度学习数据加载)")
    print("=" * 60)

    if not check_dependencies():
        print("依赖检查失败，退出测试")
        sys.exit(1)

    # 运行所有子测试
    results = []

    print("\n[1/4] Mock 数据集测试...")
    r = test_mock_dataset()
    if r:
        results.append(r)

    print("\n[2/4] OpenCV 数据集测试...")
    r = test_opencv_dataset()
    if r:
        results.append(r)

    print("\n[3/4] PyTorch DataLoader 测试...")
    r = test_torch_dataloader()
    if r:
        results.append(r)

    print("\n[4/4] 数据增强数据集测试...")
    r = test_data_augmentation_dataset()
    if r:
        results.append(r)

    # 汇总
    print("\n" + "=" * 60)
    print("测试样例 3 汇总")
    print("=" * 60)
    if results:
        passed = sum(1 for r in results if r.success)
        print(f"通过: {passed}/{len(results)}")
        for r in results:
            status = "PASS" if r.success else "FAIL"
            print(f"  [{status}] {r.case_name}")
            # 显示关键统计
            print(f"      节点数: {r.node_count}, 边数: {r.edge_count}")
    else:
        print("没有测试被执行")
