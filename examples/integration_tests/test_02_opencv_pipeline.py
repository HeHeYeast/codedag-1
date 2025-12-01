"""
测试样例 2: 复杂计算 (OpenCV 图像处理管道)

侧重点:
- 测试图粗化（Coarsening）能力（识别连续的 cv2 调用）
- 测试参数值映射（如 dsize 处理、color code 转换）
- 测试 OpenCV -> Kornia 的迁移策略
- 验证 Dispatcher 模式 (cv2.cvtColor, cv2.threshold)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from test_harness import CodeDAGTester, DependencyChecker


def check_dependencies():
    """检查必要的依赖"""
    features = DependencyChecker.get_available_features()

    if not features['opencv']:
        print("警告: OpenCV 不可用，跳过此测试")
        return False

    if not features['kornia']:
        print("警告: Kornia 不可用，迁移功能将使用 fallback")

    return True


def create_test_images(count: int = 5, size: tuple = (800, 600)):
    """创建测试图像"""
    images = []
    for i in range(count):
        # 创建具有不同特征的图像
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        images.append(img)
    return images


def image_preprocessing_pipeline(img_list):
    """
    图像预处理管道

    这个管道模拟典型的深度学习数据预处理流程：
    1. Resize - 调整尺寸
    2. Color Convert - 颜色空间转换
    3. GaussianBlur - 高斯模糊（数据增强）
    4. Normalize - 归一化

    CodeDAG 应该能：
    - 追踪这些 cv2 调用
    - 识别循环中的重复模式进行图粗化
    - 在迁移时将 cv2 函数替换为 kornia 等价物
    """
    import cv2

    processed = []
    for img in img_list:
        # 1. Resize (cv2.resize -> kornia.geometry.transform.resize)
        # 注意: dsize 参数需要 swap_hw 映射
        x = cv2.resize(img, (224, 224))

        # 2. Convert Color (cv2.cvtColor -> kornia.color.*)
        # 使用 Dispatcher 根据 color code 选择具体函数
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        # 3. GaussianBlur (cv2.GaussianBlur -> kornia.filters.gaussian_blur2d)
        x = cv2.GaussianBlur(x, (5, 5), 0)

        # 4. Normalize (算术操作)
        x = x.astype(np.float32) / 255.0

        processed.append(x)

    return processed


def image_augmentation_pipeline(img_list):
    """
    图像增强管道

    测试更多 OpenCV 函数的迁移能力：
    - cv2.flip
    - cv2.threshold
    - cv2.blur
    """
    import cv2

    augmented = []
    for img in img_list:
        # 水平翻转 (cv2.flip -> torch.flip)
        flipped = cv2.flip(img, 1)

        # 转灰度
        gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)

        # 阈值处理 (cv2.threshold -> Dispatcher)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 均值模糊 (cv2.blur -> kornia.filters.box_blur)
        blurred = cv2.blur(binary, (3, 3))

        augmented.append(blurred)

    return augmented


def edge_detection_pipeline(img_list):
    """
    边缘检测管道

    测试边缘检测相关函数：
    - cv2.Canny
    - cv2.Sobel
    - cv2.Laplacian
    """
    import cv2

    edges_list = []
    for img in img_list:
        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Canny 边缘检测 (cv2.Canny -> kornia.filters.canny)
        canny = cv2.Canny(gray, 100, 200)

        # Sobel 算子 (cv2.Sobel -> kornia.filters.sobel)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 合并边缘
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)

        edges_list.append({
            'canny': canny,
            'sobel': edges
        })

    return edges_list


def test_preprocessing_pipeline():
    """测试图像预处理管道"""
    if not check_dependencies():
        return None

    # 创建测试图像
    images = create_test_images(count=5, size=(800, 600))

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="02a_CV_Preprocessing",
        target_func=image_preprocessing_pipeline,
        args=(images,),
        max_depth=15,
        track_memory=True,
        k_partitions=2
    )

    return result


def test_augmentation_pipeline():
    """测试图像增强管道"""
    if not check_dependencies():
        return None

    images = create_test_images(count=5, size=(640, 480))

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="02b_CV_Augmentation",
        target_func=image_augmentation_pipeline,
        args=(images,),
        max_depth=15,
        k_partitions=2
    )

    return result


def test_edge_detection_pipeline():
    """测试边缘检测管道"""
    if not check_dependencies():
        return None

    images = create_test_images(count=3, size=(512, 512))

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="02c_CV_EdgeDetection",
        target_func=edge_detection_pipeline,
        args=(images,),
        max_depth=15,
        k_partitions=2
    )

    return result


def test_single_image_operations():
    """测试单图像操作（验证参数映射）"""
    if not check_dependencies():
        return None

    import cv2

    def single_image_ops():
        """单图像操作，便于调试参数映射"""
        # 创建测试图像
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # 测试各种尺寸格式
        resized1 = cv2.resize(img, (50, 50))  # (W, H) 格式
        resized2 = cv2.resize(img, dsize=(75, 75))  # 关键字参数

        # 测试颜色转换
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 测试各种模糊
        gaussian = cv2.GaussianBlur(img, (5, 5), 1.0)
        median = cv2.medianBlur(img, 5)
        box = cv2.blur(img, (3, 3))

        return {
            'resized1_shape': resized1.shape,
            'resized2_shape': resized2.shape,
            'rgb_shape': rgb.shape,
            'hsv_shape': hsv.shape,
            'gray_shape': gray.shape,
            'gaussian_shape': gaussian.shape,
            'median_shape': median.shape,
            'box_shape': box.shape,
        }

    tester = CodeDAGTester(output_dir="results", clear_existing=False)

    result = tester.run_test(
        case_name="02d_CV_SingleImage",
        target_func=single_image_ops,
        max_depth=10,
        k_partitions=2
    )

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("测试样例 2: OpenCV 图像处理管道")
    print("=" * 60)

    if not check_dependencies():
        print("依赖检查失败，退出测试")
        sys.exit(1)

    # 运行所有子测试
    results = []

    print("\n[1/4] 图像预处理管道测试...")
    r = test_preprocessing_pipeline()
    if r:
        results.append(r)

    print("\n[2/4] 图像增强管道测试...")
    r = test_augmentation_pipeline()
    if r:
        results.append(r)

    print("\n[3/4] 边缘检测管道测试...")
    r = test_edge_detection_pipeline()
    if r:
        results.append(r)

    print("\n[4/4] 单图像操作测试...")
    r = test_single_image_operations()
    if r:
        results.append(r)

    # 汇总
    print("\n" + "=" * 60)
    print("测试样例 2 汇总")
    print("=" * 60)
    if results:
        passed = sum(1 for r in results if r.success)
        print(f"通过: {passed}/{len(results)}")
        for r in results:
            status = "PASS" if r.success else "FAIL"
            print(f"  [{status}] {r.case_name}")
            if r.warnings:
                for w in r.warnings[:3]:  # 只显示前3个警告
                    print(f"      警告: {w}")
    else:
        print("没有测试被执行")
