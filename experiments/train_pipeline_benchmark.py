"""
CodeDAG 数据管道优化测试 (Training Loop Integrated)

测试方案设计:
1. 模拟训练场景：
   - Dataset：包含高负载的 CPU 图像处理（Resize, Blur, ColorJitter 等）
   - Model：轻量级 CNN，确保计算瓶颈主要在数据加载上（Data-Bound）
   - Task：模拟分类任务

2. 精确的时间监控：
   - Data Time: 每个 batch 的数据加载时间
   - Compute Time: 每个 batch 的 GPU 计算时间
   - Total Time: 整个训练过程的总时间
   - Throughput: Total Time / Batch Count

3. 流程：
   - Baseline: 正常运行，记录各项时间指标
   - Trace: 针对 dataset.__getitem__ 进行追踪，构建 DAG
   - Optimize: 生成迁移计划
   - Migrated Run: 应用迁移，再次运行训练，记录优化后的指标
"""

import sys
import os
import time
import threading
import numpy as np

# 引入 CodeDAG
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("警告: psutil 未安装，将跳过 CPU 利用率监控")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("警告: tabulate 未安装，将使用简单格式输出")

from core.enhanced_tracer import EnhancedTracer
from optimizer.optimizer_manager import OptimizerManager
from migration.api import PipelineMigrator


# ==========================================
# 1. 高负载图像处理 Dataset
# ==========================================
class ImagePipelineDataset(Dataset):
    """
    模拟高负载 CPU 图像处理的 Dataset

    包含以下 CPU 密集型操作：
    - cv2.resize: 大图缩放
    - cv2.cvtColor: 颜色空间转换
    - cv2.GaussianBlur: 高斯模糊
    - numpy 归一化和布局转换
    """

    def __init__(self, size=500, img_dim=1024):
        self.size = size
        self.img_dim = img_dim
        # 模拟原始数据 (H, W, C)
        print(f"正在生成 {size} 张 {img_dim}x{img_dim} 的模拟图像...")
        self.data = [np.random.randint(0, 255, (img_dim, img_dim, 3), dtype=np.uint8)
                     for _ in range(size)]
        self.labels = [np.random.randint(0, 10) for _ in range(size)]
        print("数据生成完成")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # --- 优化目标：CPU 瓶颈区域 ---
        # 1. Resize (大图缩放，CPU 计算密集)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 2. Color Space (内存拷贝密集)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Augmentation (高斯模糊，模拟复杂增强)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # 4. Normalize (浮点运算)
        img = img.astype(np.float32) / 255.0

        # 5. Layout (HWC -> CHW)
        img = img.transpose(2, 0, 1)
        # ---------------------------

        return img, label


# ==========================================
# 2. 简易模型
# ==========================================
class SimpleNet(nn.Module):
    """
    轻量级 CNN，确保计算瓶颈主要在数据加载上
    """
    def __init__(self):
        super().__init__()
        # 输入 224x224
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),  # 112
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 56
            nn.Flatten()
        )
        self.classifier = nn.Linear(32 * 56 * 56, 10)

    def forward(self, x):
        return self.classifier(self.features(x))


# ==========================================
# 3. 训练与监控核心
# ==========================================
class TrainingMonitor:
    """训练过程监控器"""

    def __init__(self):
        self.data_times = []     # 每个 batch 的数据加载耗时
        self.compute_times = []  # 每个 batch 的 GPU 计算耗时
        self.batch_times = []    # 每个 batch 的整体耗时
        self.cpu_usages = []
        self.gpu_mems = []
        self.running = False
        self.thread = None

        # 总体统计
        self.total_start_time = None
        self.total_end_time = None
        self.batch_count = 0

    def start_bg_monitor(self):
        """启动后台系统监控"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_sys)
        self.thread.start()

    def stop_bg_monitor(self):
        """停止后台监控"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_sys(self):
        """后台监控线程"""
        while self.running:
            if HAS_PSUTIL:
                self.cpu_usages.append(psutil.cpu_percent())
            if torch.cuda.is_available():
                self.gpu_mems.append(torch.cuda.memory_allocated() / 1024**2)
            time.sleep(0.05)

    def get_stats(self):
        """获取统计数据"""
        # 去掉前2个batch的预热数据计算平均值
        valid_dt = self.data_times[2:] if len(self.data_times) > 2 else self.data_times
        valid_ct = self.compute_times[2:] if len(self.compute_times) > 2 else self.compute_times
        valid_bt = self.batch_times[2:] if len(self.batch_times) > 2 else self.batch_times

        avg_data = np.mean(valid_dt) if valid_dt else 0
        avg_compute = np.mean(valid_ct) if valid_ct else 0
        avg_batch = np.mean(valid_bt) if valid_bt else 0

        # 总时间计算
        total_time = (self.total_end_time - self.total_start_time) if self.total_start_time and self.total_end_time else 0
        time_per_batch = total_time / self.batch_count if self.batch_count > 0 else 0

        # 累计时间
        sum_data = np.sum(self.data_times)
        sum_compute = np.sum(self.compute_times)

        return {
            # 总体统计
            "Total Time (s)": total_time,
            "Batch Count": self.batch_count,
            "Time/Batch (s)": time_per_batch,
            # 平均每 batch 统计（去掉预热）
            "Avg Data Time (s)": avg_data,
            "Avg Compute Time (s)": avg_compute,
            "Avg Batch Time (s)": avg_batch,
            # 累计时间
            "Sum Data Time (s)": sum_data,
            "Sum Compute Time (s)": sum_compute,
            # 比例
            "Data/Total Ratio": (avg_data / avg_batch) if avg_batch > 0 else 0,
            # 系统资源
            "Avg CPU Util (%)": np.mean(self.cpu_usages) if self.cpu_usages else 0,
            "Avg GPU Mem (MB)": np.mean(self.gpu_mems) if self.gpu_mems else 0
        }


def train_one_epoch(dataloader, model, optimizer, criterion, device,
                    migrator=None, steps=20, desc="Training"):
    """
    训练一个 epoch，带精确时间监控

    Args:
        dataloader: 数据加载器
        model: 模型
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        migrator: PipelineMigrator 实例（可选）
        steps: 训练步数
        desc: 描述

    Returns:
        统计信息字典
    """
    monitor = TrainingMonitor()
    monitor.start_bg_monitor()

    model.train()

    # 如果有 migrator，应用迁移
    if migrator:
        migrator.apply()

    try:
        iterator = iter(dataloader)

        # 预热 GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 记录总开始时间
        monitor.total_start_time = time.time()
        t_prev_end = monitor.total_start_time

        for i in range(steps):
            # =========== 1. 监控数据加载时间 (Data Time) ===========
            try:
                # 这一步触发 dataset.__getitem__
                inputs, targets = next(iterator)
            except StopIteration:
                # 数据用完了，重新创建迭代器
                iterator = iter(dataloader)
                inputs, targets = next(iterator)

            # 同步 GPU，确保计时准确
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_data_end = time.time()
            t_data = t_data_end - t_prev_end

            # 数据搬运
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy(np.ascontiguousarray(inputs))
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)

            inputs = inputs.to(device)
            targets = targets.to(device)

            # =========== 2. 训练计算 (Compute Time) ===========
            t_compute_start = time.time()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_compute_end = time.time()

            t_compute = t_compute_end - t_compute_start
            t_batch = t_compute_end - t_prev_end

            monitor.data_times.append(t_data)
            monitor.compute_times.append(t_compute)
            monitor.batch_times.append(t_batch)
            monitor.batch_count += 1

            # 检查数据来源
            data_loc = "GPU" if inputs.is_cuda else "CPU"

            # 打印进度
            print(f"\r[{desc}] Step {i+1}/{steps} | "
                  f"Data: {t_data:.3f}s | Compute: {t_compute:.3f}s | "
                  f"Total: {t_batch:.3f}s | Loc: {data_loc}", end="")

            t_prev_end = time.time()

        # 记录总结束时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        monitor.total_end_time = time.time()

    finally:
        if migrator:
            migrator.restore()
        monitor.stop_bg_monitor()
        print("")  # 换行

    return monitor.get_stats()


# ==========================================
# 4. 全流程执行
# ==========================================
def main():
    if not torch.cuda.is_available():
        print("需要 GPU 环境")
        return

    device = torch.device("cuda")
    print("=" * 70)
    print("CodeDAG 数据管道优化测试 (Training Loop Integrated)")
    print("=" * 70)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ==========================================
    # 1. 准备数据和模型
    # ==========================================
    print("\n>>> 准备数据集和模型...")

    # 数据集配置
    dataset = ImagePipelineDataset(size=200, img_dim=512)

    # 单进程 DataLoader (num_workers=0)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model = SimpleNet().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 模型预热
    print("预热模型...")
    dummy_input = torch.randn(16, 3, 224, 224).to(device)
    for _ in range(3):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    num_steps = 20

    # ==========================================
    # 2. Baseline 训练
    # ==========================================
    print("\n>>> 阶段 1: Baseline 训练 (Standard CPU Pipeline)")
    print("-" * 50)
    base_stats = train_one_epoch(
        loader, model, opt, criterion, device,
        migrator=None, steps=num_steps, desc="Baseline"
    )

    # ==========================================
    # 3. CodeDAG 追踪与优化
    # ==========================================
    print("\n>>> 阶段 2: CodeDAG 分析与优化")
    print("-" * 50)

    tracer = EnhancedTracer(max_depth=10, track_memory=False)

    print("   [Trace] 捕获 Dataset 处理逻辑...")
    with tracer.tracing_context():
        # 仅执行一次数据获取逻辑，用于构建 DAG
        _ = dataset[0]

    dag = tracer.dag_builder.dag
    print(f"   [DAG] 节点数: {len(dag.nodes)}, 边数: {len(dag.edges)}")

    print("   [Optimize] 生成迁移计划...")
    opt_mgr = OptimizerManager(k=2)
    opt_result = opt_mgr.optimize(dag)

    plan = opt_mgr.get_migration_plan(
        opt_result['optimized_partitions'],
        opt_result['coarsened_dag'],
        gpu_count=1
    )

    # 打印要迁移的操作
    context_map = plan.get('context_device_map', {})
    print(f"   [Plan] 识别到可迁移算子: {len(context_map)} 个")

    # 显示关键的 CV 操作映射
    cv_ops = [k for k in context_map.keys() if 'cv2' in k.lower() or 'resize' in k.lower()]
    if cv_ops:
        print(f"   [Plan] CV 操作映射示例:")
        for op in cv_ops[:5]:
            print(f"          {op} -> {context_map[op]}")

    # ==========================================
    # 4. Optimized 训练
    # ==========================================
    print("\n>>> 阶段 3: Optimized 训练 (With GPU Migration)")
    print("-" * 50)

    migrator = PipelineMigrator.from_plan(plan)
    opt_stats = train_one_epoch(
        loader, model, opt, criterion, device,
        migrator=migrator, steps=num_steps, desc="Optimized"
    )

    # ==========================================
    # 5. 结果对比
    # ==========================================
    print("\n" + "=" * 70)
    print("性能指标对比")
    print("=" * 70)

    # 主要指标
    main_keys = [
        "Total Time (s)",
        "Batch Count",
        "Time/Batch (s)",
    ]

    # 详细指标
    detail_keys = [
        "Avg Data Time (s)",
        "Avg Compute Time (s)",
        "Avg Batch Time (s)",
        "Sum Data Time (s)",
        "Sum Compute Time (s)",
        "Data/Total Ratio",
        "Avg CPU Util (%)",
        "Avg GPU Mem (MB)"
    ]

    def print_comparison(keys, title):
        print(f"\n{title}")
        print("-" * 70)

        if HAS_TABULATE:
            headers = ["Metric", "Baseline", "Optimized", "Change"]
            table = []

            for k in keys:
                v1 = base_stats[k]
                v2 = opt_stats[k]

                if v1 != 0:
                    diff = (v2 - v1) / v1 * 100
                else:
                    diff = 0

                # 判定好坏（时间、比例、CPU越小越好）
                if "Time" in k or "Ratio" in k or "CPU" in k:
                    better = diff < -1  # 至少改善1%
                    change_str = f"{diff:+.1f}%" + (" ✓" if better else "")
                elif "Count" in k:
                    change_str = "-"
                else:
                    change_str = f"{diff:+.1f}%"

                table.append([k, f"{v1:.4f}", f"{v2:.4f}", change_str])

            print(tabulate(table, headers=headers, tablefmt="grid"))
        else:
            print(f"{'Metric':<25} {'Baseline':>12} {'Optimized':>12} {'Change':>15}")
            print("-" * 70)
            for k in keys:
                v1 = base_stats[k]
                v2 = opt_stats[k]
                diff = ((v2 - v1) / v1 * 100) if v1 != 0 else 0
                print(f"{k:<25} {v1:>12.4f} {v2:>12.4f} {diff:>+14.1f}%")

    print_comparison(main_keys, "【总体统计】")
    print_comparison(detail_keys, "【详细统计】")

    # ==========================================
    # 6. 结论
    # ==========================================
    print("\n" + "=" * 70)
    print("分析结论")
    print("=" * 70)

    # 计算加速比
    total_speedup = base_stats["Total Time (s)"] / opt_stats["Total Time (s)"] \
        if opt_stats["Total Time (s)"] > 0 else 1
    data_speedup = base_stats["Avg Data Time (s)"] / opt_stats["Avg Data Time (s)"] \
        if opt_stats["Avg Data Time (s)"] > 0 else 1
    throughput_speedup = base_stats["Time/Batch (s)"] / opt_stats["Time/Batch (s)"] \
        if opt_stats["Time/Batch (s)"] > 0 else 1

    print(f"\n加速比:")
    print(f"  总时间加速比:     {total_speedup:.2f}x")
    print(f"  数据加载加速比:   {data_speedup:.2f}x")
    print(f"  吞吐量加速比:     {throughput_speedup:.2f}x")

    base_ratio = base_stats["Data/Total Ratio"]
    opt_ratio = opt_stats["Data/Total Ratio"]
    print(f"\nData-Bound 分析:")
    print(f"  Baseline:  {base_ratio*100:.1f}% 时间花在数据加载")
    print(f"  Optimized: {opt_ratio*100:.1f}% 时间花在数据加载")

    if total_speedup > 1.05:
        print("\n✓ [结论] CodeDAG 迁移有效提升了训练效率!")
    elif total_speedup > 0.95:
        print("\n○ [结论] 迁移效果不明显，可能计算已经是瓶颈")
    else:
        print("\n✗ [结论] 迁移后性能下降，需要检查迁移策略")


if __name__ == "__main__":
    main()
