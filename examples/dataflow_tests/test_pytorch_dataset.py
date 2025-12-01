"""
测试样例3：PyTorch Dataset
测试PyTorch数据集的数据流图追踪 - 只追踪关键的数据获取入口函数
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.enhanced_tracer import EnhancedTracer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

class SimpleImageDataset(Dataset):
    """简单的图像数据集"""
    
    def __init__(self, size=16):
        self.size = size
        self.data = torch.randn(size, 3, 32, 32)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return self.size
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 标准化
        normalized = (image - image.mean()) / (image.std() + 1e-8)
        
        # 简单的数据增强
        if torch.rand(1) > 0.5:
            normalized = normalized * 1.1
        
        return normalized
    
    def apply_transforms(self, image):
        """应用变换"""
        # resize操作
        resized = F.interpolate(image.unsqueeze(0), size=(28, 28), 
                              mode='bilinear', align_corners=False).squeeze(0)
        
        # 添加噪声
        if torch.rand(1) > 0.7:
            noise = torch.randn_like(resized) * 0.1
            resized = resized + noise
        
        return resized
    
    def __getitem__(self, idx):
        """数据获取方法 - 包含完整的数据处理pipeline"""
        raw_image = self.data[idx]
        label = self.labels[idx]
        
        # 预处理
        processed_image = self.preprocess_image(raw_image)
        
        # 应用变换
        final_image = self.apply_transforms(processed_image)
        
        return final_image, label

def pytorch_dataset_test():
    """PyTorch数据集测试的入口函数"""
    # 创建数据集
    dataset = SimpleImageDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # 获取一个batch的数据
    batch_data = []
    batch_labels = []
    
    for i, (data, label) in enumerate(dataloader):
        batch_data.append(data)
        batch_labels.append(label)
        if i >= 1:  # 只获取2个batch
            break
    
    # 简单的处理
    all_data = torch.cat(batch_data, dim=0)
    all_labels = torch.cat(batch_labels, dim=0)
    
    # 计算统计信息
    stats = {
        'batch_count': len(batch_data),
        'total_samples': all_data.shape[0],
        'data_shape': all_data.shape,
        'mean': all_data.mean().item(),
        'std': all_data.std().item(),
        'label_distribution': torch.bincount(all_labels).tolist()
    }
    
    return stats

def main():
    print("测试样例3：PyTorch Dataset")
    print("=" * 50)
    
    # 创建追踪器 - 使用激进过滤减少类型检查噪音
    from core.function_filter import FilterLevel
    tracer = EnhancedTracer(
        max_depth=6, 
        track_memory=True,
        filter_level=FilterLevel.AGGRESSIVE
    )
    
    # 在追踪外创建数据集和数据加载器
    dataset = SimpleImageDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch_data = []
    batch_labels = []
    
    # 只追踪数据获取过程 - 这里会触发 __getitem__, preprocess_image, apply_transforms
    with tracer.tracing_context():
        for i, (data, label) in enumerate(dataloader):
            batch_data.append(data)
            batch_labels.append(label)
            if i >= 1:  # 只获取2个batch
                break
    
    # 在追踪外处理数据和计算统计
    all_data = torch.cat(batch_data, dim=0)
    all_labels = torch.cat(batch_labels, dim=0)
    
    result = {
        'batch_count': len(batch_data),
        'total_samples': all_data.shape[0],
        'data_shape': all_data.shape,
        'mean': all_data.mean().item(),
        'std': all_data.std().item(),
        'label_distribution': torch.bincount(all_labels).tolist()
    }
    
    print(f"数据集处理结果: {result}")
    
    # 导出数据流图
    dataflow_graph = tracer.export_dataflow_graph()
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    json_path = os.path.join(results_dir, "pytorch_dataset_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataflow_graph, f, indent=2, ensure_ascii=False)
    
    print(f"数据流图已保存到: {json_path}")
    
    # 生成可视化
    try:
        svg_path = os.path.join(results_dir, "pytorch_dataset_graph.svg")
        success = tracer.export_visualization(svg_path)
        if success:
            print(f"可视化图表已生成: {svg_path}")
        else:
            print("可视化生成失败")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    main()