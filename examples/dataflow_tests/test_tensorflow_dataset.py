"""
测试样例4：TensorFlow风格数据处理
测试TensorFlow风格的数据处理流程 - 不模拟tf.data API
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.enhanced_tracer import EnhancedTracer
import numpy as np
import json

def load_dataset():
    """加载数据集"""
    # 模拟加载原始数据
    images = np.random.randn(24, 224, 224, 3).astype(np.float32)
    labels = np.random.randint(0, 10, 24)
    return images, labels

def drop_study_inst_id(images, labels):
    """移除study_inst_id (在这里就是简单返回原数据)"""
    return images, labels

def preprocess_image(images):
    """图像预处理"""
    processed_images = []
    
    for image in images:
        # 标准化
        normalized = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        # 数据增强
        if np.random.rand() > 0.5:
            normalized = np.fliplr(normalized)  # 水平翻转
        
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.05, image.shape)
            normalized = normalized + noise
        
        normalized = np.clip(normalized, -3, 3)
        processed_images.append(normalized)
    
    return np.stack(processed_images)

def create_batches(images, labels, batch_size):
    """创建批次"""
    batches = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batches.append((batch_images, batch_labels))
    return batches

def tensorflow_pipeline():
    """TensorFlow风格的数据处理流程"""
    batch_size = 4
    
    # 1. 加载数据集
    images, labels = load_dataset()
    print(f"加载数据: {images.shape}")
    
    # 2. 移除不需要的字段
    images, labels = drop_study_inst_id(images, labels)
    
    # 3. 图像预处理
    processed_images = preprocess_image(images)
    print(f"预处理完成: {processed_images.shape}")
    
    # 4. 创建批次
    batches = create_batches(processed_images, labels, batch_size)
    print(f"创建了 {len(batches)} 个批次")
    
    # 5. 处理前3个批次
    batch_results = []
    for i, (batch_imgs, batch_lbls) in enumerate(batches[:3]):
        batch_info = {
            'batch_id': i,
            'batch_shape': batch_imgs.shape,
            'mean': float(np.mean(batch_imgs)),
            'std': float(np.std(batch_imgs)),
            'label_distribution': np.bincount(batch_lbls, minlength=10).tolist()
        }
        batch_results.append(batch_info)
    
    return {
        'total_samples': len(images),
        'batch_size': batch_size,
        'processed_batches': len(batch_results),
        'batch_details': batch_results
    }

def main():
    print("测试样例4：TensorFlow风格数据处理")
    print("=" * 50)
    
    # 创建追踪器
    tracer = EnhancedTracer(max_depth=7, track_memory=True)
    
    # 使用上下文管理器进行追踪
    with tracer.tracing_context():
        result = tensorflow_pipeline()
        print(f"数据处理结果: {result}")
    
    # 导出数据流图
    dataflow_graph = tracer.export_dataflow_graph()
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    json_path = os.path.join(results_dir, "tensorflow_dataset_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataflow_graph, f, indent=2, ensure_ascii=False)
    
    print(f"数据流图已保存到: {json_path}")
    
    # 生成可视化
    try:
        svg_path = os.path.join(results_dir, "tensorflow_dataset_graph.svg")
        success = tracer.export_visualization(svg_path)
        if success:
            print(f"可视化图表已生成: {svg_path}")
        else:
            print("可视化生成失败")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    main()