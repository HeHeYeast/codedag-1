"""
测试样例1：简单函数计算
测试基本的算术运算和函数调用的数据流图追踪
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.enhanced_tracer import EnhancedTracer
import json

def add_numbers(a, b):
    """简单的加法运算"""
    return a + b

def multiply_numbers(x, y):
    """简单的乘法运算"""
    return x * y

def compute_average(numbers):
    """计算平均值"""
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average

def simple_computation():
    """简单的数学计算流程"""
    # 基本运算
    a = 10
    b = 20
    sum_result = add_numbers(a, b)
    
    # 乘法运算
    x = 5
    y = 6
    product = multiply_numbers(x, y)
    
    # 组合运算
    final_result = add_numbers(sum_result, product)
    
    # 平均值计算
    numbers = [sum_result, product, final_result]
    avg = compute_average(numbers)
    
    return {
        'sum': sum_result,
        'product': product,
        'final': final_result,
        'average': avg
    }

def main():
    print("测试样例1：简单函数计算")
    print("=" * 50)
    
    # 创建追踪器
    tracer = EnhancedTracer(max_depth=5, track_memory=True)
    
    # 使用上下文管理器进行追踪
    with tracer.tracing_context():
        result = simple_computation()
        print(f"计算结果: {result}")
    
    # 导出数据流图
    dataflow_graph = tracer.export_dataflow_graph()
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    json_path = os.path.join(results_dir, "simple_computation_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataflow_graph, f, indent=2, ensure_ascii=False)
    
    print(f"数据流图已保存到: {json_path}")
    
    # 生成可视化
    try:
        svg_path = os.path.join(results_dir, "simple_computation_graph.svg")
        success = tracer.export_visualization(svg_path)
        if success:
            print(f"可视化图表已生成: {svg_path}")
        else:
            print("可视化生成失败")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    main()