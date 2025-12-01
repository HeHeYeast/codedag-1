"""
测试样例2：复杂函数计算
测试包含循环、条件语句和复杂逻辑的数据流图追踪
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.enhanced_tracer import EnhancedTracer
import json
import math

def matrix_operations():
    """矩阵运算"""
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = 0
    
    # 嵌套循环处理
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val % 2 == 0:
                result += val * 2
            else:
                result += val
    
    return result

def fibonacci_computation(n=10):
    """斐波那契数列计算"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for i in range(2, n + 1):
        temp = a + b
        a = b
        b = temp
    
    return b

def complex_math_operations():
    """复杂数学运算"""
    numbers = [1.0, 4.0, 9.0, 16.0, 25.0]
    results = []
    
    for num in numbers:
        sqrt_val = math.sqrt(num)
        log_val = math.log(num) if num > 0 else 0
        
        # 条件处理
        if sqrt_val > 3.0:
            processed = sqrt_val * log_val
        else:
            processed = sqrt_val + log_val
        
        results.append(processed)
    
    # 聚合计算
    total_sum = sum(results)
    average = total_sum / len(results)
    
    return {
        'results': results,
        'sum': total_sum,
        'average': average
    }

def complex_computation():
    """复杂计算的入口函数"""
    # 矩阵运算
    matrix_result = matrix_operations()
    
    # 斐波那契计算
    fib_result = fibonacci_computation()
    
    # 复杂数学运算
    math_result = complex_math_operations()
    
    # 综合结果
    final_result = {
        'matrix_sum': matrix_result,
        'fibonacci': fib_result,
        'math_operations': math_result,
        'combined_total': matrix_result + fib_result + math_result['sum']
    }
    
    return final_result

def main():
    print("测试样例2：复杂函数计算")
    print("=" * 50)
    
    # 创建追踪器
    tracer = EnhancedTracer(max_depth=6, track_memory=True)
    
    # 使用上下文管理器进行追踪
    with tracer.tracing_context():
        result = complex_computation()
        print(f"计算结果: {result}")
    
    # 导出数据流图
    dataflow_graph = tracer.export_dataflow_graph()
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    json_path = os.path.join(results_dir, "complex_computation_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataflow_graph, f, indent=2, ensure_ascii=False)
    
    print(f"数据流图已保存到: {json_path}")
    
    # 生成可视化
    try:
        svg_path = os.path.join(results_dir, "complex_computation_graph.svg")
        success = tracer.export_visualization(svg_path)
        if success:
            print(f"可视化图表已生成: {svg_path}")
        else:
            print("可视化生成失败")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    main()