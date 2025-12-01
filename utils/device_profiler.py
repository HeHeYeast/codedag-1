"""
设备性能分析器 - 通过实际测量获取设备性能数据
"""
import torch
import time
import psutil
import platform
import subprocess
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DeviceProfiler:
    """通过实际测量获取设备性能特征"""
    
    @staticmethod
    def measure_memory_bandwidth(device: str = 'cpu', size_mb: int = 500, iterations: int = 10) -> float:
        """
        测量内存带宽
        
        Args:
            device: 设备名称 ('cpu' 或 'cuda:0' 等)
            size_mb: 测试数据大小（MB）- 需要足够大以避免缓存
            iterations: 测试迭代次数
        
        Returns:
            float: 测量的带宽（GB/s）
        """
        # 创建测试数据 - 使用更大的数据避免缓存影响
        elements = size_mb * 1024 * 1024 // 4  # float32元素数
        
        try:
            if device == 'cpu':
                # CPU内存带宽测试
                src = torch.randn(elements, device='cpu')
                dst = torch.empty_like(src)
                
                # 预热
                for _ in range(3):
                    dst.copy_(src)
                
                # 测量
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                
                for _ in range(iterations):
                    dst.copy_(src)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
            elif 'cuda' in device:
                if not torch.cuda.is_available():
                    logger.warning(f"CUDA不可用，无法测量{device}带宽")
                    return 0.0
                
                # GPU内存带宽测试 - 使用不同的源和目标避免缓存
                # 创建多个缓冲区来避免缓存优化
                buffers = []
                for i in range(4):
                    buffers.append(torch.randn(elements, device=device))
                
                # 预热
                for _ in range(3):
                    torch.cuda.synchronize()
                    _ = buffers[0].clone()
                
                # 测量 - 使用clone而不是copy_避免就地操作优化
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                for i in range(iterations):
                    src_idx = i % len(buffers)
                    _ = buffers[src_idx].clone()
                
                torch.cuda.synchronize()
                end_time = time.perf_counter()
            else:
                return 0.0
            
            # 计算带宽
            elapsed_time = end_time - start_time
            data_transferred = size_mb * iterations * 2 / 1024  # GB (读+写)
            bandwidth = data_transferred / elapsed_time  # GB/s
            
            return bandwidth
            
        except Exception as e:
            logger.error(f"测量{device}带宽时出错: {e}")
            return 0.0
    
    @staticmethod
    def measure_compute_power(device: str = 'cpu', matrix_size: int = 1024, iterations: int = 10) -> float:
        """
        测量计算能力
        
        Args:
            device: 设备名称
            matrix_size: 矩阵大小
            iterations: 测试迭代次数
        
        Returns:
            float: 测量的计算能力（GFLOPS/TFLOPS）
        """
        try:
            # 创建测试矩阵
            if device == 'cpu':
                a = torch.randn(matrix_size, matrix_size, device='cpu')
                b = torch.randn(matrix_size, matrix_size, device='cpu')
            elif 'cuda' in device:
                if not torch.cuda.is_available():
                    logger.warning(f"CUDA不可用，无法测量{device}计算能力")
                    return 0.0
                a = torch.randn(matrix_size, matrix_size, device=device)
                b = torch.randn(matrix_size, matrix_size, device=device)
            else:
                return 0.0
            
            # 预热
            for _ in range(3):
                c = torch.matmul(a, b)
            
            # 测量
            if 'cuda' in device:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            for _ in range(iterations):
                c = torch.matmul(a, b)
            
            if 'cuda' in device:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # 计算FLOPS
            # 矩阵乘法的浮点操作数: 2 * n^3
            flops = 2 * (matrix_size ** 3) * iterations
            elapsed_time = end_time - start_time
            gflops = flops / elapsed_time / 1e9
            
            if 'cuda' in device:
                return gflops / 1000  # 转换为TFLOPS
            else:
                return gflops  # GFLOPS
            
        except Exception as e:
            logger.error(f"测量{device}计算能力时出错: {e}")
            return 0.0
    
    @staticmethod
    def measure_latency(device: str = 'cpu', size: int = 1000) -> float:
        """
        测量设备延迟
        
        Args:
            device: 设备名称
            size: 测试数据大小
        
        Returns:
            float: 平均延迟（ms）
        """
        try:
            # 创建小数据测试延迟
            if device == 'cpu':
                data = torch.randn(size, device='cpu')
            elif 'cuda' in device:
                if not torch.cuda.is_available():
                    return 0.0
                data = torch.randn(size, device=device)
            else:
                return 0.0
            
            # 测量多次小操作的延迟
            latencies = []
            
            for _ in range(100):
                if 'cuda' in device:
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                result = data * 2.0  # 简单操作
                
                if 'cuda' in device:
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
            
            # 返回中位数以避免异常值影响
            return np.median(latencies)
            
        except Exception as e:
            logger.error(f"测量{device}延迟时出错: {e}")
            return 0.0
    
    @staticmethod
    def get_gpu_properties_via_nvidia_smi(device_id: int) -> Dict:
        """
        通过nvidia-smi获取GPU属性
        
        Args:
            device_id: GPU设备ID
        
        Returns:
            dict: GPU属性
        """
        properties = {}
        
        try:
            # 获取GPU名称
            cmd = f"nvidia-smi -i {device_id} --query-gpu=name --format=csv,noheader,nounits"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                properties['name'] = result.stdout.strip()
            
            # 获取内存信息
            cmd = f"nvidia-smi -i {device_id} --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                memory_info = result.stdout.strip().split(', ')
                properties['total_memory_mb'] = float(memory_info[0])
                properties['used_memory_mb'] = float(memory_info[1])
                properties['free_memory_mb'] = float(memory_info[2])
            
            # 获取功耗信息
            cmd = f"nvidia-smi -i {device_id} --query-gpu=power.draw,power.limit --format=csv,noheader,nounits"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                power_info = result.stdout.strip().split(', ')
                properties['power_draw'] = float(power_info[0])
                properties['power_limit'] = float(power_info[1])
            
            # 获取温度
            cmd = f"nvidia-smi -i {device_id} --query-gpu=temperature.gpu --format=csv,noheader,nounits"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                properties['temperature'] = float(result.stdout.strip())
            
            # 获取利用率
            cmd = f"nvidia-smi -i {device_id} --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                util_info = result.stdout.strip().split(', ')
                properties['gpu_utilization'] = float(util_info[0])
                properties['memory_utilization'] = float(util_info[1])
                
        except Exception as e:
            logger.error(f"通过nvidia-smi获取GPU {device_id}属性时出错: {e}")
        
        return properties
    
    @staticmethod
    def profile_all_devices() -> Dict:
        """
        分析所有可用设备的性能
        
        Returns:
            dict: 设备性能分析结果
        """
        profile = {
            'cpu': {},
            'gpus': []
        }
        
        # CPU性能分析
        logger.info("开始CPU性能分析...")
        profile['cpu'] = {
            'device_type': 'cpu',
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'platform': platform.processor(),
            # 实测性能
            'measured_bandwidth_gb_s': DeviceProfiler.measure_memory_bandwidth('cpu'),
            'measured_compute_gflops': DeviceProfiler.measure_compute_power('cpu'),
            'measured_latency_ms': DeviceProfiler.measure_latency('cpu')
        }
        logger.info(f"CPU - 带宽: {profile['cpu']['measured_bandwidth_gb_s']:.1f} GB/s, "
                   f"计算: {profile['cpu']['measured_compute_gflops']:.1f} GFLOPS, "
                   f"延迟: {profile['cpu']['measured_latency_ms']:.3f} ms")
        
        # GPU性能分析
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"发现 {device_count} 个GPU设备")
            
            for i in range(device_count):
                logger.info(f"开始GPU {i} 性能分析...")
                device_name = f'cuda:{i}'
                
                # 获取基本属性
                props = torch.cuda.get_device_properties(i)
                
                gpu_profile = {
                    'device_id': device_name,
                    'device_type': 'cuda',
                    'name': props.name,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count,
                    'total_memory_gb': props.total_memory / (1024**3),
                    # 实测性能
                    'measured_bandwidth_gb_s': DeviceProfiler.measure_memory_bandwidth(device_name),
                    'measured_compute_tflops': DeviceProfiler.measure_compute_power(device_name),
                    'measured_latency_ms': DeviceProfiler.measure_latency(device_name)
                }
                
                # 添加nvidia-smi信息
                nvidia_props = DeviceProfiler.get_gpu_properties_via_nvidia_smi(i)
                gpu_profile.update(nvidia_props)
                
                profile['gpus'].append(gpu_profile)
                
                logger.info(f"GPU {i} - 带宽: {gpu_profile['measured_bandwidth_gb_s']:.1f} GB/s, "
                           f"计算: {gpu_profile['measured_compute_tflops']:.2f} TFLOPS, "
                           f"延迟: {gpu_profile['measured_latency_ms']:.3f} ms")
        
        return profile
    
    @staticmethod
    def save_profile(profile: Dict, filepath: str = "device_profile.json"):
        """保存设备性能分析结果"""
        import json
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"设备性能分析结果已保存到 {filepath}")
    
    @staticmethod
    def load_profile(filepath: str = "device_profile.json") -> Optional[Dict]:
        """加载设备性能分析结果"""
        import json
        import os
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None


if __name__ == "__main__":
    # 测试设备性能分析
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("设备性能分析器测试")
    print("=" * 60)
    
    profiler = DeviceProfiler()
    profile = profiler.profile_all_devices()
    
    print("\n分析结果汇总:")
    print("-" * 40)
    
    # CPU结果
    cpu = profile['cpu']
    print(f"\nCPU性能:")
    print(f"  核心: {cpu['physical_cores']} 物理 / {cpu['logical_cores']} 逻辑")
    print(f"  频率: {cpu['cpu_freq_mhz']:.0f} MHz")
    print(f"  内存: {cpu['total_memory_gb']:.1f} GB")
    print(f"  实测带宽: {cpu['measured_bandwidth_gb_s']:.1f} GB/s")
    print(f"  实测计算: {cpu['measured_compute_gflops']:.1f} GFLOPS")
    print(f"  实测延迟: {cpu['measured_latency_ms']:.3f} ms")
    
    # GPU结果
    for gpu in profile['gpus']:
        print(f"\n{gpu['device_id']} ({gpu['name']}):")
        print(f"  计算能力: {gpu['compute_capability']}")
        print(f"  SM数量: {gpu['multi_processor_count']}")
        print(f"  显存: {gpu['total_memory_gb']:.1f} GB")
        print(f"  实测带宽: {gpu['measured_bandwidth_gb_s']:.1f} GB/s")
        print(f"  实测计算: {gpu['measured_compute_tflops']:.2f} TFLOPS")
        print(f"  实测延迟: {gpu['measured_latency_ms']:.3f} ms")
        
        if 'temperature' in gpu:
            print(f"  温度: {gpu['temperature']:.0f}°C")
        if 'power_draw' in gpu:
            print(f"  功耗: {gpu['power_draw']:.0f}W / {gpu['power_limit']:.0f}W")
    
    # 保存结果
    profiler.save_profile(profile, "test_results/device_profile.json")
    print(f"\n详细结果已保存到 test_results/device_profile.json")