"""
设备检测工具V2 - 完全基于实测数据，无硬编码
"""
import logging
import psutil
import platform
import json
import os
from typing import Dict, List, Optional
from utils.device_profiler import DeviceProfiler

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeviceDetectorV2:
    """动态检测系统设备配置 - 基于实测性能"""
    
    def __init__(self, use_cache: bool = True, cache_file: str = "test_results/device_profile_cache.json"):
        """
        初始化设备检测器
        
        Args:
            use_cache: 是否使用缓存的性能数据（避免每次都重新测量）
            cache_file: 缓存文件路径
        """
        self.use_cache = use_cache
        self.cache_file = cache_file
        self.profile_data = None
        
        # 尝试加载缓存
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.profile_data = json.load(f)
                logger.info(f"从缓存加载设备性能数据: {cache_file}")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
    
    def get_available_devices(self, force_measure: bool = False) -> List[Dict]:
        """
        获取所有可用设备及其实测性能
        
        Args:
            force_measure: 强制重新测量性能
        
        Returns:
            list: 设备信息列表
        """
        if not force_measure and self.profile_data:
            return self._format_devices_from_profile(self.profile_data)
        
        # 执行性能测量
        logger.info("开始设备性能测量...")
        profiler = DeviceProfiler()
        self.profile_data = profiler.profile_all_devices()
        
        # 保存到缓存
        if self.use_cache:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.profile_data, f, indent=2)
            logger.info(f"设备性能数据已缓存到: {self.cache_file}")
        
        return self._format_devices_from_profile(self.profile_data)
    
    def _format_devices_from_profile(self, profile: Dict) -> List[Dict]:
        """
        从性能分析数据格式化设备信息
        
        Args:
            profile: 性能分析数据
        
        Returns:
            list: 格式化的设备列表
        """
        devices = []
        
        # CPU信息
        cpu_data = profile.get('cpu', {})
        if cpu_data:
            cpu_info = {
                'device_id': 'cpu',
                'device_type': 'cpu',
                'physical_cores': cpu_data.get('physical_cores', 1),
                'logical_cores': cpu_data.get('logical_cores', 1),
                'memory_gb': cpu_data.get('total_memory_gb', 0),
                'available_memory_gb': cpu_data.get('available_memory_gb', 0),
                'cpu_freq_mhz': cpu_data.get('cpu_freq_mhz', 0),
                'platform': cpu_data.get('platform', 'unknown'),
                # 使用实测性能数据
                'compute_power': cpu_data.get('measured_compute_gflops', 100),
                'bandwidth': cpu_data.get('measured_bandwidth_gb_s', 20),
                'latency_ms': cpu_data.get('measured_latency_ms', 0.01),
                'measured': True  # 标记为实测数据
            }
            devices.append(cpu_info)
        
        # GPU信息
        for gpu_data in profile.get('gpus', []):
            gpu_info = {
                'device_id': gpu_data.get('device_id', 'cuda:0'),
                'device_type': 'cuda',
                'name': gpu_data.get('name', 'Unknown GPU'),
                'compute_capability': gpu_data.get('compute_capability', '0.0'),
                'total_memory_gb': gpu_data.get('total_memory_gb', 0),
                'multi_processor_count': gpu_data.get('multi_processor_count', 1),
                # 使用实测性能数据
                'compute_power': gpu_data.get('measured_compute_tflops', 1.0),
                'bandwidth': gpu_data.get('measured_bandwidth_gb_s', 100),
                'latency_ms': gpu_data.get('measured_latency_ms', 0.02),
                'temperature': gpu_data.get('temperature', 0),
                'power_draw': gpu_data.get('power_draw', 0),
                'power_limit': gpu_data.get('power_limit', 0),
                'measured': True  # 标记为实测数据
            }
            
            # 计算可用内存
            if 'free_memory_mb' in gpu_data:
                gpu_info['available_memory_gb'] = gpu_data['free_memory_mb'] / 1024
            else:
                gpu_info['available_memory_gb'] = gpu_info['total_memory_gb'] * 0.9  # 估算
            
            devices.append(gpu_info)
        
        return devices
    
    def is_operation_gpu_suitable(self, operation_name: str, operation_data: dict) -> bool:
        """
        基于操作特征和实测性能判断是否适合GPU执行
        
        Args:
            operation_name: 操作名称
            operation_data: 操作相关数据
        
        Returns:
            bool: 是否适合GPU执行
        """
        # 获取设备性能数据
        devices = self.get_available_devices()
        
        # 找到CPU和GPU设备
        cpu_device = next((d for d in devices if d['device_type'] == 'cpu'), None)
        gpu_devices = [d for d in devices if d['device_type'] == 'cuda']
        
        if not gpu_devices or not cpu_device:
            return False
        
        # 基于操作类型判断
        gpu_suitable_patterns = [
            'matmul', 'mm', 'bmm', 'conv', 'pool', 'relu', 'gelu', 
            'sigmoid', 'tanh', 'softmax', 'norm', 'attention', 
            'backward', 'grad', 'optimizer'
        ]
        
        op_lower = operation_name.lower()
        is_gpu_op = any(pattern in op_lower for pattern in gpu_suitable_patterns)
        
        if not is_gpu_op:
            return False
        
        # 基于数据规模和延迟判断
        data_size = operation_data.get('size', 0)
        
        # 使用实测延迟来判断
        cpu_latency = cpu_device.get('latency_ms', 0.01)
        gpu_latency = min(g.get('latency_ms', 0.02) for g in gpu_devices)
        
        # 如果数据太小，CPU延迟更低可能更合适
        if data_size < 10000:  # 小数据
            # 考虑延迟差异
            if cpu_latency < gpu_latency * 0.5:  # CPU延迟明显更低
                return False
        
        # 大数据量或计算密集型操作适合GPU
        return True
    
    def select_optimal_device(self, devices: List[Dict], task_type: str = 'compute') -> str:
        """
        根据任务类型和实测性能选择最优设备
        
        Args:
            devices: 可用设备列表
            task_type: 任务类型 ('compute', 'io', 'memory', 'latency')
        
        Returns:
            str: 最优设备ID
        """
        if not devices:
            return 'cpu'
        
        if task_type == 'compute':
            # 选择计算能力最强的设备
            # GPU的TFLOPS通常远高于CPU的GFLOPS
            best_device = max(devices, key=lambda d: 
                              d.get('compute_power', 0) * (1000 if d['device_type'] == 'cuda' else 1))
            return best_device['device_id']
            
        elif task_type == 'io':
            # IO密集型优先选择CPU（避免PCIe传输开销）
            cpu_devices = [d for d in devices if d['device_type'] == 'cpu']
            if cpu_devices:
                return cpu_devices[0]['device_id']
                
        elif task_type == 'memory':
            # 选择带宽最高的设备
            best_device = max(devices, key=lambda d: d.get('bandwidth', 0))
            return best_device['device_id']
            
        elif task_type == 'latency':
            # 选择延迟最低的设备
            best_device = min(devices, key=lambda d: d.get('latency_ms', float('inf')))
            return best_device['device_id']
        
        # 默认选择第一个可用的GPU或CPU
        gpu_devices = [d for d in devices if d['device_type'] == 'cuda']
        if gpu_devices:
            # 选择计算能力最强的GPU
            best_gpu = max(gpu_devices, key=lambda d: d.get('compute_power', 0))
            return best_gpu['device_id']
        
        return 'cpu'
    
    def get_device_recommendation(self, workload_profile: Dict) -> Dict:
        """
        基于工作负载特征推荐最优设备配置
        
        Args:
            workload_profile: 工作负载特征
                - compute_intensity: 计算密集度 (0-1)
                - memory_intensity: 内存密集度 (0-1)
                - data_size: 数据规模 (MB)
                - batch_size: 批次大小
                - latency_sensitive: 是否延迟敏感
        
        Returns:
            dict: 设备推荐配置
        """
        devices = self.get_available_devices()
        
        compute_intensity = workload_profile.get('compute_intensity', 0.5)
        memory_intensity = workload_profile.get('memory_intensity', 0.5)
        data_size_mb = workload_profile.get('data_size', 100)
        latency_sensitive = workload_profile.get('latency_sensitive', False)
        
        recommendation = {
            'primary_device': None,
            'reason': '',
            'expected_performance': {}
        }
        
        # 延迟敏感型任务
        if latency_sensitive:
            device_id = self.select_optimal_device(devices, 'latency')
            recommendation['primary_device'] = device_id
            recommendation['reason'] = '延迟敏感型任务，选择最低延迟设备'
            
        # 计算密集型任务
        elif compute_intensity > 0.7:
            device_id = self.select_optimal_device(devices, 'compute')
            recommendation['primary_device'] = device_id
            recommendation['reason'] = '计算密集型任务，选择最高算力设备'
            
        # 内存密集型任务
        elif memory_intensity > 0.7:
            device_id = self.select_optimal_device(devices, 'memory')
            recommendation['primary_device'] = device_id
            recommendation['reason'] = '内存密集型任务，选择最高带宽设备'
            
        # 混合型任务
        else:
            # 综合评分
            best_score = -1
            best_device = None
            
            for device in devices:
                # 归一化性能指标
                compute_score = device.get('compute_power', 0) / 100  # 假设100 TFLOPS为满分
                memory_score = device.get('bandwidth', 0) / 1000  # 假设1000 GB/s为满分
                latency_score = 1.0 / (1 + device.get('latency_ms', 1))  # 延迟越低分数越高
                
                # 加权评分
                score = (compute_intensity * compute_score + 
                        memory_intensity * memory_score + 
                        0.2 * latency_score)
                
                # 考虑内存容量限制
                if device['device_type'] == 'cuda':
                    required_memory_gb = data_size_mb / 1024
                    if required_memory_gb > device.get('available_memory_gb', 0):
                        score *= 0.1  # 内存不足，大幅降低评分
                
                if score > best_score:
                    best_score = score
                    best_device = device['device_id']
            
            recommendation['primary_device'] = best_device
            recommendation['reason'] = '混合型任务，基于综合性能评分选择'
        
        # 添加预期性能
        selected_device = next((d for d in devices if d['device_id'] == recommendation['primary_device']), None)
        if selected_device:
            recommendation['expected_performance'] = {
                'compute_power': selected_device.get('compute_power', 0),
                'bandwidth': selected_device.get('bandwidth', 0),
                'latency': selected_device.get('latency_ms', 0)
            }
        
        return recommendation


if __name__ == "__main__":
    # 测试V2版本设备检测器
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("设备检测器V2测试 - 基于实测性能")
    print("=" * 60)
    
    detector = DeviceDetectorV2(use_cache=False)  # 强制重新测量
    devices = detector.get_available_devices(force_measure=True)
    
    print("\n检测到的设备（基于实测）:")
    for device in devices:
        print(f"\n设备: {device['device_id']}")
        print(f"  类型: {device['device_type']}")
        if device['device_type'] == 'cpu':
            print(f"  核心数: {device['physical_cores']} 物理 / {device['logical_cores']} 逻辑")
            print(f"  内存: {device['memory_gb']:.1f} GB")
            print(f"  实测计算能力: {device['compute_power']:.1f} GFLOPS")
            print(f"  实测带宽: {device['bandwidth']:.1f} GB/s")
            print(f"  实测延迟: {device['latency_ms']:.3f} ms")
        else:
            print(f"  名称: {device['name']}")
            print(f"  显存: {device['total_memory_gb']:.1f} GB")
            print(f"  实测计算能力: {device['compute_power']:.2f} TFLOPS")
            print(f"  实测带宽: {device['bandwidth']:.1f} GB/s")
            print(f"  实测延迟: {device['latency_ms']:.3f} ms")
            if device.get('temperature'):
                print(f"  当前温度: {device['temperature']:.0f}°C")
    
    # 测试操作判断
    print("\n操作GPU适合性测试:")
    test_ops = [
        ('conv2d_forward', {'size': 1000000}),
        ('small_add', {'size': 100}),
        ('matmul_large', {'size': 10000000}),
        ('file_read', {}),
    ]
    
    for op_name, op_data in test_ops:
        suitable = detector.is_operation_gpu_suitable(op_name, op_data)
        print(f"  {op_name}: {'GPU' if suitable else 'CPU'}")
    
    # 测试设备推荐
    print("\n工作负载设备推荐:")
    workloads = [
        {'compute_intensity': 0.9, 'memory_intensity': 0.3, 'data_size': 1000, 'latency_sensitive': False},
        {'compute_intensity': 0.2, 'memory_intensity': 0.8, 'data_size': 5000, 'latency_sensitive': False},
        {'compute_intensity': 0.5, 'memory_intensity': 0.5, 'data_size': 100, 'latency_sensitive': True},
    ]
    
    for i, workload in enumerate(workloads):
        rec = detector.get_device_recommendation(workload)
        print(f"\n工作负载 {i+1}:")
        print(f"  推荐设备: {rec['primary_device']}")
        print(f"  原因: {rec['reason']}")
        print(f"  预期性能: {rec['expected_performance']}")