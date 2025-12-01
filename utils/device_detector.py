"""
设备检测工具 - 动态检测系统硬件配置
"""
import logging
import psutil
import platform
import multiprocessing

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeviceDetector:
    """动态检测系统设备配置"""
    
    @staticmethod
    def get_available_devices():
        """获取所有可用设备"""
        devices = []
        
        # CPU信息
        cpu_info = DeviceDetector.get_cpu_info()
        devices.append(cpu_info)
        
        # GPU信息
        if TORCH_AVAILABLE:
            gpu_devices = DeviceDetector.get_gpu_devices()
            devices.extend(gpu_devices)
        
        return devices
    
    @staticmethod
    def get_cpu_info():
        """获取CPU实际配置"""
        cpu_info = {
            'device_id': 'cpu',
            'device_type': 'cpu',
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'platform': platform.processor(),
            # 基于实际硬件估算性能
            'compute_power': DeviceDetector._estimate_cpu_compute_power(),
            'bandwidth': DeviceDetector._estimate_memory_bandwidth()
        }
        return cpu_info
    
    @staticmethod
    def get_gpu_devices():
        """获取所有GPU设备信息"""
        devices = []
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return devices
        
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            # 获取实际GPU内存
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            
            gpu_info = {
                'device_id': f'cuda:{i}',
                'device_type': 'cuda',
                'name': props.name,
                'total_memory_gb': total_memory,
                'allocated_memory_gb': allocated_memory,
                'available_memory_gb': total_memory - allocated_memory,
                'multi_processor_count': props.multi_processor_count,
                'cuda_cores': DeviceDetector._estimate_cuda_cores(props),
                # 基于GPU型号估算性能
                'compute_power': DeviceDetector._estimate_gpu_compute_power(props),
                'bandwidth': DeviceDetector._estimate_gpu_bandwidth(props)
            }
            devices.append(gpu_info)
        
        return devices
    
    @staticmethod
    def _estimate_cpu_compute_power():
        """基于实际CPU估算计算能力（GFLOPS）"""
        cores = psutil.cpu_count(logical=False) or 4
        freq_ghz = (psutil.cpu_freq().current if psutil.cpu_freq() else 2000) / 1000
        # 假设每个核心每周期4个浮点运算（现代CPU典型值）
        ops_per_cycle = 4
        return cores * freq_ghz * ops_per_cycle
    
    @staticmethod
    def _estimate_memory_bandwidth():
        """估算内存带宽（GB/s）"""
        # 根据平台估算典型DDR4/DDR5带宽
        # DDR4-2400: ~19.2 GB/s per channel
        # DDR4-3200: ~25.6 GB/s per channel  
        # 假设双通道
        return 25.6 * 2  # ~51.2 GB/s for dual channel DDR4-3200
    
    @staticmethod
    def _estimate_cuda_cores(props):
        """估算CUDA核心数"""
        # 基于已知GPU架构的CUDA核心数
        sm_to_cores = {
            # Kepler
            (3, 0): 192, (3, 5): 192, (3, 7): 192,
            # Maxwell
            (5, 0): 128, (5, 2): 128, (5, 3): 128,
            # Pascal
            (6, 0): 64, (6, 1): 128, (6, 2): 128,
            # Volta, Turing
            (7, 0): 64, (7, 2): 64, (7, 5): 64,
            # Ampere
            (8, 0): 64, (8, 6): 128, (8, 7): 128,
            # Ada Lovelace
            (8, 9): 128,
            # Hopper
            (9, 0): 128
        }
        
        cores_per_sm = sm_to_cores.get(
            (props.major, props.minor),
            64  # 默认值
        )
        return props.multi_processor_count * cores_per_sm
    
    @staticmethod  
    def _estimate_gpu_compute_power(props):
        """基于GPU规格估算计算能力（TFLOPS）"""
        cuda_cores = DeviceDetector._estimate_cuda_cores(props)
        # GPU boost clock通常在1.5-2.0 GHz范围
        clock_ghz = 1.7  # 保守估计
        # 每个CUDA核心每周期2个浮点运算（FP32）
        ops_per_cycle = 2
        return (cuda_cores * clock_ghz * ops_per_cycle) / 1000  # TFLOPS
    
    @staticmethod
    def _estimate_gpu_bandwidth(props):
        """估算GPU内存带宽（GB/s）"""
        # 基于GPU型号估算典型带宽
        # 这是一个保守估计，实际带宽可能更高
        name = props.name.lower()
        
        # 基于已知GPU型号的带宽
        if 'rtx 3090' in name or 'rtx 3080' in name:
            return 936.0  # GB/s
        elif 'rtx 3070' in name:
            return 448.0
        elif 'rtx 3060' in name:
            return 360.0
        elif 'rtx 2080' in name:
            return 616.0
        elif 'rtx 2070' in name:
            return 448.0
        elif 'v100' in name:
            return 900.0
        elif 'a100' in name:
            return 1555.0
        elif 't4' in name:
            return 320.0
        else:
            # 默认估算基于计算能力
            return 400.0  # 保守估计
    
    @staticmethod
    def is_operation_gpu_suitable(operation_name: str, operation_data: dict) -> bool:
        """
        基于操作特征判断是否适合GPU执行
        
        Args:
            operation_name: 操作名称
            operation_data: 操作相关数据（如张量大小、计算复杂度等）
        
        Returns:
            bool: 是否适合GPU执行
        """
        # 基于操作特征的判断规则
        gpu_suitable_patterns = [
            # 矩阵运算
            'matmul', 'mm', 'bmm', 'addmm',
            # 卷积操作
            'conv', 'conv1d', 'conv2d', 'conv3d', 
            # 池化操作
            'pool', 'maxpool', 'avgpool', 'adaptive',
            # 激活函数
            'relu', 'gelu', 'sigmoid', 'tanh', 'softmax',
            # 归一化
            'norm', 'batch_norm', 'layer_norm', 'group_norm',
            # 注意力机制
            'attention', 'multi_head', 'scaled_dot',
            # 损失函数
            'cross_entropy', 'mse_loss', 'nll_loss',
            # 优化器步骤
            'backward', 'grad', 'optimizer_step'
        ]
        
        # 检查操作名是否包含GPU适合的模式
        op_lower = operation_name.lower()
        for pattern in gpu_suitable_patterns:
            if pattern in op_lower:
                # 检查数据规模是否足够大
                if 'size' in operation_data:
                    # 如果数据量太小（<1000个元素），CPU可能更快
                    total_elements = operation_data.get('size', 1)
                    if total_elements < 1000:
                        return False
                return True
        
        # 检查是否是张量操作
        if 'tensor' in op_lower or 'cuda' in str(operation_data.get('device', '')):
            return True
            
        return False
    
    @staticmethod
    def select_optimal_device(devices: list, task_type: str = 'compute') -> str:
        """
        根据任务类型选择最优设备
        
        Args:
            devices: 可用设备列表
            task_type: 任务类型 ('compute', 'io', 'memory')
        
        Returns:
            str: 最优设备ID
        """
        if not devices:
            return 'cpu'
        
        if task_type == 'compute':
            # 计算密集型选择计算能力最强的设备
            best_device = max(devices, key=lambda d: d.get('compute_power', 0))
            return best_device['device_id']
        elif task_type == 'io':
            # IO密集型优先选择CPU
            cpu_devices = [d for d in devices if d['device_type'] == 'cpu']
            if cpu_devices:
                return cpu_devices[0]['device_id']
        elif task_type == 'memory':
            # 内存密集型选择可用内存最大的设备
            best_device = max(devices, key=lambda d: d.get('available_memory_gb', 0))
            return best_device['device_id']
        
        # 默认返回第一个GPU设备或CPU
        gpu_devices = [d for d in devices if d['device_type'] == 'cuda']
        if gpu_devices:
            return gpu_devices[0]['device_id']
        return 'cpu'


def get_system_info():
    """获取完整系统信息用于日志记录"""
    detector = DeviceDetector()
    devices = detector.get_available_devices()
    
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'devices': devices
    }
    
    logger.info(f"系统信息: {info}")
    return info


if __name__ == "__main__":
    # 测试设备检测
    logging.basicConfig(level=logging.INFO)
    
    detector = DeviceDetector()
    devices = detector.get_available_devices()
    
    print("检测到的设备:")
    for device in devices:
        print(f"\n设备: {device['device_id']}")
        print(f"  类型: {device['device_type']}")
        if device['device_type'] == 'cpu':
            print(f"  核心数: {device['physical_cores']} 物理 / {device['logical_cores']} 逻辑")
            print(f"  内存: {device['memory_gb']:.1f} GB")
            print(f"  计算能力: {device['compute_power']:.1f} GFLOPS")
        else:
            print(f"  名称: {device['name']}")
            print(f"  内存: {device['total_memory_gb']:.1f} GB")
            print(f"  CUDA核心: {device['cuda_cores']}")
            print(f"  计算能力: {device['compute_power']:.1f} TFLOPS")
            print(f"  带宽: {device['bandwidth']:.1f} GB/s")
    
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