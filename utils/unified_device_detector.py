"""
统一设备检测器 - 自动选择最佳版本并提供回退机制
结合V1（基于估算）和V2（基于实测）的优势
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class UnifiedDeviceDetector:
    """
    统一的设备检测器，智能选择和回退机制
    
    优先级：
    1. DeviceDetectorV2 - 基于实测数据（最准确）
    2. DeviceDetectorV1 - 基于硬件规格估算（快速）
    3. 基础检测 - 仅检测设备存在性（最基础）
    """
    
    def __init__(self, prefer_measured: bool = True, use_cache: bool = True):
        """
        初始化统一检测器
        
        Args:
            prefer_measured: 是否优先使用实测数据（V2）
            use_cache: 是否使用缓存（对V2有效）
        """
        self.prefer_measured = prefer_measured
        self.detector = None
        self.detector_version = None
        
        # 尝试初始化检测器
        self._initialize_detector(use_cache)
        
    def _initialize_detector(self, use_cache: bool):
        """初始化最佳可用的检测器"""
        
        if self.prefer_measured:
            # 优先尝试V2（基于实测）
            try:
                from .device_detector_v2 import DeviceDetectorV2
                self.detector = DeviceDetectorV2(use_cache=use_cache)
                self.detector_version = 'v2'
                logger.info("使用DeviceDetectorV2（基于实测数据）")
                return
            except Exception as e:
                logger.debug(f"DeviceDetectorV2不可用: {e}")
        
        # 回退到V1（基于估算）
        try:
            from .device_detector import DeviceDetector
            self.detector = DeviceDetector()
            self.detector_version = 'v1'
            logger.info("使用DeviceDetectorV1（基于硬件估算）")
            return
        except Exception as e:
            logger.debug(f"DeviceDetectorV1不可用: {e}")
        
        # 最终回退：基础检测
        logger.warning("使用基础设备检测（仅检测存在性）")
        self.detector = None
        self.detector_version = 'basic'
    
    def get_available_devices(self, force_measure: bool = False) -> List[Dict]:
        """
        获取可用设备列表
        
        Args:
            force_measure: 强制重新测量（仅V2支持）
            
        Returns:
            设备信息列表
        """
        if self.detector_version == 'v2':
            # V2支持force_measure参数
            return self.detector.get_available_devices(force_measure=force_measure)
        elif self.detector_version == 'v1':
            # V1不支持force_measure，使用静态方法
            return self.detector.get_available_devices()
        else:
            # 基础检测
            return self._basic_device_detection()
    
    def _basic_device_detection(self) -> List[Dict]:
        """基础设备检测 - 仅检测设备存在性"""
        devices = []
        
        # CPU始终存在
        devices.append({
            'device_id': 'cpu',
            'device_type': 'cpu',
            'name': 'CPU',
            'compute_power': 1.0,  # 基准值
            'bandwidth': 10.0,  # 假设基础带宽
            'latency_ms': 0.001
        })
        
        # 检测GPU
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    devices.append({
                        'device_id': f'cuda:{i}',
                        'device_type': 'cuda',
                        'name': torch.cuda.get_device_name(i),
                        'compute_power': 10.0,  # 假设GPU比CPU快10倍
                        'bandwidth': 100.0,  # 假设较高带宽
                        'latency_ms': 0.02  # PCIe延迟
                    })
        except:
            pass
        
        return devices
    
    def select_optimal_device(self, task_type: str = 'compute') -> str:
        """
        选择最优设备
        
        Args:
            task_type: 任务类型（compute/memory/io/latency）
            
        Returns:
            最优设备ID
        """
        devices = self.get_available_devices()
        
        if self.detector_version == 'v2' and hasattr(self.detector, 'select_optimal_device'):
            # V2有专门的选择方法
            return self.detector.select_optimal_device(devices, task_type)
        else:
            # 简单选择逻辑
            return self._simple_device_selection(devices, task_type)
    
    def _simple_device_selection(self, devices: List[Dict], task_type: str) -> str:
        """简单的设备选择逻辑"""
        if not devices:
            return 'cpu'
        
        if task_type == 'compute':
            # 选择计算能力最强的
            best = max(devices, key=lambda d: d.get('compute_power', 0))
            return best['device_id']
        elif task_type == 'memory':
            # 选择带宽最高的
            best = max(devices, key=lambda d: d.get('bandwidth', 0))
            return best['device_id']
        elif task_type == 'latency':
            # 选择延迟最低的
            best = min(devices, key=lambda d: d.get('latency_ms', float('inf')))
            return best['device_id']
        else:
            # 默认选择第一个GPU或CPU
            gpu_devices = [d for d in devices if d['device_type'] == 'cuda']
            return gpu_devices[0]['device_id'] if gpu_devices else 'cpu'
    
    def get_detector_info(self) -> Dict:
        """获取当前使用的检测器信息"""
        return {
            'version': self.detector_version,
            'capabilities': {
                'measured_performance': self.detector_version == 'v2',
                'estimated_performance': self.detector_version == 'v1',
                'basic_detection': self.detector_version == 'basic',
                'cache_support': self.detector_version == 'v2',
                'force_measure': self.detector_version == 'v2'
            }
        }


# 便捷函数
def get_unified_detector(prefer_measured: bool = True) -> UnifiedDeviceDetector:
    """
    获取统一的设备检测器实例
    
    Args:
        prefer_measured: 是否优先使用实测数据
        
    Returns:
        UnifiedDeviceDetector实例
    """
    return UnifiedDeviceDetector(prefer_measured=prefer_measured)


# 主要区别说明
"""
DeviceDetectorV1 vs DeviceDetectorV2 主要区别：

1. **性能数据来源**：
   - V1: 基于硬件规格估算（CPU核心数、GPU CUDA核心数等）
   - V2: 基于实际测量（运行基准测试获得真实性能）

2. **准确性**：
   - V1: 快速但可能不准确（估算值可能与实际相差较大）
   - V2: 准确但需要时间测量（首次运行需要执行基准测试）

3. **缓存机制**：
   - V1: 无缓存（每次都重新估算）
   - V2: 支持缓存（避免重复测量，提高效率）

4. **硬编码问题**：
   - V1: 包含硬编码的估算公式和基准值
   - V2: 完全基于实测，无硬编码值

5. **使用场景**：
   - V1: 适合快速原型、开发调试
   - V2: 适合生产环境、性能关键应用

自动选择策略：
1. 默认优先使用V2（更准确）
2. V2不可用时自动回退到V1（更快）
3. 都不可用时使用基础检测（最小功能）
"""