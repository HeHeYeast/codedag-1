"""
统一的函数过滤模块
整合所有过滤规则：系统级 + 用户配置级
"""
import re
from typing import List, Optional
from enum import Enum

class FilterLevel(Enum):
    """过滤级别枚举"""
    DISABLED = 0        # 不过滤
    CONSERVATIVE = 1    # 保守过滤
    BALANCED = 2        # 平衡过滤  
    AGGRESSIVE = 3      # 激进过滤

class UnifiedFunctionFilter:
    """统一的函数过滤器
    
    过滤优先级：
    1. SYSTEM级别 - 系统稳定性函数（原SKIP_FUNCTIONS/SKIP_MODULES）
    2. PROTECTION级别 - 重要计算函数保护 
    3. USER级别 - 用户配置的过滤规则
    """
    
    def __init__(self, 
                 filter_level: FilterLevel = FilterLevel.CONSERVATIVE,
                 custom_filters: Optional[List[str]] = None,
                 custom_protections: Optional[List[str]] = None):
        """
        初始化统一过滤器
        
        Args:
            filter_level: 过滤级别
            custom_filters: 自定义过滤规则列表 (支持通配符 *)
            custom_protections: 自定义保护规则列表 (支持通配符 *)
        """
        self.filter_level = filter_level
        self.custom_filters = custom_filters or []
        self.custom_protections = custom_protections or []
        
        # SYSTEM级别 - 系统稳定性过滤（原硬编码规则）
        self.system_functions = {
            '<module>', '<listcomp>', '<genexpr>', '<lambda>',
            # 追踪器内部函数（避免递归）
            'start_tracking', 'stop_tracking', 'info', 'debug', 'warning', 'error',
            'isEnabledFor', '_acquireLock', '_releaseLock', 'disable', 'getEffectiveLevel',
            # PyTorch内部系统函数
            '_lazy_init', 'is_available', 'device_count', 'get_device_properties',
            '_get_device_index', 'get_arch_list', '_check_cubins', 'get_device_capability', 'get_device_name'
        }
        
        self.system_modules = {
            'logging', 'torch.cuda', 'torch._utils',
            'core.memory_profiler', 'core.enhanced_tracer',  # 过滤追踪器自身
            # NumPy 内部噪音 - 彻底屏蔽
            'numpy.core', 'numpy.core.overrides', 'numpy.core.fromnumeric',
            'numpy.core._methods', 'numpy.core.multiarray', 'numpy.lib', 'numpy.lib.arraypad',
            'numpy.core.numeric', 'numpy.core.shape_base',
            'contextlib',
            # 标准库噪音模块 - 这些与业务逻辑无关
            'posixpath', 'genericpath', 'os.path', 'ntpath',
            'threading', 'multiprocessing', 'multiprocessing.process',
            'subprocess', 'signal', 'weakref', 'abc', 'enum', 'selectors',
            'collections', 'importlib', 'zipfile', 'shutil', 'sre_compile',
            'sre_parse', 'copyreg', 'functools', 'operator',
            # 编码和IO相关
            'codecs', 'encodings', 'io', '_io',
            # 路径和文件系统
            'pathlib', 'glob', 'fnmatch',
        }
        
        # PROTECTION级别 - 重要计算函数保护（绝对不能被过滤）
        self.protection_patterns = [
            # PyTorch核心计算函数
            "torch.*interpolate*", "torch.*resize*", 
            "torch.*conv*", "torch.*relu*", "torch.*sigmoid*", "torch.*tanh*",
            "torch.*matmul*", "torch.*add*", "torch.*mul*", "torch.*div*",
            "torch.*mean*", "torch.*sum*", "torch.*view*", "torch.*reshape*",
            "torch.*clamp*", "torch.*stack*", "torch.*cat*", "torch.*squeeze*", "torch.*unsqueeze*",
            # 图像处理函数 - 重要新增
            "torch.nn.functional.*", "F.*interpolate*", "F.*conv2d*",
            "torchvision.*", "transforms.*", "*resize*", "*crop*", 
            "*ToPILImage*", "*ToTensor*", "*Resize*", "*RandomCrop*", "*RandomHorizontalFlip*",
            "*ColorJitter*", "*functional*",
            # PIL图像处理
            "PIL.*", "*Image.*", "*rotate*", "*transform*",
            # 数据加载
            "torch.utils.data.*",
            # 用户函数
            "__main__.*",
            # TensorFlow和图像处理
            "tf.*", "tensorflow.*", "*tf_*", "*image*",
            # 数值计算
            "*numpy.*", "*np.*", "*ndarray*", "*array*",
        ]
        
        # USER级别 - 用户配置的过滤规则
        self.user_filter_patterns = {
            FilterLevel.CONSERVATIVE: [
                # 类型检查函数（用户特别指定的）
                "*.__instancecheck__",
                "*.__subclasscheck__",
                # 类型检查模块
                "typing.*",
                "abc.*",
                "_abc.*",
                "warnings.*",
                "linecache.*",
                # [新增] numpy 内部实现细节 - 只保留顶层调用
                # 注意：这不会屏蔽用户代码中调用的 np.dot，只会屏蔽其内部的纯 Python 辅助函数
                "numpy.core.*",
                "numpy.lib.*",
                "numpy.linalg.*",
                "numpy._core.*",
                "numpy.ma.*",
                # [新增] 其他内部模块
                "re.*",
                "sre_*",
            ],
            FilterLevel.BALANCED: [
                # 包含保守模式
                "*.__instancecheck__", "*.__subclasscheck__",
                "typing.*", "abc.*", "_abc.*", "warnings.*", "linecache.*",
                # 字符串表示函数
                "*.__str__", "*.__repr__", "*.__format__",
                # PyTorch内部
                "torch._C._log_api_usage_once",
            ],
            FilterLevel.AGGRESSIVE: [
                # 包含平衡模式
                "*.__instancecheck__", "*.__subclasscheck__", "*.__subclasshook__",
                "typing.*", "abc.*", "_abc.*", "warnings.*", "linecache.*",
                "*.__str__", "*.__repr__", "*.__format__",
                "torch._C._log_api_usage_once",
                # 线程和属性访问
                "*current_thread", "threading.*",
                "*.__getattribute__", "*.__setattr__",
                "torch._jit_internal.*",
                # PyTorch内部基础设施
                "torch.autograd.profiler.*",
                "torch.jit._builtins.*",
                "torch._ops.*",
            ]
        }
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译所有过滤模式"""
        # 编译保护模式
        self.compiled_protection_patterns = []
        all_protections = self.protection_patterns + self.custom_protections
        for pattern in all_protections:
            regex = self._wildcard_to_regex(pattern)
            self.compiled_protection_patterns.append(re.compile(regex))
        
        # 编译用户过滤模式
        self.compiled_filter_patterns = []
        if self.filter_level != FilterLevel.DISABLED:
            # 获取当前级别的过滤规则
            level_filters = self.user_filter_patterns.get(self.filter_level, [])
            all_filters = level_filters + self.custom_filters
            
            for pattern in all_filters:
                regex = self._wildcard_to_regex(pattern)
                self.compiled_filter_patterns.append(re.compile(regex))
    
    def _wildcard_to_regex(self, pattern: str) -> str:
        """将通配符模式转换为正则表达式"""
        # 转义特殊字符，但保留 * 和 ?
        escaped = re.escape(pattern)
        # 将转义后的 * 和 ? 替换回通配符含义
        regex = escaped.replace(r'\*', '.*').replace(r'\?', '.')
        # 添加开始和结束锚点
        return f"^{regex}$"
    
    def should_filter(self, func_name: str, module_name: str) -> bool:
        """
        判断是否应该过滤此函数
        
        过滤优先级：
        1. SYSTEM - 系统函数优先过滤
        2. PROTECTION - 重要函数绝对不过滤  
        3. USER - 用户配置过滤规则
        
        Args:
            func_name: 函数名
            module_name: 模块名
            
        Returns:
            True: 应该过滤（停止监控但继续执行）
            False: 不过滤（继续监控）
        """
        # 构建完整的函数标识
        if module_name and module_name != 'unknown':
            full_name = f"{module_name}.{func_name}"
        else:
            full_name = func_name
        
        # 1. SYSTEM级别检查 - 系统稳定性函数优先过滤
        if func_name in self.system_functions:
            return True
        if module_name in self.system_modules:
            return True
            
        # 2. PROTECTION级别检查 - 重要函数绝对不过滤
        for pattern in self.compiled_protection_patterns:
            if pattern.match(full_name) or pattern.match(func_name):
                return False
        
        # 3. USER级别检查 - 用户配置的过滤规则
        if self.filter_level == FilterLevel.DISABLED:
            return False
            
        for pattern in self.compiled_filter_patterns:
            if pattern.match(full_name) or pattern.match(func_name):
                return True
        
        return False
    
    def set_filter_level(self, level: FilterLevel):
        """设置过滤级别"""
        self.filter_level = level
        self._compile_patterns()
    
    def add_custom_filter(self, pattern: str):
        """添加自定义过滤规则"""
        if pattern not in self.custom_filters:
            self.custom_filters.append(pattern)
            self._compile_patterns()
    
    def add_custom_protection(self, pattern: str):
        """添加自定义保护规则"""
        if pattern not in self.custom_protections:
            self.custom_protections.append(pattern)
            self._compile_patterns()
    
    def get_stats(self) -> dict:
        """获取过滤统计信息"""
        return {
            'filter_level': self.filter_level.name,
            'system_functions_count': len(self.system_functions),
            'system_modules_count': len(self.system_modules),
            'protection_patterns_count': len(self.compiled_protection_patterns),
            'filter_patterns_count': len(self.compiled_filter_patterns),
            'custom_filters': len(self.custom_filters),
            'custom_protections': len(self.custom_protections)
        }

# 向后兼容 - 重新导出原有的类名和常量
SimpleFunctionFilter = UnifiedFunctionFilter
CONSERVATIVE_FILTER = UnifiedFunctionFilter(FilterLevel.CONSERVATIVE)
BALANCED_FILTER = UnifiedFunctionFilter(FilterLevel.BALANCED)
AGGRESSIVE_FILTER = UnifiedFunctionFilter(FilterLevel.AGGRESSIVE)
DISABLED_FILTER = UnifiedFunctionFilter(FilterLevel.DISABLED)