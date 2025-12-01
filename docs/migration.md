
# 数据管道迁移计算模块设计文档

## 1. 模块定位与边界
*   **输入**：
    1.  **优化方案 (Optimization Plan)**：由前序优化模块输出，描述了“哪些上下文路径（Context Key）下的节点需要迁移到哪个设备”。
        *   *Example*: `{"Dataset/__getitem__/preprocess": "cuda:0", "cv2.resize": "cuda:0"}`
    2.  **原始数据管道代码**：用户的 Python 脚本/项目。
*   **输出**：
    *   **Patch 后的运行时环境**：关键函数被 Hook，具备自动迁移能力。
*   **职责**：在不修改用户源码的前提下，基于优化方案，动态地将 CPU 计算负载调度至 GPU。

---

## 2. 系统架构图

```mermaid
graph TD
    subgraph Configuration
        Plan[优化方案 (Optimization Plan)]
    end

    subgraph Migration_Module [迁移计算模块]
        Registry[迁移策略库 (Knowledge Base)]
        ContextMgr[稀疏上下文追踪器 (Context Tracker)]
        
        subgraph Patch_Engine [注入与执行引擎]
            Injector[Patch 注入器]
            Wrapper[通用包装器 (Universal Wrapper)]
        end
    end

    subgraph Runtime
        UserCode[用户代码 / 第三方库]
        GPU_Backend[GPU 实现 (Kornia/Torch/etc.)]
    end

    Plan --> Injector
    Injector -->|1. 替换目标函数| UserCode
    
    UserCode -->|2. 调用被拦截函数| Wrapper
    Wrapper -->|3. 更新路径| ContextMgr
    Wrapper -->|4. 查询策略| Registry
    Wrapper -->|5. 执行计算| GPU_Backend
    
    Registry -.->|提供转换逻辑| Wrapper
    ContextMgr -.->|提供当前Key| Wrapper
```

---

## 3. 子模块详细设计

### 3.1 迁移策略库 (Migration Strategy Registry)
**职责**：存储“如何迁移”的静态知识。它是我们在上一轮对话中梳理的核心成果。

*   **数据结构**：
    *   维护一个全局字典 `Map<FunctionPath, MigrationStrategy>`。
*   **核心实体 `MigrationStrategy`**：
    *   `input_processors`: 输入参数转换链（如 `[EnsureTensor('cuda'), PassThrough] `）。
    *   `arg_mapper`: 参数签名修正逻辑（如 `SwapArgs(0, 1)` 用于 cv2.resize）。
    *   `backend`: 目标 GPU 函数（如 `kornia.geometry.resize` 或 `OriginalFunc`）。
    *   `output_processor`: 结果处理（如 `KeepOnDevice` 或 `ToNumpy`）。
*   **扩展性**：支持通过装饰器或配置文件动态注册新的库支持（CV, Audio, NLP）。

### 3.2 稀疏上下文追踪器 (Sparse Context Tracker)
**职责**：维护运行时调用栈，生成 Context Key 以匹配优化方案。

*   **设计要点**：
    *   **线程安全**：必须使用 `threading.local()`，因为 `DataLoader` 可能在多线程环境下工作（虽然 Python GIL 限制了计算并行，但 Context 必须隔离）。
    *   **稀疏性**：只记录**被 Patch 的函数**。
        *   真实调用：`__getitem__` -> `func_A` -> `func_B` (未Patch) -> `cv2.resize` (Patch)。
        *   栈状态：`['__getitem__', 'func_A', 'cv2.resize']`。
        *   Key：`"__getitem__/func_A/cv2.resize"`。
*   **接口**：
    *   `enter(name)`: 入栈。
    *   `exit()`: 出栈。
    *   `current_key()`: 获取当前路径字符串。

### 3.3 Patch 注入器 (Patch Injector)
**职责**：在程序启动阶段，根据优化方案实施 Monkey Patch。

*   **工作流程**：
    1.  **解析计划**：遍历 `Optimization Plan` 中的所有 Key。
    2.  **提取目标**：从 Key 中提取末端函数名（例如从 `.../cv2.resize` 提取 `cv2` 模块和 `resize` 函数）。
    3.  **备份原函数**：将 `original_func` 保存到 `Wrapper` 的闭包或属性中，防止无限递归。
    4.  **实施替换**：`setattr(module, func_name, UniversalWrapper(original_func, ...))`。
    5.  **特殊处理**：针对类方法（如 `transforms.Resize`），需要 Patch 类的 `__call__` 或 `forward`。

### 3.4 通用包装器 (Universal Wrapper) —— **核心执行单元**
**职责**：运行时的调度员。这是最复杂的组件，承载了所有的控制流逻辑。

#### 逻辑流程（伪代码）：

```python
def universal_wrapper(original_func, func_name):
    def wrapper(*args, **kwargs):
        # 1. 上下文入栈
        context_tracker.enter(func_name)
        current_key = context_tracker.current_key()
        
        # 2. 决策：当前节点是否在优化计划中？且目标设备是 GPU？
        target_device = optimization_plan.get(current_key)
        should_migrate = target_device and target_device.startswith('cuda')
        
        result = None
        try:
            if should_migrate:
                # --- GPU 分支 ---
                
                # A. 获取迁移策略 (从 Registry)
                # 如果是用户自定义函数，通常没有注册策略，则使用默认策略（Move Input Only）
                strategy = registry.get(func_name, default=MoveInputStrategy)
                
                # B. 参数预处理 (Input Processor)
                # 将 args/kwargs 转换为 Tensor 并搬运到 target_device
                gpu_args, gpu_kwargs = strategy.process_inputs(args, kwargs, target_device)
                
                # C. 参数映射 (Arg Mapper - 处理签名差异)
                if strategy.arg_mapper:
                    gpu_args, gpu_kwargs = strategy.arg_mapper(gpu_args, gpu_kwargs)
                
                # D. 执行后端 (Backend Execution)
                # 可能是 Kornia 函数，也可能是原函数(依赖 PyTorch Dispatch)
                result = strategy.backend(*gpu_args, **gpu_kwargs)
                
                # E. 结果后处理 (Output Processor)
                result = strategy.process_output(result)
                
            else:
                # --- CPU 分支 (未命中优化计划) ---
                result = original_func(*args, **kwargs)

        except Exception as e:
            # --- 容错降级 (Fallback) ---
            logger.warning(f"Migration failed at {current_key}: {e}. Falling back to CPU.")
            # 必须确保输入数据在 CPU (如果之前被部分搬运了，这里可能需要回退逻辑，
            # 但最简单的是直接用原始 args 调原函数)
            result = original_func(*args, **kwargs)
            
        finally:
            # 3. 上下文出栈
            context_tracker.exit()
            
        # 4. IPC 边界检查 (IPC Guard) - 针对多进程 DataLoader
        if is_worker_process() and is_top_level_node(current_key):
             result = ensure_cpu(result)
             
        return result
    return wrapper
```

---

## 4. 关键特性的实现保障

### 4.1 混合粒度支持
*   **用户自定义函数 (Function-Level)**：
    *   Registry 中无记录。
    *   Wrapper 使用默认策略：`InputProcessors=[EnsureTensor(device)]`, `Backend=OriginalFunc`。
    *   **效果**：仅搬运 Tensor，函数内部的 PyTorch 算子自动在 GPU 执行。
*   **第三方库函数 (Operator-Level)**：
    *   Registry 中有记录（如 `cv2.resize`）。
    *   Wrapper 使用注册策略：`InputProcessors=[ImageHWC2CHW]`, `Backend=kornia.resize`。
    *   **效果**：完全替换实现。

### 4.2 数据驻留 (Data Residency / Lazy Transfer)
*   在 `InputProcessor` 中实现。
*   逻辑：`if isinstance(arg, Tensor) and arg.device == target_device: return arg`。
*   **收益**：如果管道是 `Resize(GPU) -> Rotate(GPU) -> Normalize(GPU)`，只有第一个 `Resize` 会触发 CPU->GPU 拷贝，后续操作直接复用 GPU 数据，消除 PCIe 瓶颈。

### 4.3 多进程 IPC 安全 (IPC Guard)
*   **问题**：PyTorch `DataLoader` 的 `num_workers > 0` 时，子进程返回 GPU Tensor 可能导致 CUDA 初始化错误或 IPC 失败（除非使用 Shared Memory，但复杂）。
*   **机制**：在 Wrapper 的最后（Return 之前），检查：
    1.  当前是否在 Worker 进程中？
    2.  当前是否是 Context 栈的**栈底**（即最外层被 Hook 的函数，通常是 `__getitem__`）？
    3.  如果是，强制执行 `.cpu()`。
*   **妥协**：这确实引入了 D->H 的拷贝，但保证了稳定性。对于极致性能，建议用户设置 `num_workers=0` 并完全依赖 GPU 的吞吐能力。

---

## 5. 模块集成接口

该模块对外暴露简洁的 API：

```python
class PipelineMigrator:
    def __init__(self, optimization_json_path: str):
        self.plan = load_plan(optimization_json_path)
        self.registry = default_registry() # 加载我们梳理好的四大类库
        self.tracker = SparseContextTracker()
        
    def activate(self):
        """启用迁移：执行 Monkey Patch"""
        injector = PatchInjector(self.plan, self.registry, self.tracker)
        injector.apply_all()
        print("🔥 DPMCM Activated: GPU Migration Hooks Installed.")

    def deactivate(self):
        """恢复原始环境"""
        # 恢复 _original_func
        pass
```

**使用示例**：
```python
# 用户代码
migrator = PipelineMigrator("plan.json")
migrator.activate()

# ... 正常运行数据加载 ...
for data in dataloader:
    pass 
```

这个设计文档涵盖了从策略定义到运行时执行的完整链路，既保证了灵活性（Registry），又保证了稳定性（IPC Guard/Fallback）。