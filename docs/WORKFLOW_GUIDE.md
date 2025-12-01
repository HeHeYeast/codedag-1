# CodeDAG 工作流程指南

本文档详细描述 CodeDAG 各模块的工作流程，并提供参考代码。

---

## 1. 整体架构流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户代码                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    阶段1: 代码追踪                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │EnhancedTracer│──▶│ DAGBuilder  │──▶│ExecutionDAG │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│         │                                                       │
│         ├── MemoryProfiler                                      │
│         └── PerformanceMonitor                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    阶段2: 图优化                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │GraphCoarsening│─▶│KWayPartitioner│─▶│IterativeOpt │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│    粗化后DAG          分区结果         优化后分区               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    阶段3: 迁移执行                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │OptimizerMgr │──▶│MigrationPlan│──▶│PipelineMigrator│         │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                             │                   │
│                                             ▼                   │
│                                      PatchInjector              │
│                                             │                   │
│                                             ▼                   │
│                                      GPU 执行                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段1: 代码追踪流程

### 2.1 追踪流程详解

```
用户代码执行
     │
     ▼
EnhancedTracer.tracing_context()
     │
     ├─▶ sys.settrace() 设置追踪钩子
     │
     ▼
trace_callback 被调用 (每个函数调用/返回)
     │
     ├─▶ FunctionFilter.should_trace() 判断是否追踪
     │
     ├─▶ 函数入口: 记录开始时间、创建函数节点
     │
     ├─▶ 参数分析: 创建变量节点、添加 uses 边
     │
     ├─▶ MemoryProfiler.take_snapshot() 内存快照
     │
     ├─▶ 函数返回: 计算执行时间、创建返回值节点
     │
     └─▶ DAGBuilder.add_edge() 添加 produces 边
```

### 2.2 参考代码

```python
# core/enhanced_tracer.py 关键代码片段

class EnhancedTracer:
    def __init__(self, max_depth=10, track_memory=True, filter_config=None):
        self.max_depth = max_depth
        self.track_memory = track_memory
        self.dag_builder = DAGBuilder()
        self.memory_profiler = MemoryProfiler() if track_memory else None
        self.function_filter = FunctionFilter(filter_config)
        self._call_stack = []

    @contextmanager
    def tracing_context(self):
        """追踪上下文"""
        old_trace = sys.gettrace()
        sys.settrace(self._trace_callback)
        try:
            yield
        finally:
            sys.settrace(old_trace)

    def _trace_callback(self, frame, event, arg):
        """追踪回调函数"""
        # 深度检查
        if len(self._call_stack) >= self.max_depth:
            return None

        func_name = frame.f_code.co_name
        module = frame.f_globals.get('__name__', '')

        # 过滤检查
        if not self.function_filter.should_trace(func_name, module):
            return None

        if event == 'call':
            self._handle_call(frame, func_name)
        elif event == 'return':
            self._handle_return(frame, arg)

        return self._trace_callback

    def _handle_call(self, frame, func_name):
        """处理函数调用"""
        # 记录开始时间
        start_time = time.perf_counter()

        # 创建函数节点
        node_id = self.dag_builder.add_function_node(
            name=func_name,
            context=self._get_context_key()
        )

        # 分析参数，创建 uses 边
        local_vars = frame.f_locals
        for var_name, var_value in local_vars.items():
            var_node_id = self.dag_builder.add_variable_node(var_name)
            self.dag_builder.add_edge(var_node_id, node_id, 'uses')

        # 压栈
        self._call_stack.append({
            'node_id': node_id,
            'start_time': start_time,
            'func_name': func_name
        })

        # 内存快照
        if self.memory_profiler:
            self.memory_profiler.take_snapshot()

    def _handle_return(self, frame, return_value):
        """处理函数返回"""
        if not self._call_stack:
            return

        call_info = self._call_stack.pop()

        # 计算执行时间
        execution_time = time.perf_counter() - call_info['start_time']

        # 更新节点性能数据
        node = self.dag_builder.dag.nodes[call_info['node_id']]
        node.performance['execution_time'] = execution_time

        # 创建返回值节点
        if return_value is not None:
            ret_node_id = self.dag_builder.add_variable_node(
                f"{call_info['func_name']}_return"
            )
            self.dag_builder.add_edge(
                call_info['node_id'],
                ret_node_id,
                'produces'
            )
```

### 2.3 DAGBuilder 工作流程

```
添加节点
    │
    ├─▶ add_function_node(name, context)
    │       │
    │       ├─▶ 创建 DAGNode(type="function_call")
    │       ├─▶ node_counter += 1
    │       └─▶ dag.nodes[id] = node
    │
    ├─▶ add_variable_node(name, version)
    │       │
    │       ├─▶ 检查同名变量是否存在
    │       ├─▶ 版本号递增 (SSA形式)
    │       └─▶ 创建 DAGNode(type="variable")
    │
    └─▶ add_edge(source, target, type)
            │
            └─▶ dag.edges.append((source, target, type))
```

---

## 3. 阶段2: 图优化流程

### 3.1 OptimizerManager 完整流程

```
输入: ExecutionDAG
        │
        ▼
┌───────────────────────────────────────┐
│        GraphCoarsening.coarsen()       │
│   识别重复子图模式，合并同构子图          │
│                                        │
│   1. _identify_anchors()               │
│      └─ 找到重复出现的函数调用            │
│                                        │
│   2. _extract_subgraphs()              │
│      └─ BFS 从锚点提取关联子图           │
│                                        │
│   3. _group_by_hash()                  │
│      └─ 计算结构Hash，分组同构子图        │
│                                        │
│   4. _merge_subgraphs()                │
│      └─ 合并同构子图，聚合性能数据        │
└───────────────────────────────────────┘
        │
        ▼ 粗化后 DAG
┌───────────────────────────────────────┐
│       KWayPartitioner.partition()      │
│   将 DAG 划分为 K 个分区                 │
│                                        │
│   1. _preprocess()                     │
│      └─ 计算图直径、节点权重             │
│                                        │
│   2. _select_bottlenecks()             │
│      └─ 选择 K 个瓶颈节点作为中心        │
│                                        │
│   3. _initialize_partitions()          │
│      └─ 创建初始分区                    │
│                                        │
│   4. _greedy_assignment()              │
│      └─ 贪心迭代分配节点到分区           │
│                                        │
│   5. _handle_isolated_nodes()          │
│      └─ 处理未分配的孤立节点             │
└───────────────────────────────────────┘
        │
        ▼ 初始分区
┌───────────────────────────────────────┐
│      IterativeOptimizer.optimize()     │
│   迭代优化分区，最小化目标函数            │
│                                        │
│   1. _initialize()                     │
│      └─ 计算节点权重、关键路径、边界节点   │
│                                        │
│   2. _build_initial_heap()             │
│      └─ 计算所有可能迁移的收益           │
│                                        │
│   3. 主循环:                            │
│      ├─ 取最大收益迁移                   │
│      ├─ 验证迁移有效性                   │
│      ├─ 执行迁移                        │
│      ├─ 更新受影响节点收益               │
│      └─ 检查终止条件                    │
└───────────────────────────────────────┘
        │
        ▼ 优化后分区
```

### 3.2 GraphCoarsening 参考代码

```python
# optimizer/graph_coarsening.py 关键代码片段

class GraphCoarsening:
    def coarsen(self, dag: ExecutionDAG) -> ExecutionDAG:
        """执行图粗化"""
        logger.info("开始图粗化聚类...")

        # 1. 识别锚点
        self._identify_anchors(dag)
        if not self.anchors:
            logger.info("未发现重复锚点，跳过聚类")
            return deepcopy(dag)

        # 2. 提取子图
        self._extract_subgraphs(dag)

        # 3. Hash分组
        self._group_by_hash(dag)

        # 4. 合并子图
        coarsened_dag = self._merge_subgraphs(dag)

        logger.info(f"粗化完成: {len(dag.nodes)} -> {len(coarsened_dag.nodes)} 节点")
        return coarsened_dag

    def _identify_anchors(self, dag: ExecutionDAG):
        """识别重复出现的锚点节点"""
        name_groups = defaultdict(list)

        for node_id, node in dag.nodes.items():
            if node.node_type == "function_call":
                # 归一化名称 (去除 #数字 后缀)
                normalized = re.sub(r'#\d+$', '', node.name)
                name_groups[normalized].append(node_id)

        # 只保留出现 >= 2 次的
        for name, node_ids in name_groups.items():
            if len(node_ids) >= 2:
                self.anchors[name] = node_ids

    def _compute_subgraph_hash(self, dag: ExecutionDAG, subgraph: Subgraph) -> str:
        """计算子图结构Hash"""
        signature = []
        queue = deque([(subgraph.anchor_id, 0)])
        visited = {subgraph.anchor_id}

        while queue:
            node_id, depth = queue.popleft()
            node = dag.nodes[node_id]

            # 记录结构特征
            norm_name = self._normalize_name(node.name)
            signature.append(f"{depth}:{node.node_type}:{norm_name}")

            # BFS遍历子节点
            for src, tgt, etype in subgraph.internal_edges:
                if src == node_id and tgt not in visited:
                    visited.add(tgt)
                    queue.append((tgt, depth + 1))

        # 生成Hash
        sig_str = "|".join(signature)
        return hashlib.sha256(sig_str.encode()).hexdigest()
```

### 3.3 KWayPartitioner 参考代码

```python
# optimizer/kway_partitioner.py 关键代码片段

class KWayPartitioner:
    def partition(self, dag: ExecutionDAG) -> List[Partition]:
        """执行 K-way 划分"""
        logger.info(f"开始K-way划分 (k={self.k})...")

        # 阶段0: 预处理
        self._preprocess(dag)

        # 阶段1: 选择瓶颈节点
        bottlenecks = self._select_bottlenecks(dag)

        # 阶段2: 初始化分区
        partitions, candidates, assigned = self._initialize_partitions(dag, bottlenecks)

        # 阶段3: 贪心分配
        self._greedy_assignment(dag, partitions, candidates, assigned)

        # 阶段4: 孤立节点处理
        self._handle_isolated_nodes(dag, partitions, assigned)

        return partitions

    def _select_bottlenecks(self, dag: ExecutionDAG) -> List[int]:
        """选择 K 个瓶颈节点"""
        selected = []
        total_weight = sum(self.node_weights.values())

        for i in range(min(self.k, len(dag.nodes))):
            best_node = None
            best_score = -float('inf')

            for node_id in dag.nodes:
                if node_id in selected:
                    continue

                # 计算综合得分
                dist_score = self._get_distance_score(node_id, selected)
                perf_score = self.node_weights[node_id] / total_weight

                score = (self.lambda_weight * dist_score +
                        (1 - self.lambda_weight) * perf_score)

                if score > best_score:
                    best_score = score
                    best_node = node_id

            selected.append(best_node)

        return selected

    def _compute_assignment_score(self, node_id: int, partition: Partition,
                                   all_partitions: List[Partition]) -> float:
        """计算节点分配得分"""
        # 性能得分
        perf_score = self.node_weights[node_id] / self.max_weight

        # 距离惩罚
        dist = self.center_distances[node_id][partition.center]
        dist_penalty = dist / self.diameter if self.diameter > 0 else 0

        # 负载惩罚
        avg_load = sum(p.total_compute for p in all_partitions) / len(all_partitions)
        would_be_load = partition.total_compute + self.node_weights[node_id]
        load_ratio = would_be_load / avg_load if avg_load > 0 else 1.0
        load_penalty = max(0, (load_ratio - 1.0) ** 2)

        # 综合得分
        return self.alpha * perf_score - self.beta * dist_penalty - self.gamma * load_penalty
```

### 3.4 IterativeOptimizer 参考代码

```python
# optimizer/iterative_optimizer.py 关键代码片段

class IterativeOptimizer:
    def optimize(self, partitions: List[Partition], dag: ExecutionDAG) -> List[Partition]:
        """执行迭代优化"""
        # 初始化
        self._initialize(partitions, dag)

        # 建立初始 GainHeap
        gain_heap = self._build_initial_heap(partitions, dag)

        # 主循环
        no_improvement_count = 0
        iteration = 0

        while gain_heap and iteration < self.max_iterations:
            iteration += 1

            # 取最大收益迁移
            neg_gain, node_id, from_pid, to_pid = heapq.heappop(gain_heap)
            gain = -neg_gain

            # 验证有效性
            if not self._is_valid_move(node_id, from_pid, to_pid, partitions, dag):
                continue

            # 重新计算收益
            current_gain = self._compute_gain(node_id, from_pid, to_pid, partitions, dag)
            if current_gain <= 0:
                continue

            # 执行迁移
            self._execute_move(node_id, from_pid, to_pid, partitions, dag)

            # 更新受影响节点
            affected = self._get_affected_nodes(node_id, from_pid, to_pid, partitions, dag)
            for affected_node in affected:
                self._update_node_gains(affected_node, partitions, dag, gain_heap)

            # 检查改善
            current_obj = self._compute_objective(partitions)
            if current_obj < self.stats['best_objective']:
                self.stats['best_objective'] = current_obj
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # 终止条件
            if no_improvement_count >= self.no_improvement_threshold:
                break

        return partitions

    def _compute_objective(self, partitions: List[Partition]) -> float:
        """计算目标函数 J = α·CV(loads) + β·normalized_max_cp"""
        loads = [p.total_compute for p in partitions]

        # 负载变异系数
        mean_load = sum(loads) / len(loads)
        std_load = (sum((l - mean_load)**2 for l in loads) / len(loads)) ** 0.5
        cv_load = std_load / mean_load if mean_load > 0 else 0

        # 归一化最大关键路径
        max_cp = max(self.cp_lengths.values())
        mean_cp = sum(self.cp_lengths.values()) / len(self.cp_lengths)
        normalized_cp = max_cp / mean_cp if mean_cp > 0 else 0

        return self.alpha * cv_load + self.beta * normalized_cp
```

---

## 4. 阶段3: 迁移执行流程

### 4.1 迁移流程详解

```
迁移计划 (context_device_map)
        │
        ▼
PipelineMigrator.apply()
        │
        ▼
PatchInjector.apply_all()
        │
        ├─▶ 遍历所有需要 Patch 的函数
        │
        └─▶ 对每个函数:
                │
                ├─▶ resolve_object_path() 解析函数对象
                │
                ├─▶ 获取 OpSpec 策略
                │
                ├─▶ 创建 UniversalWrapper
                │
                └─▶ setattr() 替换原函数
                        │
                        ▼
              原函数被 UniversalWrapper 包装
```

### 4.2 UniversalWrapper 执行流程

```
被 Patch 的函数被调用
        │
        ▼
UniversalWrapper.__call__()
        │
        ├─▶ context_tracker.enter() 上下文入栈
        │
        ├─▶ _get_target_device() 获取目标设备
        │       │
        │       ├─▶ 精确匹配 context_device_map
        │       ├─▶ 后缀匹配
        │       └─▶ 函数名匹配
        │
        ├─▶ _should_migrate() 判断是否迁移
        │       │
        │       ├─▶ 检查目标设备是否为 CUDA
        │       ├─▶ 检查策略类型
        │       └─▶ 检查 fallback 条件
        │
        └─▶ 根据策略执行:
                │
                ├─▶ MoveOnly: _execute_move_only()
                │       └─ 只搬运输入到 GPU
                │
                ├─▶ StandardOp: _execute_standard_op()
                │       │
                │       ├─ 1. _transform_args() 转换输入
                │       ├─ 2. _apply_value_maps() 值映射
                │       ├─ 3. _apply_renames() 参数改名
                │       ├─ 4. _inject_kwargs() 注入参数
                │       ├─ 5. _apply_structural_ops() 结构调整
                │       ├─ 6. _get_backend() 获取后端函数
                │       ├─ 7. backend() 执行
                │       └─ 8. _process_output() 处理输出
                │
                ├─▶ FactoryOp: _execute_factory_op()
                │       │
                │       ├─ 分离数据参数和配置参数
                │       ├─ 获取/创建 Transform 实例
                │       └─ 执行 instance(data)
                │
                └─▶ 异常: 降级到原函数
```

### 4.3 迁移模块参考代码

```python
# migration/core.py 关键代码片段

class UniversalWrapper:
    def __call__(self, *args, **kwargs):
        """拦截函数调用"""
        # 上下文入栈
        self.context_tracker.enter(self.func_path)
        current_key = self.context_tracker.current_key()
        normalized_key = normalize_context_key(current_key)

        try:
            # 决策：是否迁移
            target_device = self._get_target_device(normalized_key)
            should_migrate = self._should_migrate(target_device)

            if should_migrate:
                # GPU 分支
                result = self._execute_migrated(args, kwargs, target_device)
            else:
                # CPU 分支
                result = self.original_func(*args, **kwargs)

        except Exception as e:
            # 异常降级
            logger.warning(f"Migration failed: {e}. Falling back to CPU.")
            result = self.original_func(*args, **kwargs)

        finally:
            self.context_tracker.exit()

        return result

    def _execute_standard_op(self, args, kwargs, target_device):
        """StandardOp 策略执行"""
        # 1. 转换输入参数
        new_args = self._transform_args(args, target_device)
        new_kwargs = self._transform_kwargs(kwargs, target_device)

        # 2. 参数值映射
        new_args, new_kwargs = self._apply_value_maps(new_args, new_kwargs)

        # 3. 参数改名
        new_kwargs = self._apply_renames(new_kwargs)

        # 4. 注入默认参数
        new_kwargs = self._inject_kwargs(new_kwargs, target_device)

        # 5. 结构性调整
        new_args, new_kwargs = self._apply_structural_ops(new_args, new_kwargs)

        # 6. 获取后端函数
        backend = self._get_backend()

        # 7. 执行
        result = backend(*new_args, **new_kwargs)

        # 8. 处理输出
        return self._process_output(result)

    def _transform_args(self, args, target_device):
        """转换位置参数"""
        new_args = list(args)

        for idx, rule_name in self.spec.args_trans.items():
            if idx < len(new_args):
                processor = self.processors.get_input_processor(rule_name)
                if processor:
                    new_args[idx] = processor(new_args[idx], target_device)

        return tuple(new_args)


# migration/api.py 关键代码片段

class PipelineMigrator:
    def apply(self) -> Dict[str, int]:
        """应用迁移 Patch"""
        if self._is_applied:
            return {'patched': 0, 'failed': 0, 'skipped': 0}

        self._injector = PatchInjector(
            context_device_map=self.context_device_map,
            registry=self.registry,
            processors=self.processors,
            context_tracker=self.context_tracker,
            default_device=self.default_device,
        )

        stats = self._injector.apply_all()
        self._is_applied = True

        return stats

    def restore(self) -> int:
        """恢复原始函数"""
        if not self._is_applied:
            return 0

        count = self._injector.restore_all()
        self._is_applied = False
        self.context_tracker.reset()

        return count
```

---

## 5. 可视化流程

### 5.1 可视化生成流程

```
ExecutionDAG
     │
     ▼
GraphvizDataflowVisualizer
     │
     ├─▶ _organize_nodes_by_layer() 拓扑分层
     │       │
     │       ├─▶ 分析边依赖关系
     │       └─▶ 拓扑排序分配层级
     │
     ├─▶ _create_graphviz_graph() 创建图
     │       │
     │       ├─▶ 设置图属性 (布局、间距等)
     │       ├─▶ 创建节点 (HTML标签、样式)
     │       ├─▶ 使用 subgraph 约束同层节点
     │       └─▶ 创建边 (样式、标签)
     │
     └─▶ dot_graph.render() 生成SVG
```

### 5.2 参考代码

```python
# visualization/graphviz_visualizer.py 关键代码片段

class GraphvizDataflowVisualizer:
    def generate_dataflow_svg(self, nodes, edges, output_path, title):
        """生成数据流图 SVG"""
        # 创建 Graphviz 图
        dot_graph = self._create_graphviz_graph(nodes, edges, title)

        # 生成 SVG
        svg_path = output_path.replace('.svg', '')
        dot_graph.render(svg_path, format='svg', cleanup=True)

        return True

    def _create_graphviz_graph(self, nodes, edges, title):
        """创建 Graphviz 图对象"""
        import graphviz

        dot = graphviz.Digraph(comment=title)

        # 设置图属性
        dot.attr(rankdir='TB')      # 自上而下布局
        dot.attr(splines='spline')  # 样条曲线
        dot.attr(nodesep='0.8')     # 节点间距
        dot.attr(ranksep='1.2')     # 层间距

        # 分层组织节点
        layers = self._organize_nodes_by_layer(nodes, edges)

        # 创建节点
        for node in nodes:
            node_id = str(node.node_id)
            style = self.node_styles.get(node.node_type, self.node_styles['variable'])
            label = self._create_node_label_html(node)

            dot.node(
                node_id,
                label=label,
                shape=style.shape,
                style=style.style,
                fillcolor=style.fillcolor
            )

        # 同层约束
        for layer_num, layer_nodes in layers.items():
            with dot.subgraph() as subgraph:
                subgraph.attr(rank='same')
                for node in layer_nodes:
                    subgraph.node(str(node.node_id))

        # 创建边
        for from_id, to_id, edge_type in edges:
            edge_style = self.edge_styles.get(edge_type, {'color': 'gray'})
            dot.edge(str(from_id), str(to_id), label=edge_type, **edge_style)

        return dot
```

---

## 6. 完整示例

### 6.1 数据管道优化完整示例

```python
import json
from pathlib import Path

from core.enhanced_tracer import EnhancedTracer
from optimizer import OptimizerManager
from migration import PipelineMigrator

# ========== 定义数据处理管道 ==========
def image_pipeline(images):
    """图像处理管道"""
    results = []
    for img in images:
        # 这些操作会被追踪
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = resized / 255.0
        results.append(normalized)
    return results

# ========== 阶段1: 追踪 ==========
tracer = EnhancedTracer(max_depth=8, track_memory=True)

# 准备测试数据
import numpy as np
test_images = [np.random.rand(100, 100, 3).astype(np.float32) for _ in range(10)]

# 追踪执行
with tracer.tracing_context():
    result = image_pipeline(test_images)

# 导出数据流图
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

dataflow = tracer.export_dataflow_graph()
with open(output_dir / "dataflow.json", "w") as f:
    json.dump(dataflow, f, indent=2)

tracer.export_visualization(str(output_dir / "dataflow.svg"))

# ========== 阶段2: 优化 ==========
dag = tracer.dag_builder.dag

optimizer = OptimizerManager(
    k=2,                    # 2个分区
    coarsen_max_depth=5,
    max_iterations=100
)

# 执行优化并导出各阶段结果
result = optimizer.optimize(dag, export_dir=str(output_dir))

stats = result['statistics']
print(f"粗化: {stats['coarsening']}")
print(f"划分: {stats['partitioning']}")
print(f"优化: {stats['iteration']}")

# 生成迁移计划
migration_plan = optimizer.get_migration_plan(
    partitions=result['optimized_partitions'],
    dag=result['coarsened_dag'],
    gpu_count=1
)

with open(output_dir / "migration_plan.json", "w") as f:
    json.dump(migration_plan, f, indent=2)

# ========== 阶段3: 迁移执行 ==========
migrator = PipelineMigrator.from_plan(migration_plan)

# 应用迁移
stats = migrator.apply()
print(f"迁移状态: {stats}")

# 执行迁移后的管道 (自动使用GPU)
with migrator:
    gpu_result = image_pipeline(test_images)

# 验证结果
print(f"GPU 结果数量: {len(gpu_result)}")
print(f"结果设备: {gpu_result[0].device if hasattr(gpu_result[0], 'device') else 'numpy'}")
```

### 6.2 PyTorch DataLoader 优化示例

```python
from torch.utils.data import Dataset, DataLoader
from core.enhanced_tracer import EnhancedTracer
from optimizer import OptimizerManager
from migration import PipelineMigrator

class ImageDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 模拟图像加载和处理
        image = self.load_image(idx)
        image = self.preprocess(image)
        return image

    def load_image(self, idx):
        import numpy as np
        return np.random.rand(256, 256, 3).astype(np.float32)

    def preprocess(self, image):
        import cv2
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

# 追踪
tracer = EnhancedTracer(max_depth=10)
dataset = ImageDataset(size=16)
dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

with tracer.tracing_context():
    for batch in dataloader:
        break  # 只追踪一个 batch

# 优化
dag = tracer.dag_builder.dag
optimizer = OptimizerManager(k=2)
result = optimizer.optimize(dag)

# 迁移
plan = optimizer.get_migration_plan(result['optimized_partitions'], result['coarsened_dag'], gpu_count=1)
migrator = PipelineMigrator.from_plan(plan)

# 使用迁移后的数据加载
with migrator:
    for batch in dataloader:
        # batch 的预处理现在在 GPU 上执行
        print(f"Batch shape: {batch.shape}")
        break
```

---

## 7. 调试和问题排查

### 7.1 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 或针对特定模块
logging.getLogger('optimizer').setLevel(logging.DEBUG)
logging.getLogger('migration').setLevel(logging.DEBUG)
```

### 7.2 检查追踪结果

```python
# 查看追踪到的节点
for node_id, node in tracer.dag_builder.dag.nodes.items():
    print(f"Node {node_id}: {node.name} ({node.node_type})")
    print(f"  Performance: {node.performance}")

# 查看边
for src, tgt, etype in tracer.dag_builder.dag.edges:
    print(f"Edge: {src} --{etype}--> {tgt}")
```

### 7.3 检查优化结果

```python
# 查看分区
for partition in result['optimized_partitions']:
    print(f"Partition {partition.id}:")
    print(f"  Center: {partition.center}")
    print(f"  Nodes: {len(partition.nodes)}")
    print(f"  Load: {partition.total_compute}")

# 查看统计
print(f"Load balance: {result['statistics']['partitioning']['load_balance']}")
```

### 7.4 检查迁移状态

```python
# 列出支持的函数
from migration import list_supported_functions
supported = list_supported_functions()
for strategy, funcs in supported.items():
    print(f"{strategy}: {len(funcs)} functions")

# 检查迁移器状态
print(f"Applied: {migrator.is_applied()}")
print(f"Stats: {migrator.get_stats()}")
```
