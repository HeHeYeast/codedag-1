核心思想：贪心迭代 + 局部增量更新
一句话概括：维护一个"收益堆"，每次取出收益最大的节点迁移执行，然后只更新受影响节点的收益。

一、目标函数（极简版）
J = α · CV(loads) + β · max(CPs)

其中：
- CV(loads) = std(load_1,...,load_K) / mean(load_1,...,load_K)
- max(CPs) = max(CP_1,...,CP_K)
- α, β: 权重，推荐 α=0.4, β=0.6
为什么这样设计？

负载用CV：相对度量，更合理
关键路径用max：最大瓶颈决定整体性能
不需要std(CPs)：max已经抓住了主要矛盾


二、核心数据结构（只维护最少必要信息）
2.1 每个分区维护3个东西
Partition {
    total_load: float           # 总负载
    CP_length: float            # 关键路径长度
    boundary_nodes: Set[Node]   # 边界节点集合
}
就这3个！其他的都是按需临时计算。
2.2 全局维护1个优先队列
GainHeap: 优先队列，存储 (gain, node, from_part, to_part)
按gain从大到小排序，这样每次取堆顶就是当前最优迁移。

三、主算法流程（5个步骤）
步骤0：初始化
1. 计算每个分区的 total_load 和 CP_length
2. 识别所有边界节点
3. 对每个边界节点v：
   - 计算它迁移到相邻分区的收益
   - 若收益 > 0，加入 GainHeap
步骤1-5：主循环
while GainHeap非空 and 未达迭代上限:
    
    【步骤1】取出收益最大的迁移
    (gain, v, i, j) = GainHeap.pop()
    
    【步骤2】检查是否有效
    if 不满足约束条件:
        continue
    
    【步骤3】执行迁移
    - 将v从分区i移到分区j
    - 更新 i.total_load 和 j.total_load
    - 更新 i.boundary_nodes 和 j.boundary_nodes
    
    【步骤4】局部更新关键路径
    - 重算分区i的CP_length（因为少了v）
    - 重算分区j的CP_length（因为多了v）
    
    【步骤5】更新受影响节点的收益
    affected = {v的邻居} ∪ {i的边界节点} ∪ {j的边界节点}
    for u in affected:
        - 移除u的旧收益记录
        - 重新计算u的迁移收益
        - 若收益 > 0，加入 GainHeap
终止条件：堆为空 或 连续20轮无改善 或 达到最大迭代次数

四、关键子问题：如何高效计算收益？
4.1 收益的定义
对于节点v从分区i迁移到j：
Gain(v, i→j) = J_before - J_after
             = [α·CV_old + β·max_CP_old] - [α·CV_new + β·max_CP_new]
4.2 分解计算
负载部分（简单）：
old_loads = [load_1, ..., load_K]
new_loads = old_loads.copy()
new_loads[i] -= cost(v)
new_loads[j] += cost(v)

ΔJ_load = α · [CV(old_loads) - CV(new_loads)]
复杂度：O(K)，K很小（GPU数量），可以接受。
关键路径部分（需要技巧）：
old_CPs = [CP_1, ..., CP_K]
new_CPs = old_CPs.copy()
new_CPs[i] = estimate_CP_after_remove(i, v)  # 估算移除v后的CP
new_CPs[j] = estimate_CP_after_add(j, v)     # 估算添加v后的CP

ΔJ_cp = β · [max(old_CPs) - max(new_CPs)]
最终收益：
Gain = ΔJ_load + ΔJ_cp
4.3 关键路径的快速估算
这是核心技巧！不需要完整重算DAG的关键路径。
估算策略：
情况A：从分区i移除节点v
方法1：快速上界估算（最简单）
如果 v 不在i的关键路径上：
    new_CP_i = old_CP_i  # 不变
否则：
    new_CP_i = 重算i的关键路径（局部BFS）
怎么判断v是否在关键路径上？

预先用O(V+E)算一遍所有分区的关键路径，标记关键路径上的节点
在迁移时检查标记即可

方法2：保守估算（避免完全重算）
如果 v 不在关键路径上：
    new_CP_i ≈ old_CP_i
否则：
    new_CP_i ≈ old_CP_i - cost(v)  # 保守估计（可能偏小）
情况B：向分区j添加节点v
new_CP_j ≈ max(old_CP_j, path_through_v)

其中 path_through_v = 
    max{dist_to_source[u] for u in v.predecessors_in_j} + 
    cost(v) + 
    max{dist_to_sink[w] for w in v.successors_in_j}
关键观察：

添加v可能创建新的更长路径
只需要检查"通过v的路径"是否更长
v的前驱后继的距离可以预先计算或快速查询


五、两个关键优化
优化1：周期性精确重算
每隔N轮迭代（例如N=50），对所有分区完整重算一次关键路径
目的：纠正累积的估算误差
成本：O(V+E)，但不频繁
优化2：懒惰删除
当节点u的收益失效时，不立即从GainHeap删除
而是在取出时检查：
    if 当前状态下重新计算的gain与堆中存储的gain不一致:
        continue  # 跳过这个过期的记录
这样避免频繁的堆删除操作（O(B)变成O(log B)）。

六、完整流程示意图
初始化
├── 计算每个分区的 load, CP, boundary
└── 建立初始的 GainHeap
    
主循环（迭代100-200轮）
│
├── [取] 从GainHeap取堆顶 (gain, v, i, j)
│
├── [查] 检查约束（负载比、依赖关系）
│
├── [移] 执行迁移：v从i到j
│   ├── 更新 loads[i], loads[j]
│   └── 更新 boundary_nodes
│
├── [算] 更新关键路径
│   ├── 重算 CP[i]（如果v在关键路径上）
│   └── 重算 CP[j]（检查是否有新的更长路径）
│
└── [更新] 更新受影响节点的收益
    ├── 找出 affected = v的邻居 + i的边界 + j的边界
    ├── 对每个u in affected:
    │   ├── 计算 gain(u, current_part → adjacent_parts)
    │   └── 加入 GainHeap
    └──
    
每50轮：完整重算所有CP（纠正误差）

终止
└── 堆为空 or 连续20轮无改善 or 达到最大迭代

七、数据维护总结
全局只需维护：

K个分区对象，每个有3个字段：{load, CP, boundaries}
1个优先队列：GainHeap
1个标记数组：is_on_critical_path[v]（可选，用于加速）

不需要维护：

❌ 不需要完整的拓扑排序（按需计算）
❌ 不需要每个节点的dist_from_source/dist_to_sink（按需计算）
❌ 不需要GainCache（用懒惰删除代替）
❌ 不需要紧凑性度量


八、复杂度分析（简化版）
初始化：O(V+E)，算一遍所有分区的CP
单次迭代：

取堆顶：O(log B)
执行迁移：O(1)
更新CP：O(affected_nodes)，通常很小
更新收益：O(affected_nodes · K)

总体：O(V+E + T·affected·K)，T是迭代次数
实际中，affected通常只有几十个节点，K是GPU数（~4-8），所以非常快。

九、伪代码（精简版）
pythondef iterative_optimization(partitions):
    # 初始化
    for p in partitions:
        p.load = sum(cost(v) for v in p.nodes)
        p.CP = compute_critical_path(p)  # 完整计算一次
        p.boundaries = find_boundary_nodes(p)
    
    heap = []
    for v in all_boundary_nodes():
        for j in adjacent_partitions(v):
            gain = compute_gain(v, v.partition, j)
            if gain > 0:
                heappush(heap, (-gain, v, v.partition, j))
    
    # 主循环
    iteration = 0
    while heap and iteration < MAX_ITER:
        gain, v, i, j = heappop(heap)
        gain = -gain
        
        # 检查约束
        if not is_valid_move(v, i, j):
            continue
        
        # 执行迁移
        move_node(v, i, j)
        partitions[i].load -= cost(v)
        partitions[j].load += cost(v)
        
        # 更新关键路径
        if v in partitions[i].critical_path_nodes:
            partitions[i].CP = recompute_CP(partitions[i])
        partitions[j].CP = recompute_CP(partitions[j])
        
        # 更新受影响节点
        affected = get_affected_nodes(v, i, j)
        for u in affected:
            for target in adjacent_partitions(u):
                new_gain = compute_gain(u, u.partition, target)
                if new_gain > 0:
                    heappush(heap, (-new_gain, u, u.partition, target))
        
        # 周期性重算
        if iteration % 50 == 0:
            for p in partitions:
                p.CP = compute_critical_path(p)
        
        iteration += 1
    
    return partitions