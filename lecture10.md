# Stanford CS336n | 第10讲：推理


本次课程我们将深入探讨语言模型的**推理（Inference）** 过程。主要内容包括：

1.  **理解推理的工作负载**:
    * 推理的应用场景 (`landscape`)
    * 回顾 Transformer 模型 (`review transformer`)
    * 回顾计算强度 (`review of arithmetic intensity`)
    * 分析推理过程的计算强度 (`arithmetic intensity of inference`)
    * 吞吐量与延迟的权衡 (`throughput and latency`)

2.  **有损加速捷径 (Lossy Shortcuts)**:
    * 减小 KV 缓存大小 (`reduce kv cache size`)
    * Transformer 的替代架构 (`alternatives to the transformer`)
    * 量化 (`quantization`)
    * 模型剪枝 (`model pruning`)

3.  **无损加速捷径 (Lossless Shortcuts)**:
    * 推测采样 (`speculative sampling`)

4.  **处理动态工作负载**:
    * 连续批处理 (`continuous batching`)
    * 分页注意力 (`paged attention`)

## 1. 理解推理的工作负载

### 1.1 推理的应用场景 (Landscape)

推理在许多场景中都扮演着核心角色：
* **实际应用**: 聊天机器人、代码补全、批量数据处理。
* **模型评估**: 例如，在指令遵循任务上的评估。
* **测试时计算**: 更深度的“思考”需要更多的推理计算。
* **强化学习训练**: 用于生成样本，然后对样本进行评分。

**为什么效率至关重要？** 训练是一次性成本，而推理则会反复进行，规模巨大。例如，OpenAI 每天生成约1000亿个词元，而代码补全工具 Cursor 每天能生成十亿行被接受的代码。

**衡量推理性能的指标**:
* **首token时间 (Time-to-first-token, TTFT)**: 用户在看到第一个token之前需要等待多长时间，对交互式应用至关重要。
* **延迟 (Latency)**: 对单个用户而言，生成每个词元的速度（秒/词元），同样影响交互体验。
* **吞吐量 (Throughput)**: 对整个系统而言，单位时间内生成的总词元数（词元/秒），对批量处理应用更重要。

### 1.2 回顾计算强度 (Review of Arithmetic Intensity)

为了理解推理的瓶颈，我们首先要回顾**计算强度**的概念。

**通俗解释**: 想象 GPU 是一个超级厨房，里面有一位世界级大厨。
* **计算能力 (FLOPs)**: 大厨切菜的速度有多快。
* **内存带宽 (Memory Bandwidth)**: 食材从冰箱搬到案板上的速度有多快。
* **计算强度 (Arithmetic Intensity)**: **“每搬一次菜，大厨能切多少下？”**

如果“计算强度”很高，说明每拿到一点食材，大厨都能忙活很久。这时，厨房的效率由大厨有多快决定，我们称之为**计算受限 (Compute-limited)**，这是我们希望看到的，因为昂贵的大厨没有闲着。

如果“计算强度”很低，说明食材拿过来，大厨一刀就切完了，然后就得叉着腰等下一批食材。这时，厨房效率由搬运工有多快决定，我们称之为**内存受限 (Memory-limited)**，这是糟糕的情况，因为我们花大价钱请来的大厨大部分时间都在发呆。

下面的 Python 代码使用 `sympy` 库进行符号计算，精确地为我们进行这个“思想实验”：

```python
from sympy import symbols, oo

# 定义一些符号来代表模型的尺寸和硬件属性
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)  # 一个用于求极限的辅助常量
memory_bandwidth = symbols("memory_bandwidth", positive=True)

# 假设我们只做一个简单的矩阵乘法 X @ W
flops = 0
bytes_transferred = 0

# --- 思想实验的详细步骤 ---
# 1. 从慢速的HBM内存中读取输入矩阵X (B x D)
bytes_transferred += 2*B*D
# 2. 从慢速的HBM内存中读取权重矩阵W (D x F)
bytes_transferred += 2*D*F
# 3. 在GPU核心上进行计算
flops += 2*B*D*F
# 4. 将结果矩阵Y (B x F) 写回到慢速的HBM内存中
bytes_transferred += 2*B*F

# --- 盘点结果 ---
# 确认计算量和内存访问量的公式
assert flops == 2*B*D*F
assert bytes_transferred == 2*B*D + 2*D*F + 2*B*F

# 计算强度 = 计算量 / 内存访问量
intensity = (flops / bytes_transferred).simplify()

# 为了看清主导因素，我们做一个简化假设：
# 假设批量B远小于模型的深度D和宽度F。
# 讲稿中，经过这个简化，`intensity`最终结果约等于 `B` (批量大小)
# 这里的 @inspect 标记在原讲稿中用于展示中间结果
intensity_simplified = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify() # @inspect intensity
assert intensity_simplified == B

# H100 GPU自身的计算强度大约是295
accelerator_intensity = 989e12 / 3.35e12 # @inspect accelerator_intensity
assert round(accelerator_intensity) == 295

# 结论：只有当计算强度 > 硬件强度时，才是计算受限的。
# 这意味着，批量大小 B 必须大于 295！
# 在 B=1 的极端情况下（生成阶段的常态），计算强度也只有1，是严重的内存受限。
```

### 1.3 推理过程的计算强度 (Arithmetic Intensity of Inference)

语言模型的推理过程主要分为两个阶段：

1.  **预填充 (Prefill)**: 并行处理输入的提示（Prompt）。这个阶段**通常是计算受限的**。
2.  **生成 (Generation)**: 逐个生成新的词元，是串行过程。这个阶段**严重受内存带宽限制**。

讲稿中用同样的方法，分别对 MLP 层和 Attention 层进行了分析：

#### MLP 层的计算强度

```python
# --- MLP层的思想实验 ---
# S: 条件序列长度, T: 生成序列长度
S, T = symbols("S T", positive=True)

flops = 0
bytes_transferred = 0

# 1. 读取输入 X (B x T x D)
bytes_transferred += 2*B*T*D
# 2. 读取三个权重矩阵 W_up, W_gate, W_down
bytes_transferred += 3 * 2*D*F
# 3. 计算 U = X @ W_up 并写回
flops += 2*B*T*D*F
bytes_transferred += 2*B*T*F
# 4. 计算 G = X @ W_gate 并写回
flops += 2*B*T*D*F
bytes_transferred += 2*B*T*F
# 5. 计算 Y = GeLU(G)*U @ W_down 并写回
flops += 2*B*T*D*F
bytes_transferred += 2*B*T*D

# --- 盘点结果 ---
assert flops == 6*B*T*D*F
assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F

intensity = (flops / bytes_transferred).simplify() # @inspect intensity

# 简化假设：B*T 远小于 D 和 F
intensity_simplified = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify() # @inspect intensity
assert intensity_simplified == B*T
```
**MLP层结论**: 其计算强度正比于 `B*T`。在预填充阶段，`T`很大，容易实现计算受限。在生成阶段 `T=1`，强度为`B`，意味着只要并发请求数`B`够大，也能让GPU忙起来。

#### Attention 层的计算强度

```python
# --- Attention层的思想实验 (使用FlashAttention) ---
flops = 0
bytes_transferred = 0

# 1. 读取 Q, K, V 矩阵
bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
# 2. 计算 A = Q @ K
flops += 2*B*S*T*D
# 3. 计算 Y = softmax(A) @ V
flops += 2*B*S*T*D
# 4. 写回 Y
bytes_transferred += 2*B*T*D

# --- 盘点结果 ---
assert flops == 4*B*S*T*D
assert bytes_transferred == 4*B*S*D + 4*B*T*D

intensity = (flops / bytes_transferred).simplify() # @inspect intensity
assert intensity == S*T / (S + T)

# --- 分情况讨论 ---
# 1. 预填充阶段 (Prefill): T = S
prefill_intensity = intensity.subs(T, S).simplify() # @inspect prefill_intensity
assert prefill_intensity == S/2 # 强度为S/2，不错！

# 2. 生成阶段 (Generation): T = 1
generate_intensity = intensity.subs(T, 1).simplify() # @inspect generate_intensity
assert generate_intensity < 1 # 强度小于1，非常糟糕！
```
**Attention层结论**: 在生成阶段，其计算强度**始终小于1**，并且**和批量大小`B`无关**！这是问题的核心。无论我们同时处理多少个请求，注意力机制在生成新词元时，始终是严重的内存受限。

### 1.4 吞吐量与延迟 (Throughput and Latency)

既然生成阶段是内存受限的，那么速度就由“搬东西”的快慢决定。我们可以据此估算理论上的延迟和吞吐量。

下面的 Python 代码建立了一个“成本计算器”，来估算不同规模的模型在不同情况下的性能表现。

```python
# 这是一个“成本计算器”函数
def compute_transformer_stats(config):
    # 输入: config, 一个包含模型所有尺寸参数的配置单
    # (例如 Llama 2 有多少层，多宽，多少个头等等)

    # 1. 计算参数量 (模型的“体重”)
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L
    parameter_size = num_params * 2 # 使用 bf16, 每个参数占2字节

    # 2. 计算KV缓存大小 (模型处理当前对话需要的“草稿纸”)
    # 每个序列的草稿纸大小 = 序列长度 * (KV头数*每个头维度) * 层数 * 2 (K和V) * 2 (bf16)
    kv_cache_size = S * (K*H) * L * 2 * 2

    # 3. 总内存开销 = 模型体重 + 所有请求的草稿纸总大小
    memory = B * kv_cache_size + parameter_size

    # 4. 估算延迟: 生成一个词元的速度取决于总内存开销 / 内存带宽
    # (东西越多，搬得越慢)
    latency = memory / memory_bandwidth

    # 5. 估算吞吐量: B个请求同时进行，所以是 B / latency
    throughput = B / latency

    # 将配置代入公式得到最终的数学表达式
    num_params = num_params.subs(config).simplify() # @inspect num_params
    memory = memory.subs(config).simplify() # @inspect memory
    latency = latency.subs(config).simplify() # @inspect latency
    throughput = throughput.subs(config).simplify() # @inspect throughput

    return num_params, memory, latency, throughput

# 这是一个针对 Llama 2 13B 模型的配置单
def llama2_13b_config(args={}):
    return {S: 1024, D: 5120, F: 13824, N: 40, K: 40, H: 128, L: 40, V: 32000, memory_bandwidth: 3.35e12, **args}

# --- 开始用计算器估算 ---
# 1. 拿到 Llama 2 13B 的配置单
config = llama2_13b_config()
num_params, memory, latency, throughput = compute_transformer_stats(config)

# 2. 代入不同批量大小，看看性能表现
#   - 如果一次只处理1个请求 (B=1)
bs1_memory = memory.subs(B, 1).simplify() # @inspect bs1_memory
bs1_latency = latency.subs(B, 1).simplify() # @inspect bs1_latency
bs1_throughput = throughput.subs(B, 1).simplify() # @inspect bs1_throughput

#   - 如果一次处理64个请求 (B=64)
bs64_memory = memory.subs(B, 64).simplify() # @inspect bs64_memory
bs64_latency = latency.subs(B, 64).simplify() # @inspect bs64_latency
bs64_throughput = throughput.subs(B, 64).simplify() # @inspect bs64_throughput

#   - 如果一次处理256个请求 (B=256)
bs256_memory = memory.subs(B, 256).simplify() # @inspect bs256_memory
bs256_latency = latency.subs(B, 256).simplify() # @inspect bs256_latency
bs256_throughput = throughput.subs(B, 256).simplify() # @inspect bs256_throughput
# 这种情况会超出H100的80GB内存，但可以看到吞吐量增益在递减
```

**结论**: 这段代码的计算结果清晰地展示了延迟和吞吐量之间的权衡关系：小批量延迟低但吞吐量低；大批量吞吐量高但延迟也高，且最终会受到内存容量的限制。

## 2. 有损加速捷径 (Lossy Shortcuts)

这类方法通过牺牲一定的模型精度来换取推理速度。

### 2.1 减小 KV 缓存大小 (Reduce KV Cache Size)

KV缓存是推理过程中的“草稿纸”，是内存消耗的大头。下面的代码展示了**分组查询注意力 (GQA)** 的效果。GQA通过让多组“查询头”共用一个“键值头”来减少内存。

```python
# --- GQA 效果的思想实验 ---
# 原始 Llama 2 13B (B=64, K=40个KV头)
config = llama2_13b_config({K: 40, B: 64})
k40_num_params, k40_memory, k40_latency, k40_throughput = compute_transformer_stats(config)
# @inspect k40_memory, @inspect k40_latency, @inspect k40_throughput

# 使用GQA，将KV头减少到8个 (B=64, K=8)
config = llama2_13b_config({K: 8, B: 64})
k8_num_params, k8_memory, k8_latency, k8_throughput = compute_transformer_stats(config)
# @inspect k8_memory, @inspect k8_latency, @inspect k8_throughput
# 结论: 内存占用大幅下降，延迟降低，吞吐量提升！

# 因为省下了内存，现在我们可以用更大的批量来进一步提升吞吐量 (B=256, K=8)
config = llama2_13b_config({K: 8, B: 256})
k8_bs_num_params, k8_bs_memory, k8_bs_latency, k8_bs_throughput = compute_transformer_stats(config)
# @inspect k8_bs_memory, @inspect k8_bs_latency, @inspect k8_bs_throughput
# 结论: 延迟虽然变差了，但吞吐量变得更高了，而且现在内存也够用了！
```
**GQA结论**: GQA能有效降低内存占用，从而允许我们使用更大的批量，最终以稍高的延迟为代价，换取了显著的吞吐量提升。

### 2.2 Transformer 的替代架构

* **状态空间模型 (State-Space Models, SSMs)**: 如 Mamba。工作模式更像RNN，推理时快且省内存。
* **扩散模型 (Diffusion Models)**: 不再一个一个地蹦词，而是并行生成所有词元，然后迭代修正。

### 2.3 量化 (Quantization)

* **思想**: 使用更低精度的数据类型（如INT8）来表示模型的权重和激活值，从而为模型的“体重”和“草稿纸”减肥。
* **效果**: 内存占用减半或更多，吞吐量飙升。

### 2.4 模型剪枝 (Model Pruning)

* **思想**: 像园丁修剪花草一样，把模型里那些不太重要的“枝叶”（如某些注意力头、网络层）给剪掉，然后通过“蒸馏”来恢复其性能。

## 3. 无损加速捷径 (Lossless Shortcuts)

这类方法在加速的同时，能保证生成结果与原始模型一模一样。

### 3.1 推测采样 (Speculative Sampling)

* **思想**: 找一个小的“草稿模型”和一个大的“目标模型”合作。小模型先“草率”地写，大模型再来“批改”。
* **如何保证无损**: 通过一个精巧的概率算法。下面的代码用一个只有两个词`{A, B}`的例子证明了这一点。

```python
# --- 推测采样无损性证明 ---
# 假设词汇表里只有两个词 {A, B}
# 目标模型（慢）的概率分布是 [q(A), q(B)]
# 草稿模型（快）的概率分布是 [p(A), p(B)]

# 假设草稿模型高估了A的概率: p(A) > q(A)，因此低估了B的概率 p(B) < q(B)

# 算法接受草稿模型提议的A的概率是 q(A)/p(A)
# 算法接受草稿模型提议的B的概率是 1 (因为p(B)<q(B))

# 计算最终采样到 A 的总概率
# P[采样到A] = (草稿提议A的概率 * 接受A的概率) + (草稿提议B的概率 * 拒绝B后采样到A的概率)
# P_A = p(A) * (q(A) / p(A)) + p(B) * ( ... 复杂的拒绝后采样公式 ... ) * (最终选A的概率)
# 经过推导，最终 P_A 恰好等于 q(A)
# P[采样到B] 也同理等于 q(B)

# 这段代码在讲稿中用于演示这一概率的守恒
P_sampling_A = "p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A)"
P_sampling_B = "p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B)"
```
**结论**: 无论过程多么曲折，最终从这个“合作流程”中采样到每个词的概率，和直接从大模型中采样是完全一样的。

## 4. 处理动态工作负载

真实服务场景中，请求是动态到达的，且长度各不相同，这给批处理带来了挑战。

* **连续批处理 (Continuous Batching)**:
    * **思想**: 把GPU想象成一辆永不停歇的公交车。车上有人到站了（生成完毕），不用等全车人都下车，马上就让站台上的新乘客上车补位。
    * **效果**: 最大化GPU的利用率，减少空等。

* **分页注意力 (Paged Attention)**:
    * **思想**: 借鉴操作系统的“虚拟内存”技术来管理KV缓存。把巨大的、连续的“草稿纸”需求，切成一小块一小块的“便签纸”，在内存里见缝插针地存储。
    * **效果**: 极大地减少了内存浪费（碎片），让有限的内存空间能容纳更多的请求，从而提升吞吐量。

## 总结

推理是一个比训练更复杂、更具挑战性的领域。虽然存在许多系统级的优化技巧，但最具潜力的性能提升可能来自于对模型架构本身的根本性变革，例如状态空间模型和扩散模型，因为它们能够从根本上绕开现有 Transformer 架构的内存瓶颈。