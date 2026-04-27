# 并行策略可视化设计（v1.2+）

> **⚠ 范围声明**：本文档为 v1.2+ 并行策略可视化的**设计预研草稿**，不属于 v1.0 交付范围。v1.0 不建 `ParallelismStrategy` 接口/Registry/目录（见 11 §4 / ADR-020）。本文档中的具体实现细节（通信原语可视化、GPU 拓扑布局等）在 v1.2 正式启动时需重新评审后方可采纳。

> **本文件不属于 v1.0 交付范围**，为 v1.2+ 并行策略可视化的概念草案，供未来迭代参考。

> 本文档从原始内部设计稿提取。
> 包含 TP/PP/DP/EP/CP/SP 并行策略的 3D 可视化设计、训练数据流设计。
> 计划在 v1.2+ 实现。

---

## 十、并行策略可视化设计

### 10.1 支持的并行策略

用户可选择一种或多种并行策略，系统自动展示该策略下的数据流过程。

| 策略 | 全称 | 核心思想 | 配置参数 |
|---|---|---|---|
| **DP** | Data Parallelism | 模型复制，数据分片 | `dp_size` |
| **TP** | Tensor Parallelism | 权重按列/行切分到多 GPU | `tp_size` |
| **PP** | Pipeline Parallelism | 层分配到不同流水线阶段 | `pp_size`, `vpp_size` |
| **EP** | Expert Parallelism | MoE 专家分布到不同 GPU | `ep_size` |
| **CP** | Context Parallelism | 长序列切分到多 GPU | `cp_size` |
| **SP** | Sequence Parallelism | TP 组内非 TP 区域的激活序列分片 | `sequence_parallel: bool` |

### 10.2 并行策略配置面板

```
┌──────────────────────────────────────────────────────┐
│  ⚙️  并行策略配置                                      │
│                                                        │
│  模型: deepseek-ai/DeepSeek-V3 (61 层, 256 专家)          │
│  总 GPU 数: [  64  ] = TP × PP × DP × CP              │
│                                                        │
│  ┌──── 基础并行 ──────────────────────────────────┐    │
│  │  TP (Tensor):    [ 8 ▾]                         │    │
│  │  PP (Pipeline):  [ 4 ▾]   VPP: [ 2 ▾]          │    │
│  │  DP (Data):      [ 2 ▾]   ☑ FSDP/ZeRO          │    │
│  └─────────────────────────────────────────────────┘    │
│                                                        │
│  ┌──── 高级并行 ──────────────────────────────────┐    │
│  │  EP (Expert):    [ 8 ▾]   策略: [ round_robin ▾]│    │
│  │  CP (Context):   [ 2 ▾]   方式: [ Ring Attn ▾]  │    │
│  │  SP (Sequence):  [☑]      (依附于 TP)            │    │
│  └─────────────────────────────────────────────────┘    │
│                                                        │
│  [ 🔍 可视化 ]  [ 📊 通信量估算 ]  [ ⚡ 推荐配置 ]   │
└──────────────────────────────────────────────────────┘
```

### 10.3 GPU 拓扑 3D 可视化

```
═══ 设备网格 (Device Mesh) ═══

按 Megatron 惯例: order = "tp-cp-ep-dp-pp"

64 GPU 示例 (TP=8, PP=4, DP=2):

             PP Stage 0        PP Stage 1        PP Stage 2        PP Stage 3
             (Layer 0-14)      (Layer 15-29)     (Layer 30-44)     (Layer 45-60)

  DP Rep 0:  [GPU 0-7]         [GPU 8-15]        [GPU 16-23]       [GPU 24-31]
             TP group 0        TP group 1         TP group 2        TP group 3

  DP Rep 1:  [GPU 32-39]       [GPU 40-47]       [GPU 48-55]       [GPU 56-63]
             TP group 4        TP group 5         TP group 6        TP group 7

3D 渲染:
  - 每个 GPU 为一个 3D 立方体
  - TP 组内 GPU 紧密排列（X 轴），共享同一层的权重分片
  - PP 阶段沿 Y 轴排列，间有流水线连接线
  - DP 副本沿 Z 轴排列，间有梯度同步连线
  - EP GPU 使用橙色标记（持有专家子集）
  - 通信连线颜色编码:
    - AllReduce: 红色双向箭头
    - AllGather: 绿色汇聚箭头
    - ReduceScatter: 蓝色分散箭头
    - All-to-All: 橙色网状连线
    - P2P Send/Recv: 白色单向箭头
```

### 10.4 各策略的数据流 3D 动画

#### TP (Tensor Parallelism) 数据流

```
━━━ 一个 Transformer Block 内的 TP 数据流 ━━━

每个 TP rank 持有权重的一个分片:

  输入: hidden_states [s, h] (全量复制在每个 TP rank)
  │
  ├─ ColumnParallelLinear (QKV / gate_up_proj):
  │    权重 [h, h_out/tp] — 按列切分
  │    前向: 本地矩阵乘法，无通信 ✓
  │    输出: [s, h_out/tp] (每 rank 持有输出的一个分片)
  │    动画: 权重块高亮显示列切分边界
  │
  ├─ Attention / SwiGLU 激活:
  │    本地计算，无通信 ✓
  │
  ├─ RowParallelLinear (O_proj / down_proj):
  │    权重 [h_in/tp, h] — 按行切分
  │    前向: 本地矩阵乘法 → AllReduce(TP group) ← 关键通信!
  │    动画: 各 rank 的部分和汇聚为完整输出
  │    粒子: 8 束粒子从 8 个 TP rank 汇聚到一点
  │
  └─ 输出: hidden_states [s, h] (全量)

  ━━━ SP 激活时的额外可视化 ━━━

  无 SP: AllReduce = 所有 rank 得到完整 [s, h]
  有 SP: ReduceScatter → [s/tp, h] → LayerNorm(local) → AllGather → [s, h]
         动画: 激活张量被"切片"成 tp 份，经过 Norm 后再"拼合"
         颜色: SP 区域用不同底色标记（半透明淡紫色）
```

#### PP (Pipeline Parallelism) 数据流

```
━━━ 流水线调度 Gantt 图 ━━━

1F1B 调度 (PP=4, microbatch=8):

  时间 →  t0   t1   t2   t3   t4   t5   t6   t7   t8   t9  ...
  GPU 0:  F0   F1   F2   F3   B0   F4   B1   F5   B2   F6  ...
  GPU 1:  ──   F0   F1   F2   F3   B0   F4   B1   F5   B2  ...
  GPU 2:  ──   ──   F0   F1   F2   F3   B0   F4   B1   F5  ...
  GPU 3:  ──   ──   ──   F0   F1   F2   F3   B0   F4   B1  ...

  F=Forward, B=Backward, ──=空闲(气泡)

3D 可视化:
  - 横轴: 时间步
  - 纵轴: PP 阶段（GPU）
  - 每个块: 前向=蓝色, 后向=红色, 气泡=灰色半透明
  - P2P 通信: 白色箭头连接相邻阶段
  - 粒子: 微批次沿流水线向下流动（前向）→ 向上回流（后向）
  - 气泡区域: 脉动灰色，标注 "Bubble: xx%"

VPP (Interleaved) 对比:
  - 同一 GPU 上的多个虚拟阶段用不同深度的颜色
  - 气泡显著缩小，用数字标注 "Bubble: xx% → yy%"
```

#### EP (Expert Parallelism) 数据流

```
━━━ MoE + EP 数据流（DeepSeek-V3: 256 专家, EP=8）━━━

每个 EP rank 持有 256/8 = 32 个专家:

  Router: 所有 rank 本地计算路由评分
  │
  ├─ Dispatch (All-to-All):
  │    每个 token 被发送到持有其目标专家的 rank
  │    动画: token 粒子从源 GPU 飞向目标 GPU
  │    8 个 rank 间形成 8×8 通信矩阵
  │    连线粗细 = 传输 token 数量
  │
  ├─ Expert Compute:
  │    每个 rank 本地计算其 32 个专家
  │    动画: MoE 网格中属于当前 rank 的专家发光
  │    不同 rank 的专家用不同底色标记
  │
  └─ Combine (All-to-All):
       结果发送回原始 rank
       动画: 粒子从专家 GPU 飞回源 GPU

3D 可视化:
  - MoE 专家网格(16×16) 按 EP 分组着色:
    rank 0: 行 0-1 (蓝)  rank 1: 行 2-3 (绿)
    rank 2: 行 4-5 (橙)  rank 3: 行 6-7 (紫)
    ...
  - All-to-All 通信: 8×8 GPU 间的橙色网状连线
  - Token 流动: 粒子从 Router → 散向各 EP rank → 回收
```

#### CP (Context Parallelism) 数据流

```
━━━ Ring Attention (CP=4, 序列长度 256K) ━━━

每个 CP rank 处理 64K tokens:
  rank 0: tokens [0, 64K)     Q_0, K_0, V_0
  rank 1: tokens [64K, 128K)  Q_1, K_1, V_1
  rank 2: tokens [128K, 192K) Q_2, K_2, V_2
  rank 3: tokens [192K, 256K) Q_3, K_3, V_3

Ring Attention 步骤:
  Step 0: 本地 Q_i × K_i^T → 部分注意力
  Step 1: K,V 环形传递 (rank i → rank i+1)
          Q_i × K_{(i-1)%4}^T → 累加部分注意力
  Step 2: 再次传递 K,V
          Q_i × K_{(i-2)%4}^T → 累加
  Step 3: 最后一次传递
          Q_i × K_{(i-3)%4}^T → 最终注意力输出

3D 可视化:
  - 4 个 GPU 排列为环形（顶视图）
  - 序列被四色切分: 蓝/绿/橙/紫
  - KV 块沿环形 P2P 传递（白色箭头动画）
  - 注意力权重矩阵逐步填充:
    ┌────┬────┬────┬────┐
    │ ██ │ ░░ │ ░░ │ ░░ │  Step 0: 对角块
    │ ░░ │ ██ │ ░░ │ ░░ │
    │ ░░ │ ░░ │ ██ │ ░░ │
    │ ░░ │ ░░ │ ░░ │ ██ │
    └────┴────┴────┴────┘
    → Step 1: 次对角块填充 → ... → Step 3: 全矩阵

━━━ Ulysses (All-to-All 方式) ━━━

  Before attention:
    All-to-All: [s/cp, h] → [s, h/cp]  (每 rank 拿到全序列，但只处理部分头)
    动画: 序列块飞散成头分片

  Attention: 每 rank 在全序列上计算其局部头
    动画: 完整序列可见但只有部分头高亮

  After attention:
    All-to-All: [s, h/cp] → [s/cp, h]  (恢复序列分片)
    动画: 头分片回合为序列块
```

### 10.5 准确性保证

并行策略数据流的准确性基于以下来源：

| 来源 | 文件 | 提供信息 |
|---|---|---|
| Megatron-LM `parallel_state.py` | 进程组构造 | GPU 分组拓扑: order="tp-cp-ep-dp-pp" |
| Megatron-LM `mappings.py` | 通信原语实现 | 精确的 AllReduce/AllGather/ReduceScatter/All-to-All 位置 |
| Megatron-LM `layers.py` | ColumnParallel/RowParallel | 权重切分方式 + 前向/反向通信 |
| Megatron-LM `schedules.py` | PP 调度 | 1F1B/Interleaved 微批次调度时序 |
| vLLM `sequence_parallelism.py` | SP 编译器 Pass | AllReduce → ReduceScatter + Norm + AllGather 变换规则 |
| vLLM `all2all.py` | EP All-to-All | Dispatch/Combine 的精确实现 |
| 模型 config.json | 层数、专家数、头数等 | 决定切分粒度 |

#### 验证方法

```python
def validate_parallel_flow(config: dict, parallel_config: ParallelConfig) -> ValidationResult:
    """验证并行数据流的正确性"""

    # 1. 验证 TP 切分: 头数和中间维度必须能被 tp_size 整除
    assert config["num_attention_heads"] % parallel_config.tp_size == 0
    assert config.get("num_key_value_heads", config["num_attention_heads"]) % parallel_config.tp_size == 0
    assert config["intermediate_size"] % parallel_config.tp_size == 0

    # 2. 验证 PP 切分: 层数能被 pp_size 整除时均分，否则末级 stage 多分 1 层
    num_layers = config["num_hidden_layers"]
    if parallel_config.vpp_size:
        assert num_layers % (parallel_config.pp_size * parallel_config.vpp_size) == 0
    # 注: DeepSeek-V3 61 层 PP=4 时不可均分（61 % 4 != 0），实际采用 15/15/15/16 非均匀切分

    # 3. 验证 EP 切分: 专家数必须能被 ep_size 整除
    if config.get("n_routed_experts"):
        assert config["n_routed_experts"] % parallel_config.ep_size == 0

    # 4. 验证 CP 切分: 序列长度必须能被 cp_size 整除
    # (运行时验证，取决于实际输入)

    # 5. 通信量估算
    comm_volume = estimate_communication_volume(config, parallel_config)
    return ValidationResult(valid=True, comm_volume=comm_volume)
```

### 10.6 通信量估算与展示

```
┌──────────────────────────────────────────────────────────┐
│  📊 每 Transformer Block 通信量估算                        │
│                                                            │
│  TP (AllReduce ×2):                                        │
│  ████████████████████████  2 × 2 × s × h bytes             │
│  = 2 × 2 × 8192 × 7168 × 2 (bf16) ≈ 450 MB               │
│                                                            │
│  SP (ReduceScatter + AllGather ×2, 替代 AllReduce):        │
│  ████████████████████████  通信量同 TP，但分两步             │
│                                                            │
│  PP (P2P Send per stage boundary):                         │
│  ████████  mb × s × h × 2 bytes                            │
│  = 1 × 8192 × 7168 × 2 = 117 MB                           │
│                                                            │
│  EP (All-to-All ×2 per MoE layer):                         │
│  ██████████████  取决于路由分布                              │
│  最坏: 2 × tokens × top_k × expert_hidden × 2 bytes       │
│                                                            │
│  DP (AllReduce on gradients, per step):                    │
│  ████████████████████████████████████  全模型梯度大小        │
│                                                            │
│  总通信: [██████████████████████] 3.2 TB/step (估算)       │
│  计算/通信比: [████████░░░░░░░░░░] 2.4                     │
└──────────────────────────────────────────────────────────┘
```

### 10.7 组合并行可视化

当用户同时选择多种并行策略时，展示组合效果：

```
TP=8 + PP=4 + EP=8 + SP 组合 (DeepSeek-V3 推荐部署):

  PP Stage 0 (Layer 0-14)                PP Stage 1 (Layer 15-29)
  ┌─────────────────────────────┐       ┌─────────────────────────────┐
  │  GPU 0  GPU 1  ... GPU 7    │  P2P  │  GPU 8  GPU 9  ... GPU 15  │
  │  TP=8: 权重按列/行切分       │ ────→ │  TP=8: 同                   │
  │  SP: LayerNorm 区域序列分片  │       │  SP: 同                     │
  │  EP: 专家分到 8 个 GPU       │       │  EP: 同                     │
  └─────────────────────────────┘       └─────────────────────────────┘
                │                                    │
                │ P2P                                │ P2P
                ▼                                    ▼
  PP Stage 2 (Layer 30-44)               PP Stage 3 (Layer 45-60)
  ┌─────────────────────────────┐       ┌─────────────────────────────┐
  │  GPU 16 ... GPU 23          │  P2P  │  GPU 24 ... GPU 31          │
  │  TP + SP + EP               │ ────→ │  TP + SP + EP + LM Head     │
  └─────────────────────────────┘       └─────────────────────────────┘

3D 动画播放模式:
  1. "TP 视角" — 展开一个 TP 组，看权重切分 + AllReduce
  2. "PP 视角" — 拉远看流水线调度 Gantt 图
  3. "EP 视角" — 聚焦 MoE 层，看 All-to-All 专家路由
  4. "全局视角" — 所有通信连线叠加，动画展示一个完整 step
```

### 10.8 训练数据流可视化

训练与推理的数据流有本质区别：推理只有前向传播，而训练包含**前向→反向→梯度同步→优化器更新**完整循环，且并行策略在反向传播中的通信模式完全不同。

#### 推理 vs 训练模式切换

```
┌──────────────────────────────────────────┐
│  模式: [ 🔮 推理 ]  [ 🏋️ 训练 ]          │
│                                            │
│  推理模式: 单向前向传播数据流              │
│  训练模式: 前向→反向→梯度同步→优化器      │
│           + 激活重计算 + 梯度累积         │
└──────────────────────────────────────────┘
```

#### A. 训练数据流全景（单个 Training Step）

```
═══════════════════ 训练数据流 (一个完整 step) ═══════════════════

1. 数据加载 + 预处理
   ┌─────────────────────────────────────────────────┐
   │  DataLoader → micro_batches[]                    │
   │  动画: 数据流从左侧"数据源"涌入                  │
   │  梯度累积 = N 个 microbatch / DP rank            │
   └──────┬──────────────────────────────────────────┘
          │
2. 前向传播 (Forward Pass) — 蓝色粒子向下流
   ┌──────┴──────────────────────────────────────────┐
   │  与推理相同，但额外标注:                          │
   │                                                   │
   │  ┌─ Layer i ─────────────────────────────┐       │
   │  │  前向计算 → 保存激活 (用于反向)        │       │
   │  │  ☑ 激活保存: activations[i] = output   │       │
   │  │  ☐ 激活重计算: 不保存，反向时重算       │       │
   │  │                                        │       │
   │  │  颜色标注:                              │       │
   │  │  🟢 保存激活的层: 绿色内存条增长         │       │
   │  │  🟡 重计算的层: 黄色标记 "⚡ recompute" │       │
   │  └────────────────────────────────────────┘       │
   │                                                   │
   │  TP 通信 (前向):                                   │
   │  ColumnParallel: 无通信 (本地计算)                 │
   │  RowParallel: AllReduce(TP) ← 蓝色箭头             │
   │                                                   │
   │  PP 通信 (前向):                                   │
   │  stage_i → P2P Send → stage_{i+1}                 │
   │  蓝色粒子沿流水线向下                              │
   └──────┬──────────────────────────────────────────┘
          │
          ▼
3. Loss 计算
   ┌──────────────────────────────────────────────────┐
   │  logits → CrossEntropyLoss(labels)                │
   │  动画: logits 与 labels 对比 → loss 值"爆发"点    │
   │  loss 标量从 LM Head 底部产生一个红色脉冲          │
   └──────┬───────────────────────────────────────────┘
          │
          ▼
4. 反向传播 (Backward Pass) — 红色粒子向上流
   ┌──────┴──────────────────────────────────────────┐
   │  梯度从 loss 反向流过每一层 ← 关键: 方向翻转!     │
   │                                                   │
   │  ┌─ Layer i (反向) ──────────────────────────┐   │
   │  │                                            │   │
   │  │  如果是激活重计算层:                        │   │
   │  │  ⚡ 先重新前向计算 (蓝色闪光)               │   │
   │  │  → 然后反向计算梯度                        │   │
   │  │                                            │   │
   │  │  反向计算:                                  │   │
   │  │  grad_output → 权重梯度 + 输入梯度          │   │
   │  │  红色粒子从下方流入 → 分裂为:               │   │
   │  │    → 权重梯度 (黄色粒子，留在本地)          │   │
   │  │    → 输入梯度 (红色粒子，继续向上流)        │   │
   │  └────────────────────────────────────────────┘   │
   │                                                   │
   │  TP 通信 (反向 — 注意方向翻转!):                   │
   │  ColumnParallel 反向: AllReduce(TP) ← 红色箭头     │
   │    (前向无通信，反向有 AllReduce — 正好翻转!)       │
   │  RowParallel 反向: 无通信                          │
   │    (前向有 AllReduce，反向无通信 — 正好翻转!)       │
   │                                                   │
   │  PP 通信 (反向):                                   │
   │  stage_{i+1} → P2P Send grad → stage_i             │
   │  红色粒子沿流水线向上                              │
   │                                                   │
   │  SP 通信 (反向):                                   │
   │  AllGather 反向 = ReduceScatter                    │
   │  ReduceScatter 反向 = AllGather                    │
   │  (前向/反向通信原语互换)                            │
   └──────┬──────────────────────────────────────────┘
          │
          ▼
5. 梯度同步 (DP AllReduce / ReduceScatter)
   ┌──────┴──────────────────────────────────────────┐
   │  所有 DP 副本的梯度需要同步                       │
   │                                                   │
   │  DDP: AllReduce(DP group) on all gradients         │
   │  动画: 所有 DP 副本间的红色双向箭头               │
   │                                                   │
   │  FSDP/ZeRO-2/3: ReduceScatter(DP group)           │
   │  动画: 梯度被"切片"分散到各 DP rank               │
   │                                                   │
   │  ⚡ 通信与计算重叠 (overlap_grad_reduce):          │
   │  反向计算尚未完成时，已完成层的梯度开始同步        │
   │  动画: 红色箭头(梯度同步)与红色粒子(反向计算)      │
   │        同时进行，用虚线分隔区域                    │
   └──────┬──────────────────────────────────────────┘
          │
          ▼
6. 优化器更新 (Optimizer Step)
   ┌──────┴──────────────────────────────────────────┐
   │  AdamW 参数更新:                                  │
   │  m_t = β₁ · m_{t-1} + (1-β₁) · grad             │
   │  v_t = β₂ · v_{t-1} + (1-β₂) · grad²            │
   │  param -= lr × m̂_t / (√v̂_t + ε) + wd × param    │
   │                                                   │
   │  动画:                                             │
   │  黄色脉冲(梯度) → 与紫色(动量m) 和橙色(方差v)    │
   │  "混合" → 产生绿色脉冲(更新量)                    │
   │  → 应用到蓝色权重块上 (权重块微闪一下)            │
   │                                                   │
   │  FSDP: AllGather(DP group) 更新后的参数            │
   │  动画: 更新后的参数分片"汇聚"为完整参数           │
   └──────┬──────────────────────────────────────────┘
          │
          ▼
7. 学习率调度 + 下一步
   ┌──────┴──────────────────────────────────────────┐
   │  lr_scheduler.step()                              │
   │  动画: 学习率曲线上的指示点前进一步               │
   │  → 循环回到 Step 1 (下一个 step)                  │
   └──────────────────────────────────────────────────┘
```

#### B. PP 训练调度 3D 时序图

```
━━━ 1F1B 调度 (PP=4, microbatch=8) — 3D Gantt 图 ━━━

纵轴: PP Stage (GPU)
横轴: 时间步
颜色: 蓝=前向, 红=反向, 灰=气泡

  GPU 0  ┃ F₀ │ F₁ │ F₂ │ F₃ │ B₀ │ F₄ │ B₁ │ F₅ │ B₂ │ F₆ │ B₃ │ F₇ │ B₄ │ B₅ │ B₆ │ B₇ │
  GPU 1  ┃ ░░ │ F₀ │ F₁ │ F₂ │ F₃ │ B₀ │ F₄ │ B₁ │ F₅ │ B₂ │ F₆ │ B₃ │ F₇ │ B₄ │ B₅ │ B₆ │
  GPU 2  ┃ ░░ │ ░░ │ F₀ │ F₁ │ F₂ │ F₃ │ B₀ │ F₄ │ B₁ │ F₅ │ B₂ │ F₆ │ B₃ │ F₇ │ B₄ │ B₅ │
  GPU 3  ┃ ░░ │ ░░ │ ░░ │ F₀ │ F₁ │ F₂ │ F₃ │ B₀ │ F₄ │ B₁ │ F₅ │ B₂ │ F₆ │ B₃ │ F₇ │ B₄ │
                warmup          steady-state (1F1B)                       cooldown

  ░░ = 气泡 (Pipeline Bubble)
  气泡占比 = (P-1) / (M + P - 1) = 3/11 ≈ 27.3%

3D 渲染:
  - 每个 F/B 块是一个 3D 长方体（蓝/红）
  - 气泡是半透明灰色块
  - P2P 通信: F 块底部 → 下一 GPU 的 F 块顶部 (白色箭头)
  - Hover 单个块: 显示 microbatch ID + 计算耗时
  - 点击 "Warmup" / "Steady" / "Cooldown" 标签聚焦该阶段

━━━ Interleaved 1F1B (VPP=2) 对比 ━━━

  同一 GPU 上有两种颜色深度的块（对应两个虚拟阶段）
  气泡占比从 27.3% 降到 ~15.8%（(P-1)/(M×v+P-1) = 3/19）
  动画: 并排播放两种调度，直观对比气泡缩减
```

#### C. 激活重计算可视化

```
━━━ 激活重计算 (Activation Checkpointing / Gradient Checkpointing) ━━━

模式切换:
  ☐ 无重计算 — 保存所有激活 (最大显存，最快速度)
  ☑ 全量重计算 — 只保存层边界激活 (最小显存，~33% 额外计算)
  ☐ 选择性重计算 — 保存 Attention 输出, 重算 MLP (平衡)

3D 可视化:

  无重计算时:
  ┌─────┐  显存
  │ L60 │  ████████████████████████ 全满
  │ ... │  ████████████████████████
  │ L1  │  ████████████████████████
  │ L0  │  ████████████████████████
  └─────┘

  全量重计算时:
  ┌─────┐  显存                        反向时:
  │ L60 │  ████                        ⚡ 重算 L59-L60 → 反向 L60
  │ ... │  (未保存)                     ⚡ 重算 L58-L59 → 反向 L59
  │ L1  │  (未保存)                     ...
  │ L0  │  ████ (checkpoint)           ⚡ 重算 L0-L1  → 反向 L1
  └─────┘                              反向 L0 (无需重算)

  动画 (反向传播时):
  - 每到一个重计算层: 蓝色闪光 (前向重算) → 然后红色 (反向)
  - 显存条: 动态显示当前显存占用
  - 对比模式: 左=无重计算(显存高,速度快), 右=全量重计算(显存低,速度慢)
```

#### D. 梯度累积可视化

```
━━━ 梯度累积 (Gradient Accumulation Steps = 4) ━━━

  microbatch 0:  Forward → Backward → grad += local_grad  (不同步!)
  microbatch 1:  Forward → Backward → grad += local_grad  (不同步!)
  microbatch 2:  Forward → Backward → grad += local_grad  (不同步!)
  microbatch 3:  Forward → Backward → grad += local_grad → AllReduce(DP) → Optimizer Step

3D 动画:
  - 4 个 microbatch 依次穿过模型（蓝→红 循环 4 次）
  - 梯度以黄色粒子形式在各层"堆积"（每次累加，黄色加深）
  - 第 4 次后: 黄色粒子"爆发"→ AllReduce 扩散到所有 DP rank
  - 优化器绿色脉冲应用更新
  - 显存条: 标注 "等效 batch_size = micro_bs × grad_accum × dp_size"
```

#### E. TP 前向 vs 反向通信对比

```
━━━ 关键洞察: TP 通信方向在前向/反向翻转 ━━━

这是理解分布式训练最重要的直觉之一:

前向传播:
  ColumnParallel (QKV, gate_up):  无通信 → 有通信(AllReduce反向)
  RowParallel (O, down):         有通信(AllReduce前向) → 无通信

  x → [ColumnParallel: 无通信] → y_partial
  y_partial → [RowParallel: AllReduce] → y

反向传播 (通信翻转!):
  ColumnParallel 反向:  AllReduce(TP) on ∂L/∂x  ← 红色箭头
  RowParallel 反向:     无通信 (Split 即可)

  ∂L/∂y → [RowParallel反向: 无通信] → ∂L/∂y_partial
  ∂L/∂y_partial → [ColumnParallel反向: AllReduce] → ∂L/∂x

3D 动画 (分屏对比):
  左面板: 前向传播 — 蓝色粒子向下，AllReduce 在 RowParallel 处
  右面板: 反向传播 — 红色粒子向上，AllReduce 在 ColumnParallel 处
  高亮标注: "注意: 通信位置翻转!"

  TP 总通信量不变:
  前向 2×AllReduce + 反向 2×AllReduce = 4×AllReduce per block
  (或 SP 模式: 4×ReduceScatter + 4×AllGather per block, 等价通信量)
```

#### F. 训练 Guided Tour 时间线

```
训练模式 Guided Tour (120s):

  T=0s    全景: "训练一个 step 的完整数据流"
  T=5s    数据加载: DataLoader → microbatch 分配给 DP ranks
  T=10s   前向 Phase 1: 粒子进入 Embedding → 注解激活保存
  T=18s   前向 Phase 2: 穿过 Attention → TP AllReduce(蓝色箭头)
  T=25s   前向 Phase 3: 穿过 MoE → EP All-to-All(橙色箭头)
  T=32s   PP 视角: 微批次在 4 个 stage 间流水线传播
  T=40s   Loss 计算: logits vs labels → 红色脉冲"爆发"
  T=45s   反向 Phase 1: 红色粒子从 LM Head 回流 → 注解"通信翻转"
  T=52s   反向 Phase 2: 激活重计算层 → 蓝色闪光+红色计算
  T=60s   反向 Phase 3: MoE 反向 → EP All-to-All(反向)
  T=68s   TP 通信对比: 分屏展示前向 vs 反向的通信位置翻转
  T=75s   PP 反向: 红色粒子沿流水线向上回流
  T=82s   梯度同步: DP AllReduce → 所有 DP rank 间红色双向箭头
  T=90s   重叠可视化: 反向计算 + 梯度同步同时进行
  T=98s   优化器: AdamW 动量+方差混合 → 绿色脉冲更新权重
  T=105s  FSDP AllGather: 更新后参数"汇聚"
  T=110s  学习率曲线: 指示点前进
  T=115s  全景拉远: 标注关键统计 (气泡占比、通信量、显存峰值)
  T=120s  循环提示: "下一个 step..."
```

---
