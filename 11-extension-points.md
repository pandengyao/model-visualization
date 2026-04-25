# 11 — 扩展点契约（Extension Points）

> 本文档是 README 原则 6「v1.0 架构必须最健壮且可扩展」的**落地契约单一事实源**。
> 所有扩展点的接口签名、注册机制、验收标准均以本文档为准。
> 与 09（后端详细设计）冲突时，本文档对「扩展契约」有优先权；09 对「具体实现」有优先权。

## 0. 总原则

1. **接入成本上限**：任何扩展的接入 ≤ **1 个新文件 + 1 处注册**。违反即视为契约破坏。
2. **单一计算源**：pipeline 只在后端 Python 实现一次，禁止前端重写。
3. **契约向前兼容**：所有 schema 通过 `revision` 字段演进，禁止破坏性修改。
4. **纯函数约束**：所有 pipeline 阶段、所有扩展插件，必须是无副作用纯函数（I/O 除外）。

---

## 1. ArchitectureAdapter（模型类型扩展）

### 1.1 接口签名（Python）

```python
class ArchitectureAdapter(Protocol):
    """每种模型架构族对应一个 Adapter。"""

    # 元数据
    name: str                       # 如 "llama", "deepseek_moe", "mamba"
    matches: list[str]              # config.model_type 可匹配值列表
    confidence: Confidence          # EXACT / INFERRED

    def detect(self, config: dict) -> bool:
        """强校验：该 config 是否真的属于本架构族。"""

    def build_graph(
        self,
        config: dict,
        safetensors_meta: dict | None,
        meta_state_dict: dict | None,
    ) -> ModuleGraph:
        """构建 ModuleGraph（节点+边+层级），必须携带 Provenance。"""

    def template_id(self) -> Literal["A", "B", "C", "G", ...]:
        """返回前端渲染模板 ID。"""
```

### 1.2 注册方式

每个 adapter 模块内使用 `@register` 装饰器自注册（与 Detector 注册模式统一，见 09 §5.2/ADR-014）：

```python
# backend/adapters/llama.py
from .registry import register

@register
class LlamaAdapter:
    ...
```

`backend/adapters/__init__.py`:
```python
# 仅导入模块即可触发 @register 装饰器（副作用注册）
from . import llama           # noqa: F401
from . import deepseek_moe    # noqa: F401
# 新增架构 = 新建 adapters/<name>.py（类上加 @register）+ 此处追加一行 import
```

### 1.3 验收
- **硬性**：`detect_features` / `synthesize_flows` / `compute_layout` 中**不得**出现 `if model_type == "..."` 分支。
- **演练**：Phase 1 结束前必须用 Mamba（无 attention）或 ViT（无 causal mask）做一次 cold-start 接入，验证 ≤ 1 文件 + 1 注册。

---

## 2. TemplateContract（前端渲染模板扩展）

### 2.1 接口签名（TypeScript）

```typescript
export interface TemplateContract {
  id: 'A' | 'B' | 'C' | 'G' | string;
  // 判定是否能渲染该 graph
  canRender(profile: ArchitectureProfile): boolean;
  // R3F 根组件
  Scene: React.FC<{ graph: ModuleGraph; layout: LayoutResult }>;
  // 教学叙事脚本（Stage-1 结构动画用）
  narrativeScript: NarrativeStep[];
}
```

### 2.2 注册
`frontend/src/templates/registry.ts`:
```typescript
import { templateA } from './llama';
import { templateG } from './generic';
export const templates: TemplateContract[] = [templateA, templateB, templateC, templateG];
```

---

## 3. AnimationLayer（动画精细度扩展）

### 3.1 四层分工

| 层 | 名称 | 职责 | v1.0 | 可叠加 |
|---|---|---|---|---|
| L1 | StructureAnimation | 模块展开/收起、层级过渡 | ✅ | — |
| L2 | DataFlowAnimation | 前向/反向 token 流动 | ✅（子集） | L1 |
| L3 | NumericalHeatmap | Attention 权重、激活值热力 | v1.1 | L1 + L2 |
| L4 | ParallelismAnimation | TP/PP/DP/EP/CP/SP 通信原语 | v1.2 | L1 + L2 |

### 3.2 接口签名

```typescript
export interface AnimationLayer {
  id: 'L1_structure' | 'L2_dataflow' | 'L3_heatmap' | 'L4_parallelism' | string;
  enabled: boolean;
  // 声明式时间轴：返回一组 (startTime, duration, targetNodeId, tween) 元组
  timeline(graph: ModuleGraph, ctx: AnimationContext): TimelineEntry[];
  // 每帧渲染委托
  render(delta: number, state: AnimationState): void;
}
```

### 3.3 约束
- 层间**不得硬编码相互依赖**；协调通过 `AnimationContext` 共享时间轴。
- 每层可独立 toggle，关闭任一层不得破坏其他层。

---

## 4. ParallelismStrategy（并行策略扩展）

### 4.1 接口签名（Python）

```python
class ParallelismStrategy(Protocol):
    id: Literal["TP", "PP", "DP", "EP", "CP", "SP", "TP+PP", "TP+PP+DP", ...]

    def partition(self, graph: ModuleGraph, world_size: int, config: dict) -> PartitionedGraph:
        """声明切分 / 复制规则。"""

    def communication_ops(self, partitioned: PartitionedGraph) -> list[CommOp]:
        """返回通信原语序列：AllReduce / AllGather / ReduceScatter / All2All / P2P。"""

    def animation_script(self, partitioned: PartitionedGraph) -> ParallelismAnimationScript:
        """前端动画脚本（时序 + 设备布局 + 数据流箭头）。"""
```

### 4.2 注册
`backend/parallelism/__init__.py` 追加一行。

### 4.3 验收
- 支持 N-D 组合：TP+PP、TP+PP+DP、TP+PP+DP+EP（DeepSeek-V3 同款）。
- Megatron-LM 论文中的 5 种基础策略 + 3 种组合必须能全部用插件实现。

---

## 5. MemoryEstimator（显存估计器扩展）

### 5.1 接口签名

```python
class MemoryEstimator(Protocol):
    id: Literal["megatron_tp_pp_sp", "fsdp_zero1", "fsdp_zero2", "fsdp_zero3", ...]

    def estimate(
        self,
        graph: ModuleGraph,
        config: TrainingConfig,   # batch / seq_len / dtype / optimizer / grad_accum / ckpt
        parallelism: ParallelismPlan,
        gpu: GPUSpec,             # 来自 gpu-catalog.yaml
    ) -> MemoryBreakdown:
        """返回按类别（weights / gradients / optimizer_states / activations / kv_cache / comm_buffer）
        的 per-device 显存消耗 + 总量 + 与 GPU 容量的占比。"""
```

### 5.2 v1.x 必须支持
- `megatron_tp_pp_sp`（Megatron-LM 3D parallelism，含 SP 激活切分）
- `fsdp_zero1` / `fsdp_zero2` / `fsdp_zero3`（PyTorch FSDP）

---

## 6. GPU Catalog（硬件规格数据表）

### 6.1 文件位置
`backend/data/gpu-catalog.yaml` — **唯一**的 GPU 规格数据源，严禁在代码里硬编码。

### 6.2 Schema

```yaml
- id: a100_80g_sxm
  vendor: nvidia
  arch: ampere
  memory_gb: 80
  memory_bandwidth_gbps: 2039
  fp16_tflops: 312
  bf16_tflops: 312
  fp8_tflops: null          # Ampere 不支持 FP8
  nvlink_gbps: 600
  tdp_w: 400
  release_year: 2020

- id: h100_80g_sxm5
  vendor: nvidia
  arch: hopper
  memory_gb: 80
  memory_bandwidth_gbps: 3350
  fp16_tflops: 989
  bf16_tflops: 989
  fp8_tflops: 1979
  nvlink_gbps: 900
  tdp_w: 700
  release_year: 2022

- id: h200_141g_sxm5
  vendor: nvidia
  memory_gb: 141
  # ...

- id: b200
  vendor: nvidia
  arch: blackwell
  # ...

- id: 4090_24g
  vendor: nvidia
  arch: ada_lovelace
  memory_gb: 24
  # ...

- id: ascend_910b
  vendor: huawei
  memory_gb: 64
  # ...

- id: cambricon_mlu370
  vendor: cambricon
  # ...

- id: kunlun_p800
  vendor: baidu
  # ...
```

### 6.3 v1.x 必须包含（最小集）
NVIDIA：A100-40G/80G、H100-80G、H200-141G、B200、4090-24G、3090-24G、L40S-48G
国产：昇腾 910B、寒武纪 MLU370、昆仑芯 P800/R200

### 6.3.1 null 字段语义（国产卡公开规格不完整的处理，对齐原则 9）

| 字段 | 允许 null | MemoryEstimator 行为 |
|---|---|---|
| `memory_gb` | ❌ 必填 | 缺失 → 条目加载失败，启动阶段 fail-fast |
| `memory_bandwidth_gbps` | ✅ | 估算仍进行；**provenance.caveats** 追加 `"memory_bandwidth_gbps=null，带宽相关预估不可用"` |
| `fp16_tflops` / `bf16_tflops` / `fp8_tflops` | ✅ | 若当前 dtype 对应字段 null → FLOPs/延迟估算跳过并在 caveats 说明；**不得**用其他 dtype 字段代偿 |
| `nvlink_gbps` / `tdp_w` / `release_year` | ✅ | 仅影响展示型 tooltip，不影响计算；UI 显示 "—" |

**原则**：任何 null 都**必须**在该估算结果的 provenance.caveats 中明确指出哪一字段缺失，严禁静默使用默认值/兜底常量（违反原则 9）。

### 6.4 扩展方式
新增 GPU = 在 yaml 追加一条记录，**无需改代码**。前端下拉菜单自动从 `/api/v1/gpus` 拉取。

---

## 7. DataFlowDirection（前向/反向传播扩展）

### 7.1 DataEdge 增强

```python
class DataEdge(BaseModel):
    source: str
    target: str
    edge_type: Literal[
        "data_flow",             # 前向激活
        "residual",
        "skip_connection",
        "branch_merge",
        "gradient_flow",         # 反向梯度（v1.1）
        "activation_checkpoint", # 激活值缓存（v1.1）
        "gradient_accumulation", # 梯度累积（v1.1）
    ]
    direction: Literal["forward", "backward", "bidirectional"]
    tensor_shape: list[int] | None
    provenance: Provenance
```

### 7.2 动画模式
- `forward_only`（v1.0 默认）
- `backward_only`
- `forward_backward_split_screen`（同屏左右对照）
- `forward_backward_overlay`（同画面叠加，反向红色逆放）

---

## 8. Dynamic Config Edit（配置参数动态编辑）

### 8.1 前端 UI
- 所有 config 字段通过 `<ConfigEditor>` 组件暴露（受控组件）
- 可编辑字段白名单：`num_hidden_layers` / `hidden_size` / `num_attention_heads` / `num_key_value_heads` / `intermediate_size` / `num_experts` / `num_experts_per_tok` / `vocab_size` / `max_position_embeddings` / `torch_dtype` / `tp_size` / `pp_size` / `dp_size` / `ep_size` / `micro_batch_size` / `global_batch_size` / `seq_len` / `grad_accum_steps` / `activation_checkpointing`

### 8.2 后端热更新路由
```
PATCH /api/v1/stream/{org}/{repo}/config
Body: { "overrides": { "num_hidden_layers": 64, "tp_size": 4 } }

→ 服务端跳过 HF Hub 与 L1 磁盘缓存
→ 复用已解析的原始 config
→ 合并 overrides → 走 detect → synthesize → estimate → layout
→ WebSocket/SSE 推送新 ModuleGraph snapshot (revision += 1, source="config_override")
```

### 8.3 延迟预算（硬约束）
| 路径 | 预算 |
|---|---|
| 后端 config-only 热更新 | < 200ms（本机） |
| 端到端（含 WS 往返 + 前端重渲染） | < 300ms |
| 前端模块点选高亮 | < 50ms |
| 动画时间轴拖动 scrub | < 16ms/frame |

> 注：本节延迟属于 README 原则 5 的「交互响应延迟」例外条款，是功能正确性硬约束，不因「前期不做性能优化」而豁免。

---

## 9. Pipeline 五阶段（纯函数契约）

| 阶段 | 输入 | 输出 | 可替换 |
|---|---|---|---|
| S1 parse_structure | config + safetensors_header | RawStructure | ✅ |
| S2 detect_features | RawStructure | ArchitectureProfile | ✅ |
| S3 synthesize_flows | ArchitectureProfile + graph | DataEdges | ✅ |
| S4 estimate_resources | graph + TrainingConfig + GPU | MemoryBreakdown + FLOPs | ✅ |
| S5 compute_layout | graph + template_id | LayoutResult | ✅ |

**硬性**：每阶段输入输出完全类型化，无全局状态，无隐式 I/O（meta-device 加载在 S1 前完成）。

---

## 10. 扩展接入验收 Checklist

新增任意扩展点（Adapter / Template / AnimationLayer / ParallelismStrategy / MemoryEstimator / GPU）时，必须同时满足：

- [ ] 新增 ≤ 1 个源文件
- [ ] 注册文件追加 ≤ 1 行
- [ ] 未修改任何核心 pipeline / 渲染循环 / 路由层代码
- [ ] 未在任何既有文件中加入 `if xxx_type == "..."` 分支
- [ ] 对应 schema 字段已走 `revision` 演进（若有契约变更）
- [ ] 至少 1 个单元测试 + 1 个端到端视觉快照
- [ ] 更新本文档对应章节的"v1.x 必须支持"列表

任一未满足 = 架构违规，拒绝合入。
