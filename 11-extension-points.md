# 11 — 扩展点契约（Extension Points）

> 本文档是 README 原则 8「v1.0 架构必须最健壮且可扩展」的**落地契约单一事实源**。
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
    matches: list[str]              # config.model_type 可匹配值列表（仅信息性元数据，detect() 不得依赖此字段做分支判断）
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

    def template_id(self) -> str:  # v1.0 约束为 "A" | "B" | "C" | "G"
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
- **演练**：Phase 2 结束前必须用 Mamba（无 attention）或 ViT（无 causal mask）做一次 cold-start 接入，验证 ≤ 1 文件 + 1 注册。

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

## 3. AnimationLayer（范围约束）

> v1.0 仅实现 L1/L2 具体组件，不在本章定义插件接口。
>
> AnimationLayer 的插件化契约统一迁移到 v1.1+ parking，避免未来能力侵入 v1.0 主契约。

---

## 4. ParallelismStrategy（范围约束）

> v1.0 不实现 ParallelismStrategy 注册表与接口签名。
>
> 并行策略契约统一迁移到 v1.2+ parking/附录文档，不在 v1.0 主契约展开。

---

## 5. MemoryEstimator（显存估计器扩展）

### 5.1 接口签名

```python
class MemoryEstimator(Protocol):
    id: str  # v1.0 仅 "inference_v1"

    def estimate(
        self,
        graph: ModuleGraph,
        config: InferenceConfig,
        gpu: GPUSpec,
    ) -> MemoryBreakdown:
        """返回推理口径显存消耗（weights / kv_cache / activations）与汇总占比。"""
```

#### 5.1.1 InferenceConfig（v1.0）

```python
# v1.0 唯一支持的配置类型
class InferenceConfig(BaseModel):
    micro_batch_size: int  # 与 PATCH /config 白名单字段同名，作为推理 batch 语义
    seq_len: int
    dtype: Literal["float16", "bfloat16", "float32", "int8", "int4"]
    # kv_cache_enabled: bool = True  # v1.0 默认 True
    # v1.0 不含 optimizer / grad_accum / ckpt 字段

# v1.0 不引入 TrainingConfig / ParallelismPlan
```

#### 5.1.2 v1.0 调用约定（唯一实现：`InferenceMemoryEstimator`）

- `id = "inference_v1"`
- `config` 必须为 `InferenceConfig` 实例
- 产出 `MemoryBreakdown` 仅填 `weights_bytes / kv_cache_bytes / activations_bytes + 汇总`
- 对齐 09 §5.1.2 + 04 §4.2.4 + ADR-019

### 5.2 v1.x 必须支持
- v1.0：`inference_v1`（唯一实现）
- v1.1+ parking：`megatron_tp_pp_sp`（Megatron-LM 3D parallelism，含 SP 激活切分）/ `fsdp_zero1` / `fsdp_zero2` / `fsdp_zero3`

---

## 6. GPU Catalog（硬件规格数据表）

### 6.1 文件位置
`backend/data/gpu-catalog.yaml` — **唯一**的 GPU 规格数据源，严禁在代码里硬编码。

### 6.2 Schema

> **注**：以下为 GPU Catalog 格式示例（省略部分字段）。完整规格数据见 09 §5.1.14 的 12 款 GPU 详细定义。实现时以 `backend/data/gpu-catalog.yaml` 实际文件为准。

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

- id: rtx_4090_24g
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
NVIDIA：A100-40G/80G、H100-80G、H200-141G、B200、RTX 4090-24G、RTX 3090-24G、L40S-48G
国产：昇腾 910B、寒武纪 MLU370、昆仑芯 P800/R200

### 6.3.1 null 字段语义（国产卡公开规格不完整的处理，对齐原则 11）

> **v1.0 国产卡最小字段约束**：昇腾 910B / 寒武纪 MLU370 / 昆仑芯 P800/R200 **至少** 填写 `memory_gb` + `bf16_tflops` 两项（显存容量与吞吐是 `InferenceMemoryEstimator` 基础计算所需）；其余字段允许 null，按下表语义走 caveats。

| 字段 | 允许 null | MemoryEstimator 行为 |
|---|---|---|
| `memory_gb` | ❌ 必填 | 缺失 → 条目加载失败，启动阶段 fail-fast |
| `memory_bandwidth_gbps` | ✅ | 估算仍进行；**provenance.caveats** 追加 `"memory_bandwidth_gbps=null，带宽相关预估不可用"` |
| `fp16_tflops` / `bf16_tflops` / `fp8_tflops` | ✅ | 若当前 dtype 对应字段 null → FLOPs/延迟估算跳过并在 caveats 说明；**不得**用其他 dtype 字段代偿 |
| `nvlink_gbps` / `tdp_w` / `release_year` | ✅ | 仅影响展示型 tooltip，不影响计算；UI 显示 "—" |

**原则**：任何 null 都**必须**在该估算结果的 provenance.caveats 中明确指出哪一字段缺失，严禁静默使用默认值/兜底常量（违反原则 11）。

### 6.4 扩展方式
新增 GPU = 在 yaml 追加一条记录，**无需改代码**。前端下拉菜单自动从 `/api/v1/gpus` 拉取。

---

## 7. DataFlowDirection（v1.0）

### 7.1 DataEdge 契约

```python
class DataEdge(BaseModel):
    source: str
    target: str
    edge_type: Literal["data_flow", "residual", "skip_connection", "branch_merge"]
    direction: Literal["forward"]
    tensor_shape: list[int] | None
    provenance: Provenance
```

> 权威 schema 见 [09 §5.1.2](09-backend-detailed-design.md#512-核心数据模型) 与 [10 §10](10-frontend-type-contracts.md)。

### 7.2 动画模式
- `forward_only`（v1.0 默认）
- 反向传播相关模式迁移到 v1.1+ parking

---

## 8. Dynamic Config Edit（配置参数动态编辑）

### 8.1 前端 UI
- 所有 config 字段通过 `<ConfigEditor>` 组件暴露（受控组件）
- v1.0 可编辑字段白名单（8 项）：`num_hidden_layers` / `hidden_size` / `num_experts` / `num_experts_per_tok` / `seq_len` / `micro_batch_size` / `torch_dtype` / `gpu_id`
- 并行字段（`tp_size` / `pp_size` / `dp_size` / `ep_size`）与训练字段（`global_batch_size` / `grad_accum_steps` / `activation_checkpointing`）为 v1.1+/v1.2+ 扩展，启用时通过 `revision += 1` 追加

### 8.2 后端热更新路由
```
PATCH /api/v1/stream/{org}/{repo}/config
Body: { "overrides": { "num_hidden_layers": 64, "torch_dtype": "bfloat16" } }

→ 服务端跳过 HF Hub 与 L1 磁盘缓存
→ 复用已解析的原始 config
→ 合并 overrides → 走 detect → synthesize → estimate → layout
→ 仅通过 WebSocket `/api/v1/stream/{org}/{repo}/updates` 推送新 ModuleGraph snapshot (revision += 1, source="config_override")
```

### 8.3 延迟预算（硬约束）
| 路径 | 预算 |
|---|---|
| 后端 config-only 热更新 | < 200ms（本机） |
| 端到端（含 WS 往返 + 前端重渲染） | < 300ms |
| 前端模块点选高亮 | < 50ms |
| 动画时间轴拖动 scrub | < 16ms/frame |
| 视角切换 / 相机动画 | < 16ms/frame |

> 注：本节延迟属于 README 原则 7 的「交互响应延迟」例外条款，是功能正确性硬约束，不因「前期不做性能优化」而豁免。

---

## 9. Pipeline 五阶段（纯函数契约）

| 阶段 | 输入 | 输出 | 可替换 |
|---|---|---|---|
| S1 parse_structure | config + safetensors_header | ParseResult (ModuleGraph + profile_hint) | ✅ |
| S2 detect_features | ParseResult + config | ArchitectureProfile | ✅ |
| S3 synthesize_flows | ArchitectureProfile + graph | DataEdges | ✅ |
| S4 estimate_resources | graph + EstimateConfig (v1.0=InferenceConfig) + GPU | MemoryBreakdown + FLOPs | ✅ |
| S5 compute_layout | graph + template_id | LayoutResult | ✅ |

**硬性**：每阶段输入输出完全类型化，无全局状态，无隐式 I/O（meta-device 加载在 S1 前完成）。

---

## 10. 扩展接入验收 Checklist

新增任意扩展点（Adapter / Template / MemoryEstimator / GPU）时，必须同时满足（AnimationLayer v1.1+ 插件化后纳入；ParallelismStrategy v1.2+ 定义后纳入）：

- [ ] 新增 ≤ 1 个源文件
- [ ] 注册文件追加 ≤ 1 行
- [ ] 未修改任何核心 pipeline / 渲染循环 / 路由层代码
- [ ] 未在任何既有文件中加入 `if xxx_type == "..."` 分支
- [ ] 对应 schema 字段已走 `revision` 演进（若有契约变更）
- [ ] 至少 1 个单元测试 + 1 个端到端视觉快照
- [ ] 更新本文档对应章节的"v1.x 必须支持"列表

任一未满足 = 架构违规，拒绝合入。
