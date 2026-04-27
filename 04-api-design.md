# 四、API 设计

> [HF Model Visualizer](README.md) 技术设计文档 — 章四

> **文档权威性**：
> - **扩展点契约以 [11-extension-points.md](11-extension-points.md) 为准**（Adapter / Template / AnimationLayer / ParallelismStrategy / MemoryEstimator / GPU Catalog / DataFlowDirection / ConfigEdit / Pipeline 的签名、注册方式、验收标准）
> - **后端实现以 [09-backend-detailed-design.md](09-backend-detailed-design.md) 为准**（pipeline 细节、缓存实现、错误处理内部流程）
> - **本文档（04）是两者之间的 API 表层**：定义 HTTP / WebSocket / SSE 端点契约、对外 Schema、响应头、错误码、交互响应预算。与 11 冲突时以 11 为准；与 09 冲突时以 09 为准。

### 4.1 端点总览

| 方法 | 路径 | 说明 | 权威定义 |
|---|---|---|---|
| GET | `/api/v1/stream/{org}/{repo}` | **主端点**：SSE 流式推送 Phase A/B 快照（revision=1/2），见 §4.5 | 09 §5.1.8 |
| PATCH | `/api/v1/stream/{org}/{repo}/config` | **配置热更新**：原则 7 交互硬约束 + 原则 8 动态编辑，见 §4.6 | 11 §8 |
| WS | `/api/v1/stream/{org}/{repo}/updates` | **热更新推送**：PATCH 后推送新 ModuleGraph snapshot，见 §4.7 | 11 §8 |
| GET | `/api/v1/model/{org}/{repo}` | 一次性 JSON 快照（内部订阅 SSE 并聚合到最终 revision） | 09 §5.2.1 |
| GET | `/api/v1/popular` | 预缓存热门模型列表 | 09 §5.2.1 |
| GET | `/api/v1/gpus` | 返回 `gpu-catalog.yaml` 内容（前端 GPU 选择器数据源） | 11 §6 |
| GET | `/api/v1/architectures` | 已注册 ArchitectureAdapter 列表（扩展点内省） | 11 §1 |
| GET | `/api/v1/memory-estimators` | 已注册 MemoryEstimator 列表（v1.0：推理版 1 条） | 11 §5 |
| GET | `/health/{live,ready,startup}` | 容器 liveness/readiness/startup 检查（非 K8s 专用） | 09 §5.2.1 |

> v1.0 不提供分片端点 `/config`、`/tree`、`/params`——SSE 统一下发单段 `segment="full"`（见 §4.5）；多段 `segment` 仅为 v1.1+ 预留。分片端点可在 v1.1 视需要重新引入。

### 4.2 核心数据模型（与 09 §5.1.2 对齐；Provenance 为全局强制字段）

前端 TypeScript 类型 1:1 对应后端 Pydantic 模型。**唯一权威来源**为 09 §5.1.2。此处仅给出前端必须了解的顶层结构。

#### 4.2.1 Provenance（原则 11 强制字段）

```python
class Confidence(str, Enum):
    EXACT = "exact"           # 来自 meta-device 加载
    INFERRED = "inferred"     # 来自 safetensors 元数据 / config 推断
    ESTIMATED = "estimated"   # 计算估算（显存/FLOPS/延迟）

class Provenance(BaseModel):
    source: str               # "meta_device" | "safetensors_metadata" | "config_json" | "config_override" | "memory_estimator" | "pipeline_aggregate"
    confidence: Confidence
    caveats: list[str] = []
```

**全局强制规则**：所有 schema（`ModuleNode` / `DataEdge` / `ArchitectureProfile` / `MemoryBreakdown` / `EstimateResult` / `ModuleGraph` 根级）**必须**携带 `Provenance` 字段或关联的 Provenance 引用。不允许任何裸数据。

#### 4.2.2 ModuleGraph（3D 渲染唯一数据源）

> 唯一事实源：09 §5.1.2

```python
class ModuleNode(BaseModel):
    id: str                                    # 唯一路径，如 "model.layers.0.self_attn"
    class_name: str                            # 如 "DeepseekV3Attention"
    level: Literal["block"]  # v1.0 固定 block 粒度
    params: int
    dtype: str                                 # "bfloat16" | "int8" | ...
    tensor_shapes: dict[str, list[int]]        # {"q_proj.weight": [7168, 7168], ...}
    metadata: dict                             # moe_num_experts, quant_bits, sliding_window, ...
    provenance: Provenance                     # 强制

class DataEdge(BaseModel):
    source: str
    target: str
    edge_type: Literal[
        "data_flow",
        "residual",
        "skip_connection",
        "branch_merge",
    ]
    direction: Literal["forward"]
    tensor_shape: list[int] | None
    provenance: Provenance                     # 强制

class HierarchyTree(BaseModel):                # 前端 expand/collapse
    id: str
    children: list["HierarchyTree"] = []

class ModuleGraph(BaseModel):
    nodes: dict[str, ModuleNode]
    edges: list[DataEdge]
    hierarchy: HierarchyTree
    provenance: Provenance                     # 根级 provenance，刻画整图的可信度
```

> v1.1+ 的枚举扩展在对应版本契约文档中单独定义，不在 v1.0 本章预置。

#### 4.2.3 ArchitectureProfile（驱动 3D 布局与视觉编码）

```python
class ArchitectureProfile(BaseModel):
    model_type: str                            # "llama" | "deepseek_v3" | "llama_moe" | ...
    features: list[str]                        # ["moe", "mla", "gqa", "sliding_window", "quantized", ...]
    config_summary: dict                       # 标准化后的关键 config（见 09 §5.1.9）
    parsing_layers_used: list[str]             # ["config_json", "safetensors_metadata", "meta_device"]
    template_id: Literal["A", "B", "C", "G"]   # 原则 10：未识别 → "G" 通用回退（非静默回退 LLaMA）
    provenance: Provenance
```

#### 4.2.4 MemoryBreakdown / EstimateResult

> **v1.0 定位**：`MemoryEstimator` v1.0 实现的是**推理版**：`weights` + `kv_cache` + `activations`，**不含**训练显存（`gradients` / `optimizer_states` / `comm_buffer`）。训练版（Megatron / FSDP）留给 v1.1，接口已在 11 §5 定义好。

```python
class MemoryBreakdown(BaseModel):
    # v1.0 必填
    weights_bytes: int
    kv_cache_bytes: int
    activations_bytes: int
    # 汇总
    per_device_total_bytes: int
    gpu_capacity_bytes: int
    utilization_ratio: float                   # per_device_total / gpu_capacity
    provenance: Provenance                     # 强制

class EstimateResult(BaseModel):
    memory: MemoryBreakdown
    flops: dict
    kv_cache: dict | None
    provenance: Provenance                     # 强制
```

**MemoryEstimator 请求语义**：v1.0：ModuleGraph + InferenceConfig + GPUSpec → MemoryBreakdown（ParallelismPlan 传 None）；v1.1+：扩展为 EstimateConfig + ParallelismPlan。结果并入主 SSE 响应的 `data.estimate` 部分（见 §4.5），不设独立端点。

#### 4.2.5 LayoutResult

```python
class LayoutResult(BaseModel):
    positions: dict[str, dict]                 # node_id → {x, y, z, width, height, depth}
    camera: dict
    bounds: dict
    provenance: Provenance                     # confidence = ESTIMATED
```

**不再使用的旧模型**：`TreeNode`、`FlowStep`、`ParamStats`、`KeyConfig`、`MoEInfo/MLAInfo/QuantInfo`、`SubConfig` —— 已由 `ModuleGraph` + `ArchitectureProfile` + 节点 `metadata` + 边 `edge_type` 替代。迁移映射见 09 §5.1.2 / 09 §5.2 `from_stage_results()`。

### 4.3 响应头约定

| Header | 适用 | 说明 |
|---|---|---|
| `X-Provenance-Summary` | 全部主路径响应 | 形如 `layers_used=config_json,safetensors_metadata;confidence=inferred`，便于前端徽标不依赖 body |
| `X-Revision` | SSE / WS / PATCH 回显 | 当前快照 revision 单调序号 |
| `Cache-Control: no-cache, no-transform` | SSE | 禁止中间代理缓冲 |
| `X-Accel-Buffering: no` | SSE | Nginx 关闭 gzip/缓冲 |

### 4.4 统一错误契约（RFC 7807 Problem Details 精简版）

```python
class ProblemDetails(BaseModel):
    type: str                                  # "about:blank" 或错误文档 URI
    title: str                                 # 人类可读标题
    status: int                                # HTTP 状态码
    detail: str                                # 具体消息
    instance: str | None = None                # 本次请求唯一 id（日志关联）
    code: str                                  # 机器可读错误码（见下）
    fix: str | None = None                     # 用户可操作的修复建议
```

**错误码枚举**：

| 错误码 | HTTP | 含义 |
|---|---|---|
| `MODEL_NOT_FOUND` | 404 | model_id 在 HF Hub 不存在 |
| `ARCH_UNSUPPORTED` | 422 | 仅用于极端异常；**常规路径应回退 Template G** |
| `ADAPTER_NOT_FOUND` | 500 | 无对应 ArchitectureAdapter 且 Template G 回退链路亦失败（极端异常） |
| `CONFIG_PARSE_ERROR` | 422 | config.json 解析失败 |
| `CONFIG_OVERRIDE_INVALID` | 400 | PATCH /config 参数超出字段级约束（类型/范围） |
| `CONFIG_OVERRIDE_IMPOSSIBLE` | 422 | 跨字段约束冲突（如 `num_experts_per_tok > num_experts`） |
| `GPU_SPEC_NOT_FOUND` | 404 | `gpu_id` 不在 `gpu-catalog.yaml` 中 |
| `META_LOAD_TIMEOUT` | 504 | meta-device 加载超时 |
| `HUB_UNAVAILABLE` | 503 | HF Hub 不可达且 L1 无缓存 |
| `INTERNAL` | 500 | 其他未分类异常 |
| `TRUST_REMOTE_CODE_BLOCKED` | 422 | `TRUST_REMOTE_CODE=false` 时模型含自定义代码 |
| `SCHEMA_VALIDATION_FAILED` | 500 | 后端产出不符 Pydantic 契约 |
| `SESSION_EXPIRED` | 410 | SSE/WS session 超时 |
| `SNAPSHOT_NOT_READY` | 409 | revision=1 期间 PATCH 请求被拒 |

> **原则 1（非商业化）落地**：**不**提供 `RATE_LIMITED` / `ADMISSION_REJECTED` / `CIRCUIT_OPEN` / 配额 / 计费 / 套餐相关错误码。内部工具无需防刷。

### 4.5 SSE 主流式协议（与 09 §5.1.8 对齐）

> 本节为 SnapshotData 类型（SSE segment 帧 data 字段）的唯一事实源。

**端点**：`GET /api/v1/stream/{org}/{repo}`，响应头 `Content-Type: text/event-stream`、`Cache-Control: no-cache, no-transform`、`X-Accel-Buffering: no`。

**帧格式**（每帧带 `id:` 单调序号、`event:` 类型、`data:` JSON）：

```
id: 1
event: segment
data: {
  "segment": "full",
  "revision": 1,
  "is_final": false,
  "source": "config_json+safetensors_metadata",
  "data": {
    "graph": { "nodes": {...}, "edges": [...], "hierarchy": {...}, "provenance": {...} },
    "profile": { "model_type": "deepseek_v3", "features": ["moe","mla","gqa"], "template_id": "C", ... },
    "estimate": { "memory": {...}, "flops": {...}, "provenance": {...} },
    "layout": { "positions": {...}, "camera": {...} }
  },
  "provenance_summary": {
    "source": "pipeline_aggregate",
    "layers_used": ["config_json", "safetensors_metadata"],
    "overall_confidence": "inferred",
    "caveats": ["meta-device 加载进行中"]
  }
}

: keep-alive         ← 每 15s 心跳

id: 2
event: segment
data: {
  "segment": "full",
  "revision": 2,
  "is_final": true,
  "source": "meta_device",
  "data": { ...更精确的 graph 与 estimate... },
  "provenance_summary": {
    "source": "pipeline_aggregate",
    "layers_used": ["config_json", "safetensors_metadata", "meta_device"],
    "overall_confidence": "exact",
    "caveats": []
  }
}
```

**终止事件**：

```
event: error
data: { "type":"about:blank","title":"Meta load timeout","status":504,"code":"META_LOAD_TIMEOUT",... }
```

**关键规则**：

1. **段即完整快照**：`revision=1` 即可完整渲染 3D 场景；常规路径下 `revision=2` 整体替换视图（不做增量合并）。
2. **is_final=true** 后服务端关闭连接；异常降级时允许 `revision=1` 即为最终帧（`is_final=true`）。
3. **重连**：客户端保留 `Last-Event-ID`；v1.0 断线直接重连，服务端返回最新完整快照。（v1.1+ 可引入 ring buffer 补发机制）
4. **不用于进度条**：进度条由前端依 revision / layers_used 自行计算，不走独立事件。

#### 4.5.1 revision=1 最小可渲染字段集（首屏契约）

> 对齐原则 3「结构 100% 正确」+ 原则 7「交互硬约束」：`revision=1` 的字段集是契约，不是尽力而为。

**必填字段**（缺任一 → 前端不得尝试渲染，视为错误态）：

| 字段 | 说明 | 来源 | confidence |
|---|---|---|---|
| `data.graph.nodes` | Block 层级节点（v1.0 最细粒度为 Block；Op/Tensor 槽位保留但可空） | config.num_hidden_layers + safetensors 扫描 | EXACT/INFERRED |
| `data.graph.hierarchy` | 节点父子关系树（模型→层组→Block） | 结构推断 | INFERRED |
| `data.graph.provenance` | 根级 Provenance（原则 11 强制） | pipeline 聚合 | — |
| `data.profile.template_id` | `"A" \| "B" \| "C" \| "G"`（ADR-015） | Adapter 路由 | EXACT/INFERRED |
| `data.profile.model_type` | HF `model_type` 字段原值 | config.json | EXACT |
| `data.layout.positions` | 每个节点的 `[x,y,z]` **初步布局**（网格/环形，由 LayoutStrategy v1 实现） | LayoutEngine | ESTIMATED |
| `data.layout.camera` | 相机初始 `{position, target, fov}` | 根据 scene bounds 计算 | ESTIMATED |

**revision=1 允许退化**（待 revision=2 增强）：
- `data.graph.edges`（数据流边） — revision=1 可为空数组 `[]`，revision=2 增强
- `data.estimate`（内存/FLOPS） — revision=1 必须存在，但允许为快速估算值（`confidence=ESTIMATED`）
- `data.profile.features` — revision=1 可为初步集合，revision=2 完善

**前端行为契约**：
- revision=1 到达 → 立即渲染节点 + 初步布局 + 相机（**不等 edges/estimate**）
- revision=2 到达 → 原地替换：添加 edges 动画、填充 estimate 面板、profile 徽标补全
- 若 revision=1 的 `layout.positions` 缺失或为空 → 前端按 **fallback layout**（均匀栅格）渲染，同时在 UI 顶部显示 "布局计算中" 提示；此 fallback 不进入视觉回归快照

> **为何首屏必须含 layout**：若前端自行临时布局，不同渲染环境（浏览器/分辨率/DPR）会产生不一致结果，违反原则 3。layout 由后端 LayoutEngine 统一计算是契约。

#### 4.5.2 revision=1 期间 UI 交互契约（GPU Selector / PATCH）

> 对齐原则 7「交互硬约束」+ 原则 3「结构正确」。revision=1 与 revision=2 间隔通常 < 1s。

| UI 元件 | revision=1（estimate 为快速估算） | revision=2 到达后 |
|---|---|---|
| `<GPUSelector>` | **可见但默认禁用**（`disabled` 置灰 + tooltip "显存估算加载中…"） | 启用，正常交互 |
| `<ConfigEditor>` 字段 | **默认只读展示**当前 config（来自 revision=1 的 `profile.config_summary`） | 启用，按 §4.6 正常 PATCH |
| `<EstimatePanel>` | 显示快速估算值（可附“精算中”提示） | 用 revision=2 精确值替换 + provenance 徽标 |
| 点选节点 / tooltip | **允许**（只读结构信息） | 追加显存/FLOPs 面板 |

**实现要点**：
- 前端 `canInteract = (is_final === true) || (revision >= 2)` 作为 PATCH / GPU 选择的唯一开关
- 若 `revision=1 && is_final=true`（降级终态），前端立即解除 GPUSelector/ConfigEditor 禁用
- 超过 3s 仍停留 revision=1 且 `is_final=false` → UI 顶部提示 "首屏已就绪，估算耗时较长"

### 4.6 PATCH /config —— 配置热更新（原则 7 硬约束 + 原则 8 动态编辑）

> 本节为 PATCH /config 响应格式的唯一事实源。

```
PATCH /api/v1/stream/{org}/{repo}/config
Content-Type: application/json

Body:
{
  "overrides": {
    "num_hidden_layers": 64,
    "hidden_size": 8192,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "seq_len": 4096,
    "micro_batch_size": 1,
    "torch_dtype": "bfloat16",
    "gpu_id": "a100_80g_sxm"
  },
  "session_id": "uuid"
}

→ 202 Accepted（立即返回，不阻塞；body 可包含 `{"estimated_wait_ms": ...}`）
```

**后端处理链路**（11 §8 为契约权威源）：

1. 跳过 HF Hub 加载 + L1 磁盘缓存；
2. 复用该 session 已解析的原始 config；
3. 合并 `overrides` → 走 S2 detect → S3 synthesize → S4 estimate → S5 layout（纯函数管线，见 11 §9）；
4. 通过 WebSocket（见 §4.7）向 `session_id` 推送新 `ModuleGraph` 快照，`revision += 1`，`source="config_override"`。

**交互响应预算（硬约束，见 §4.8）**：后端 < 200ms、端到端 < 300ms。

> **v1.0 字段白名单**（8 项核心）：num_hidden_layers / hidden_size / num_experts / num_experts_per_tok / seq_len / micro_batch_size / torch_dtype / gpu_id。
> 并行字段（tp/pp/dp/ep_size）和训练字段（global_batch_size / grad_accum_steps / activation_checkpointing）为 v1.1+/v1.2+ 扩展范围，v1.0 契约不暴露，待对应能力启用时随 `revision += 1` 扩展。

**字段级校验**：超出字段类型/范围 → `400 CONFIG_OVERRIDE_INVALID`；跨字段冲突（如 `num_experts_per_tok > num_experts`）→ `422 CONFIG_OVERRIDE_IMPOSSIBLE`。校验失败不推 WS 事件。

### 4.7 WebSocket 推送 —— 热更新通道

```
WS /api/v1/stream/{org}/{repo}/updates?session_id={uuid}
```

**架构约束**：SSE 仅用于冷启动的分段推送；PATCH /config 热更新推送**必须**采用 WebSocket（不得复用 SSE），原因：
- 双向低延迟，便于 v1.1+ 加入 client → server 控制帧（如 cancel / priority）；
- 与冷启动 SSE 流独立订阅，`session_id` 作为路由键将同一 session 的所有 PATCH 结果回推到同一 WS 连接；
- 冷启动与热更新通道解耦，语义清晰。

**Server → Client 消息**：

```json
{
  "type": "graph_update",
  "revision": 3,
  "source": "config_override",
  "data": { "graph": {...}, "profile": {...}, "estimate": {...}, "layout": {...} },
  "provenance_summary": {
    "layers_used": ["config_json", "config_override"],
    "overall_confidence": "inferred",
    "caveats": ["参数由用户覆盖，非源自 HF Hub"]
  }
}
```

**消息结构与 SSE `segment` 帧的 `data` 字段 1:1 对齐**（同一个 ModuleGraph / EstimateResult schema），前端 store 可复用 `replaceSnapshot()` 逻辑。

#### 4.7.1 Session 生命周期（对齐原则 7 / 原则 10）

> v1.0 简化为基础连接管理；细粒度 ping/timeout/重复检测/并发上限等机制视 v1.1 需要再引入。
>
> 唯一事实源：09 §5.1.18

| 事件 | 行为 |
|---|---|
| **建立** | 前端通过 `crypto.randomUUID()` 生成 `session_id`（UUIDv4），以此为 query 开启 WS，并在 PATCH /config 请求体中携带同一 `session_id`（见 §4.6 / §4.11） |
| **页面离开** | `window.beforeunload` → 前端发 close（code=1001）；后端清理订阅键 |
| **session_id 重复订阅** | 同一 session_id 再次 WS 连接 → 服务端**替换旧连接**（code=4409 "duplicate session replaced"），新连接接管 revision 序列（与 09 统一） |
| **重连策略** | WS 断开后前端指数退避重连（200ms → 400ms → 800ms → 最多 5 次）；携带 `?last_revision=N` query；v1.0 断线直接重连获取最新快照，服务端若 session 已过期则返回 close code=4426 "session not found"，前端必须重新 GET /stream |
| **revision 单调** | 同一 session 内 `revision` 严格单调递增；前端收到 revision ≤ 本地最大值的消息 → 丢弃（防乱序） |
| **Last-User-Intent 头** | §4.5.2 提到的意向选择通过 WS 第一帧 `client_hello` 上报：`{"type":"client_hello","last_intent":{"gpu_id":"..."}}`；服务端据此在下一次 revision 中附带对应 estimate |

**资源上限（v1.0）**：session 状态仅存内存（不引入 Redis，对齐原则 1）。

#### 4.7.2 Revision 竞态处理规则（SSE × WebSocket × PATCH）

> 对齐原则 3「结构正确」+ 原则 7「首屏可交互」。revision=1 / 2 / ≥3 分别对应 SSE Phase A / SSE Phase B / PATCH 热更新（见 10 §Revision 类型）。以下规则覆盖所有交叉时序。

**T1. revision 全局单调**
- 同一 session 的 SSE / WS 消息共享一个 revision 计数器，严格单调递增
- 前端维护 `lastAppliedRevision`；收到 revision ≤ 本地最大 → 丢弃（§4.7.1 已定义）

**T2. 用户在 revision=1 与 revision=2 之间发起 PATCH**
- 前端侧：按 §4.5.2 contract，revision=1 期间 `<ConfigEditor>` / `<GPUSelector>` **禁用**，PATCH 不可能被触发
- 后端侧：若客户端仍绕过 UI 发出 PATCH（手工 curl），服务端返回 `409 Conflict`，问题码 `SNAPSHOT_NOT_READY`，body `{detail: "wait for revision>=2"}`

**T3. Phase B 期间收到 PATCH**
- 后端**不取消** Phase B；让 Phase B 正常完成推 revision=2 后，再串行处理 PATCH 产生 revision=3
- 对齐 §4.5「段即完整快照」，保证每个 revision 都是完整结果

**T4. 快速连续多次 PATCH**
- 前端 300ms debounce（05 §5.7）合并为单次 PATCH；后端同一时刻仅处理一个 PATCH，新 PATCH 到达时取消前一个未完成的 PATCH 计算
- 已产出 revision 的不回滚

> **T5/T6 断连/重连**：v1.0 依赖前端 300ms debounce + 后端单槽 PATCH 即可满足 ~10 用户场景。复杂的断连队列、Last-User-Intent 头持久化等机制推迟到 v1.1 视实际需要引入。

---

### 4.8 交互响应预算（硬约束，Phase 1 起必达）

> 唯一事实源：README §交互响应预算
>
> 对齐 README 原则 7 例外条款与 11 §8.3。交互延迟属功能正确性，不因"前期不做性能优化"而豁免。

| 路径 | 预算 | 类型 |
|---|---|---|
| PATCH /config 后端处理（接收 → WS 推送离开服务端） | **< 200ms** | 硬约束 |
| PATCH /config 端到端（前端发起 → 3D 重渲染可见） | **< 300ms** | 硬约束 |
| WebSocket 单条消息延迟（本机） | < 50ms | 观测指标 |
| 模块点选 / 悬停高亮 | **< 50ms** | 硬约束 |
| 动画时间轴拖动（scrub） | **< 16ms/frame（60fps）** | 硬约束 |
| 视角切换 / 相机动画 | **< 16ms/frame** | 硬约束 |
| L0 命中 P95 | < 10ms | 观测指标 |
| L1 命中 P95 | < 50ms | 观测指标 |
| HF Hub 冷启动 P95 | < 3s | 观测指标 |

### 4.9 扩展点内省端点

为支撑原则 8「可扩展架构」的可观测性与前端选择器的自动填充，提供以下只读端点。返回内容来自后端对应 registry 的实时枚举。

#### 4.9.1 `GET /api/v1/gpus`

返回 `backend/data/gpu-catalog.yaml` 的全部条目（schema 见 11 §6.2）。前端 GPU 选择器数据源。新增 GPU = 改 yaml，无需改代码，无需改 04。

```json
[
  { "id": "a100_80g_sxm", "vendor": "nvidia", "memory_gb": 80, "bf16_tflops": 312, ... },
  { "id": "h100_80g_sxm5", "vendor": "nvidia", "memory_gb": 80, "bf16_tflops": 989, ... },
  ...
]
```

未知 `gpu_id` 请求 → `404 GPU_SPEC_NOT_FOUND`。

#### 4.9.2 `GET /api/v1/architectures`

返回当前已注册的 `ArchitectureAdapter` 列表（11 §1）。

```json
[
  { "name": "llama",         "matches": ["llama", "llama2", "llama3"], "template_id": "A", "confidence": "exact" },
  { "name": "llama_moe",     "matches": ["mixtral", "llama_moe"],       "template_id": "B", "confidence": "exact" },
  { "name": "deepseek_moe",  "matches": ["deepseek_v2", "deepseek_v3"], "template_id": "C", "confidence": "exact" },
  { "name": "generic",       "matches": ["*"],                          "template_id": "G", "confidence": "inferred" }
]
```

#### 4.9.3 `GET /api/v1/memory-estimators`

返回已注册 `MemoryEstimator` 列表（11 §5）。

```json
[
  { "id": "inference_v1", "scope": "inference", "supports": ["weights", "kv_cache", "activations"] }
]
```

> v1.0 仅 `inference_v1`；v1.1 将追加 `megatron_tp_pp_sp` / `fsdp_zero1/2/3`。

> 并行策略相关 API 设计统一迁移到 v1.2+ parking，不在 v1.0 API 主文档展开。
### 4.10 缓存策略（v1.0 两段 + L0/L1 + SHA 键）

```
两层 + 两段 TTL（权威定义见 09 §6.3）:

L0: 进程内 TTLCache (maxsize=128, 线程锁保护)
    — fast_snapshot / full_snapshot 两段
    — 每段独立 TTL

L1: 文件系统 JSON
    — cache_key = f"{model_id}:{resolved_commit_sha}"   (SHA 进 key，消除强推后读到旧数据)
    — 原子写：tempfile + os.replace
    — fast/full 两段独立 TTL
```
数据源: HF Hub API (+ huggingface_hub 本地缓存)
    — 注意：HF Hub 不是缓存层，是最终数据源

并发保护: single-flight（按 (model_id, sha) 聚合并发冷读），避免 N 次 meta-load。

PATCH /config 例外: 跳过 L0/L1 与 HF Hub，直接复用内存中已解析的原始 config + overrides。
```

#### 4.10.1 HF Hub 降级

| 状态 | 处理 |
|---|---|
| 正常 | 写 L0 + L1，返回 `revision=2, is_final=true` |
| 限流 429 | 返回 L1 缓存 + `caveats=["数据可能非最新"]` |
| 5xx | 返回 L1 缓存 + `caveats=["HF Hub 暂时不可用"]` |
| L1 也无 | 503 + `ProblemDetails(code="HUB_UNAVAILABLE")` |

启动时异步预热 Top-100 热门模型到 L0 + L1。

> **config-only 降级**（meta-device 未能加载）：`Provenance.confidence` 降为 `INFERRED`；`ModuleGraph.nodes[*].params` 为估算值；部分 `tensor_shapes` 可能缺失。前端凭 `provenance_summary.overall_confidence` 显示徽标。

### 4.11 前端集成示例

```typescript
// —— 主流 SSE ——
const es = new EventSource(`/api/v1/stream/${encodeURIComponent(org)}/${encodeURIComponent(repo)}`);
es.addEventListener('segment', (e) => {
  const msg = JSON.parse(e.data);
  useModelStore.getState().replaceSnapshot(msg);      // 整体替换，不合并
  if (msg.is_final) es.close();
});
es.addEventListener('error', (e) => {
  const problem = JSON.parse(e.data);
  useModelStore.getState().setError(problem);
  es.close();
});

// —— 配置热更新（原则 7 硬约束）——
const sessionId = crypto.randomUUID();
const ws = new WebSocket(
  `/api/v1/stream/${encodeURIComponent(org)}/${encodeURIComponent(repo)}/updates?session_id=${sessionId}`
);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'graph_update') {
    useModelStore.getState().replaceSnapshot(msg);
  }
};

async function patchConfig(overrides: Partial<ConfigOverrides>) {
  await fetch(`/api/v1/stream/${org}/${repo}/config`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ overrides, session_id: sessionId }),
  });
  // 无需等待：202 立即返回，更新经 WS 抵达
}
```

### 4.12 安全规则

> **原则 1（非商业化、内部工具）**：v1.0 **不做** public-launch 级安全硬化。不设匿名 rate limit、SSO、CAPTCHA、reCAPTCHA、配额、计费、多租户隔离。
>
> 仅保留以下最小自卫项（不视为硬约束，属于良好实践）：
> - `trust_remote_code` 默认 `True`（详见 09 §2 / 06 P0-13）：Path A 启用远程自定义代码产出 `EXACT` 徽标；可通过 `TRUST_REMOTE_CODE=false` 降级至 Path B（Template G + `INFERRED` 徽标 + caveats 标注）
> - 下载大小上限（config ≤ 256KB、README ≤ 2MB、safetensors header-only ≤ 16MB），避免误操作打爆本地磁盘
> - `model_id` 正则收紧：`^(?![.-])[A-Za-z0-9][A-Za-z0-9._-]{0,95}/(?![.-])[A-Za-z0-9][A-Za-z0-9._-]{0,95}$`
>
> 若未来改为公网开放，再按需补齐 SSRF 防护、速率限制、404 缓存等——届时走新 ADR，不在本版本承诺。

---

[← 返回目录](README.md)
