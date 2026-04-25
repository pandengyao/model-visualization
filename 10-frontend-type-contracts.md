# 十、前端类型契约（TypeScript）

> HF Model Visualizer 技术设计文档 — 章十
>
> **权威性**：本文档是前端 TypeScript 类型的**唯一事实源**。后端 Pydantic 权威源为 [09 §5.1.2](09-backend-detailed-design.md)；API 表层契约以 [04](04-api-design.md) 为准；扩展点契约以 [11](11-extension-points.md) 为准。
>
> **对齐规则**：本文件与 09 §5.1.2 必须 1:1 对应；若二者冲突，**以 09 为准**并在本文件内相应位置追加 `// TODO(10): ...` 注释，由文档维护者回流修复。
>
> **适用范围**：所有前端 `frontend/src/types/**` 必须直接或通过 re-export 使用本文件定义的类型。**禁止**在组件/store/hook 中私造同名类型。

## 10.1 类型对齐原则（对齐原则 3 / 6 / 9）

- **原则 3（结构与数据流 100% 正确）**：前端 TS `interface` 与后端 Pydantic model 严格 1:1 对应，字段名、字段类型、可选性必须一致。前端不得"就地适配"后端字段——任何 UI-only 衍生字段必须在本文件显式声明，并以注释 `// UI-only` 标注。
- **原则 6（≤ 1 文件 + 1 注册）**：新增架构 / 动画层 / 并行策略 / 显存估算器 / GPU 型号时，只允许在本文件追加一个 `interface` + 在 `frontend/src/types/index.ts` 追加一次 re-export。禁止修改现有契约的既有字段形状。
- **原则 9（Provenance 强制）**：任何面向用户的数据结构（`ModuleNode` / `DataEdge` / `ArchitectureProfile` / `MemoryBreakdown` / `EstimateResult` / `ModuleGraph` 根级）**必须**包含 `provenance: Provenance` 字段或以 `WithProvenance<T>` 包裹。裸数据视为契约违反。
- **TypeScript 风格约束**（本文件强制）：
  - 对象形状一律用 `interface`（可扩展、可 merge、便于生成 d.ts）。
  - 未知/可变结构一律用 `Record<string, unknown>`，**禁止 `any`**。
  - 判别联合（discriminated union）用字符串字面量：`'exact' | 'inferred' | 'estimated'`。
  - 后端所有权字段标记 `readonly`；仅前端 PATCH /config 产生的字段可写。

## 10.2 Provenance（原则 9 强制）

> 对齐 04 §4.2.1 + 09 §5.1.2。

```typescript
/**
 * Provenance 置信度三档。字符串值必须与后端 Pydantic `Confidence(str, Enum)` 一致，
 * 供 SSE/WS 透传后直接判等使用。
 */
export type Confidence = 'exact' | 'inferred' | 'estimated';

/**
 * 数据溯源：任何呈现给用户的字段/结构必须挂载 Provenance。
 * - source: 与后端 parsing 层对应；v1.0 可取值见下方字面量联合。
 * - confidence: EXACT/INFERRED/ESTIMATED。
 * - caveats: 需要向用户展示的告警或降级说明（与 `provenance_summary.caveats` 合并显示）。
 */
export interface Provenance {
  readonly source:
    | 'meta_device'
    | 'safetensors_metadata'
    | 'config_json'
    | 'ast_parse'        // v1.1
    | 'config_override'; // PATCH /config 产生
  readonly confidence: Confidence;
  readonly caveats: readonly string[];
}
```

## 10.3 ModuleGraph（3D 渲染唯一数据源）

> 对齐 04 §4.2.2 + 09 §5.1.2。`ModuleGraph` 是 R3F 场景、2D 拓扑、显存估算、动画层的**唯一** graph 输入源。

```typescript
/**
 * 节点粒度。v1.0 仅产出 "block"；"layer" / "op" / "tensor" 为原则 6 结构粒度预留槽位。
 * 对齐 09 §5.1.2 ModuleNode.level（已于 2026-04-25 回填）+ 04 §4.2.2。
 */
export type ModuleLevel = 'layer' | 'block' | 'op' | 'tensor';

/**
 * 边类型（对齐 04 §4.2.2 + 11 §7 DataFlowDirection 扩展点）。
 * v1.0 仅产出 data_flow / residual / skip_connection / branch_merge；
 * gradient_flow / activation_checkpoint / gradient_accumulation 留待 v1.1。
 */
export type EdgeType =
  | 'data_flow'
  | 'residual'
  | 'skip_connection'
  | 'branch_merge'
  | 'gradient_flow'
  | 'activation_checkpoint'
  | 'gradient_accumulation';

/** 方向。v1.0 仅产出 'forward'。 */
export type EdgeDirection = 'forward' | 'backward' | 'bidirectional';

/**
 * 单个模块节点。对应后端 `ModuleNode`。
 * `metadata` 为扩展字段槽（MoE num_experts、quant bits、sliding_window 等），
 * 形状随 ArchitectureAdapter 变化，因此使用 Record<string, unknown>。
 */
export interface ModuleNode {
  readonly id: string;                              // 唯一路径，如 "model.layers.0.self_attn"
  readonly class_name: string;                      // 如 "DeepseekV3Attention"
  readonly level: ModuleLevel;                      // v1.0 = "block"
  readonly params: number;                          // 精确参数量
  readonly dtype: string;                           // "bfloat16" | "int8" | "float16" | ...
  readonly tensor_shapes: Record<string, readonly number[]>;
  readonly metadata: Record<string, unknown>;
  readonly provenance: Provenance;
}

/**
 * 数据边。v1.0 仅产出 direction="forward"。
 * tensor_shape 为 null 时表示形状未知（通常 confidence=INFERRED）。
 */
export interface DataEdge {
  readonly source: string;
  readonly target: string;
  readonly edge_type: EdgeType;
  readonly direction: EdgeDirection;
  readonly tensor_shape: readonly number[] | null;
  readonly provenance: Provenance;
}

/** expand/collapse 层级树。子节点递归引用自身。 */
export interface HierarchyTree {
  readonly id: string;
  readonly children: readonly HierarchyTree[];
}

/**
 * 模型图。3D 渲染、2D 拓扑、动画层、显存估算共享同一份结构。
 * 根级 provenance 聚合整图置信度（09 §5.1.2 ModuleGraph.provenance 已于 2026-04-25 回填，对齐 04 §4.2.2）。
 */
export interface ModuleGraph {
  readonly nodes: Record<string, ModuleNode>;
  readonly edges: readonly DataEdge[];
  readonly hierarchy: HierarchyTree;
  readonly provenance: Provenance;
}
```

## 10.4 ArchitectureProfile

> 对齐 04 §4.2.3 + 09 §5.1.2。驱动 Template A/B/C/G 选型与 2D/3D 布局。

```typescript
/**
 * 前端渲染模板 ID。未识别架构回退到 "G"（非静默回退 LLaMA，对齐原则 8）。
 */
export type TemplateId = 'A' | 'B' | 'C' | 'G';

/**
 * 架构画像。
 * 对齐 09 §5.1.2 ArchitectureProfile.template_id（已于 2026-04-25 回填）+ 04 §4.2.3 + ADR-015。
 */
export interface ArchitectureProfile {
  readonly model_type: string;                       // "llama" | "deepseek_v3" | "llama_moe" | ...
  readonly features: readonly string[];              // ["moe","mla","gqa","sliding_window","quantized",...]
  readonly config_summary: Record<string, unknown>;  // 标准化关键 config（见 09 §5.1.9）
  readonly parsing_layers_used: readonly string[];   // ["config_json","safetensors_metadata","meta_device"]
  readonly template_id: TemplateId;
  readonly provenance: Provenance;
}
```

## 10.5 MemoryBreakdown / EstimateResult

> 对齐 04 §4.2.4。v1.0 是**推理版**：仅 weights + kv_cache + activations。训练版字段（gradients / optimizer_states / comm_buffer）占位预留 `null`。

```typescript
/**
 * 显存细分。v1.0 仅填前三项 + 汇总；后三项（训练版）v1.0 均为 null。
 * 对齐 09 §5.1.2 MemoryBreakdown（已于 2026-04-25 由 dict 升级为结构化类型）+ 04 §4.2.4。
 */
export interface MemoryBreakdown {
  // v1.0 必填
  readonly weights_bytes: number;
  readonly kv_cache_bytes: number;
  readonly activations_bytes: number;
  // v1.1 预留（v1.0 均为 null）
  readonly gradients_bytes: number | null;
  readonly optimizer_states_bytes: number | null;
  readonly comm_buffer_bytes: number | null;
  // 汇总
  readonly per_device_total_bytes: number;
  readonly gpu_capacity_bytes: number;
  readonly utilization_ratio: number;                // per_device_total / gpu_capacity
  readonly provenance: Provenance;
}

/**
 * 估算结果。flops/kv_cache/communication/latency 为后端各估算器的自由输出槽，
 * 前端仅按需消费（数字卡片 / tooltip），不强制结构。
 */
export interface EstimateResult {
  readonly memory: MemoryBreakdown;
  readonly flops: Record<string, unknown>;
  readonly kv_cache: Record<string, unknown> | null;
  readonly communication: Record<string, unknown> | null;  // 并行通信量（v2.0）
  readonly latency: Record<string, unknown> | null;
  readonly provenance: Provenance;
}
```

## 10.6 LayoutResult

> 对齐 04 §4.2.5 + 09 §5.1.2。`positions` 的 value shape 由 S5 compute_layout 决定，前端按 `Position3D` 消费。

```typescript
/** 节点在 3D 场景中的位置与包围尺寸。 */
export interface Position3D {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly width: number;
  readonly height: number;
  readonly depth: number;
}

/**
 * 推荐相机配置。后端 S5 可输出目标点、初始角度、fov 等；
 * 具体字段后端给到何种程度由模板决定，这里保留 Record 兼容。
 */
export interface CameraConfig {
  readonly position?: readonly [number, number, number];
  readonly target?: readonly [number, number, number];
  readonly fov?: number;
  readonly up?: readonly [number, number, number];
  readonly [k: string]: unknown;
}

/** 场景包围盒。 */
export interface SceneBounds {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly center?: readonly [number, number, number];
}

export interface LayoutResult {
  readonly positions: Record<string, Position3D>;
  readonly camera: CameraConfig;
  readonly bounds: SceneBounds;
}
```

## 10.7 Provenance-aware 工具类型

> 为前端组件层 / store 层提供便捷的 Provenance 访问与徽标构造。不对应后端 Pydantic，属于**前端本地派生类型**。

```typescript
/**
 * 为任意对象补全 `provenance` 字段。
 * 新增面向用户的结构时应优先使用 WithProvenance<T>，避免遗漏原则 9。
 */
export type WithProvenance<T extends object> = T & { readonly provenance: Provenance };

/**
 * 抽取类型 T 中所有强制携带 Provenance 的叶子结构。
 * 供编译期校验："所有用户可见数据是否都有 provenance"。
 */
export type RequiresProvenance<T> = T extends { provenance: Provenance } ? T : never;

/**
 * 整次响应的溯源摘要（对齐 04 §4.5 / §4.7 的 `provenance_summary`）。
 * 前端徽标直接消费此对象；不依赖 body 内的分散 Provenance。
 */
export interface ProvenanceSummary {
  readonly layers_used: readonly string[];
  readonly overall_confidence: Confidence;
  readonly caveats: readonly string[];
}
```

## 10.8 Stream 事件协议（SSE + WS）

> 对齐 04 §4.5（SSE）+ §4.7（WebSocket）。SSE `segment` 帧与 WS `graph_update` 帧的 `data` 载荷**字段级 1:1 对齐**，前端 store 可复用同一个 `replaceSnapshot()` 方法。

```typescript
/**
 * 快照版本。
 * - 1: Phase A，来自 config_json + safetensors_metadata（快速响应，~1s 内）。
 * - 2: Phase B，来自 meta_device 增强（2-10s，是最终精确快照）。
 * - ≥ 3: PATCH /config 热更新产生，source = "config_override"。
 */
export type Revision = 1 | 2 | number;

/**
 * SSE / WS 共用的数据体。整体替换，不做增量合并（04 §4.5 第 1 条）。
 */
export interface SnapshotData {
  readonly graph: ModuleGraph;
  readonly profile: ArchitectureProfile;
  readonly estimate: EstimateResult;
  readonly layout: LayoutResult;
}

/**
 * SSE `segment` 帧（04 §4.5）。
 * v1.0 `segment` 固定为 "full"；分段字段保留用于 v1.1+ 细粒度推送。
 */
export interface SseSegmentEvent {
  readonly segment: 'full';                          // v1.1 可扩展为 "config"|"tree"|"params"|"flow"|"estimate"|"layout"
  readonly revision: Revision;
  readonly is_final: boolean;
  readonly source: string;                           // 如 "config_json+safetensors_metadata" | "meta_device" | "config_override"
  readonly data: SnapshotData;
  readonly provenance_summary: ProvenanceSummary;
}

/**
 * SSE `error` 帧 —— 直接是 ProblemDetails 载荷（对齐 04 §4.4 RFC 7807 精简）。
 */
export interface ProblemDetails {
  readonly type: string;                             // "about:blank" 或错误文档 URI
  readonly title: string;
  readonly status: number;
  readonly detail: string;
  readonly instance?: string | null;
  readonly code: ErrorCode;
  readonly fix?: string | null;
}

/** 机器可读错误码（对齐 04 §4.4 枚举表）。 */
export type ErrorCode =
  | 'MODEL_NOT_FOUND'
  | 'ARCH_UNSUPPORTED'
  | 'ADAPTER_NOT_FOUND'
  | 'CONFIG_PARSE_ERROR'
  | 'CONFIG_OVERRIDE_INVALID'
  | 'CONFIG_OVERRIDE_IMPOSSIBLE'
  | 'GPU_SPEC_NOT_FOUND'
  | 'META_LOAD_TIMEOUT'
  | 'HUB_UNAVAILABLE'
  | 'INTERNAL';

/**
 * WebSocket Server → Client 消息（04 §4.7）。
 * v1.0 仅定义 graph_update；v1.1 可能追加 cancel/priority 控制帧的 ack。
 */
export interface WsGraphUpdateMessage {
  readonly type: 'graph_update';
  readonly revision: Revision;
  readonly source: 'config_override' | string;
  readonly data: SnapshotData;
  readonly provenance_summary: ProvenanceSummary;
}

export type WsServerMessage = WsGraphUpdateMessage; // 预留 union 位置

/**
 * 统一下发事件联合（SSE + WS 合流后传入 store）。
 */
export type StreamEvent =
  | ({ readonly channel: 'sse'; readonly event: 'segment' } & SseSegmentEvent)
  | { readonly channel: 'sse'; readonly event: 'error'; readonly payload: ProblemDetails }
  | ({ readonly channel: 'ws' } & WsGraphUpdateMessage);
```

## 10.9 ConfigOverride（PATCH /config 载荷）

> 对齐 04 §4.6。字段类型/范围由后端做**字段级校验**（`CONFIG_OVERRIDE_INVALID`）+ **跨字段校验**（`CONFIG_OVERRIDE_IMPOSSIBLE`）。前端只做最小限定（`number`/`string`/`boolean`）。

```typescript
/**
 * 热更新可编辑字段。所有字段为可选（PATCH 语义），提交时仅包含用户改动项。
 * 字段清单与 04 §4.6 Body 示例 1:1 对齐。新增可编辑字段必须：
 *   1. 先在后端 Pydantic ConfigOverride 新增；
 *   2. 再在本文件追加；
 *   3. CI schema diff 通过后方可合并（见 10.10）。
 */
export interface ConfigOverride {
  // 结构尺寸
  num_hidden_layers?: number;
  hidden_size?: number;
  num_experts?: number;
  num_experts_per_tok?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  moe_top_k?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  torch_dtype?: 'bfloat16' | 'float16' | 'float32' | 'int8' | string;
  // 并行
  tp_size?: number;
  pp_size?: number;
  dp_size?: number;
  ep_size?: number;
  // 训练/推理运行时
  micro_batch_size?: number;
  global_batch_size?: number;
  seq_len?: number;
  grad_accum_steps?: number;
  activation_checkpointing?: boolean;
}

/** PATCH /config 请求体。 */
export interface ConfigOverrideRequest {
  readonly overrides: ConfigOverride;
  readonly session_id: string;  // 由前端生成 uuid，WS 订阅同步使用
}
```

## 10.10 AnimationContext（前端动画层共享时间轴）

> 对齐 05 §5.2.3。属于**前端本地派生类型**（无后端对应 Pydantic）。保留于本文件以保证动画层类型与 `ModuleGraph` 契约同源，防止 05 草案与 10 主干漂移。

```typescript
/**
 * L1-L4 动画层标签（对齐原则 6 动画精细度扩展）。
 * - L1_structure : Stage-1 结构动画
 * - L2_dataflow  : Stage-2 数据流动画（Attention QKV / MoE / Residual）
 * - L3_heatmap   : Stage-3 数值热力（v1.1+）
 * - L4_parallelism: Stage-4 并行策略动画（v1.2+）
 */
export type AnimationLayerId =
  | 'L1_structure'
  | 'L2_dataflow'
  | 'L3_heatmap'
  | 'L4_parallelism';

/**
 * 共享时间轴上下文。
 * 默认 `TTimeline = import('gsap').core.Timeline`；使用方在 import gsap 后以
 * `AnimationContext<gsap.core.Timeline>` 获得完整类型安全，无需 `as` 断言。
 * 不 import gsap 的场景（例如 SSR/测试桩）可回落到 unknown。
 */
export interface AnimationContext<TTimeline = unknown> {
  readonly master: TTimeline;
  readonly layers: Record<AnimationLayerId, {
    readonly enabled: boolean;
    readonly sub: TTimeline;
  }>;
  readonly graph: ModuleGraph;                       // PATCH /config 后重新注入
  readonly now: number;                              // 秒
}
```

## 10.11 对齐校验机制（CI 强制）

> 目的：在 CI 阶段以机器方式保证 10 与 09 §5.1.2 / 04 的 Pydantic 定义字段级一致，消除"前后端类型漂移"类 bug。

- **Phase 1 起**，CI 中新增 `scripts/check-schema-alignment.ts`：
  1. 运行 `python -m backend.scripts.dump_pydantic_schemas > build/schemas.json`（后端导出 JSON Schema）。
  2. 解析本文件导出的 `interface` 集合（借助 `ts-morph` 或 `typescript` compiler API）。
  3. 对 `Provenance` / `ModuleNode` / `DataEdge` / `ModuleGraph` / `ArchitectureProfile` / `MemoryBreakdown` / `EstimateResult` / `LayoutResult` / `ConfigOverride` 做字段级 diff：字段名、字段类型（含联合字面量集合）、`readonly` 属性、可选性。
  4. 任一项不一致 → CI 失败并输出 diff 报告。
- **允许的差异**（白名单）：
  - 前端加 `readonly`（后端无对应概念）。
  - **UI-only 衍生字段**：必须在 `interface` 内以 `// UI-only` 注释标注且后端完全无此字段。
  - 联合字面量集合：前端允许是后端的子集（收紧），但**不允许**是超集（放宽）。
- **破坏性变更流程**：任何在 10 中修改既有字段类型/可选性的改动，必须同时提交 09 §5.1.2 的对应改动；PR 标签 `schema-breaking` 自动 CC 前端 + 后端 owner，并要求 `revision` 字段 +1（对齐 04 §4.5 向前兼容规则 + 原则 6 数据契约稳定性）。

---

[← 返回目录](README.md)
