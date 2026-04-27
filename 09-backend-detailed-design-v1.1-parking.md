# 九(v1.1+)、后端详细设计 — Parking Lot

> HF Model Visualizer — v1.1+ 候选内容存档
>
> **本文件不属于 v1.0 交付范围**。从 09-backend-detailed-design.md 剥离出的 v1.1+ 内容集中于此，便于未来迭代时复用。
>
> **权威性**：v1.0 实现以 09-backend-detailed-design.md（精简版）为准；本文件仅作为未来规划参考。
>
> 当 v1.0 GA 后启动 v1.1 规划时，应按以下清单回取对应章节内容，并与彼时的 04 / 08 / 11 重新对齐。

---

## 目录（从 09 迁移）

- [v1.1+ PARKING] 需求表补充：Req #6/#7/#8（扩散/VLA/世界模型检测）、Req #22（AST 源码解析）
- [v1.1+ PARKING] 名词/术语：Template D/E/F 分类
- [v1.1+ PARKING] §5.2.11 `services/parallel/` — 并行策略模块
- [v1.1+ PARKING] §5.3.4 Diffusers 模型解析流程
- [v1.1+ PARKING] §5.3.6 并行策略计算流程
- [v1.1+ PARKING] §5.3.7 PP 调度器流程
- [v1.1+ PARKING] §7.4 多模型对比接口（POST /api/v1/compare 的详细 schema）
- [v1.1+ PARKING] §7.5 并行策略接口（GET /api/v1/models/{id}/parallel）
- [v1.1+ PARKING] §7.11 Rate Limiting 参考设计（原则 1：v1.0 不启用）
- [v1.1+ PARKING] 已废弃 schema 名（TreeNode / FlowStep / ParamStats / KeyConfig / MoEInfo / MLAInfo / QuantInfo / SubConfig）迁移说明

---

## [v1.1+ PARKING] 需求表补充（对应 09 §二）

以下需求项曾被早期文档标注为 P0/P1，但按 v1.0 冻结范围（README + 04 + 08 + 11）应推迟到 v1.1+：

| # | 需求点 | 原始优先级 | v1.0 决议 | 目标代码路径 | 关键函数/类 | 前置依赖 | 版本 | 复杂度 |
|---|-------|-----------|-----------|-------------|------------|---------|------|-------|
| 6 | 扩散模型结构检测（UNet/DiT、VAE、文本编码器、调度器、ControlNet） | P1 | 推迟至 v1.1+ | `[新增] services/detectors/diffusion.py` | `detect_denoiser_type()`, `detect_vae()`, `detect_text_encoders()`, `detect_scheduler()`, `detect_controlnet()` | #1, #2, #9 | v1.1 | H |
| 7 | VLA 结构检测（动作头、本体感知编码器、动作 tokenizer） | P1 | 推迟至 v1.1+ | `[新增] services/detectors/vla.py` | `detect_action_head()`, `detect_proprio_encoder()`, `detect_action_tokenizer()` | #1, #2, #9 | v1.1 | M |
| 8 | 世界模型结构检测（帧编码器、动力学模型、时序建模、动作条件注入） | P1 | 推迟至 v1.1+ | `[新增] services/detectors/world_model.py` | `detect_frame_encoder()`, `detect_dynamics_model()`, `detect_temporal_modeling()` | #1, #2, #9 | v1.1 | M |
| 22 | transformers 模型结构源码解析（AST 解析 `modeling_*.py`） | P0（错标） | 推迟至 v1.1+ | `[新增] services/pipeline/ast_parser.py (v1.1)` | `inspect_model_class(config) → ModuleHierarchy`, `extract_forward_chain()`, `map_parallel_points()` | #1 | v1.1 | H |

**v1.0 替代策略**：以上能力在 v1.0 均不交付，`ArchitectureAdapter` 在无法识别时走 `GenericAdapter`（Template G，`INFERRED` 徽标）；AST 解析以 meta-device + safetensors 双路径替代。

---

## [v1.1+ PARKING] 术语/架构广度 — Template D/E/F

v1.0 架构模板按原则 10 冻结为 **A/B/C/G** 四种（对齐 ADR-015）。早期文档曾描述 7 种模板（A-G），其余三种并入本 parking：

| 模板 | 对应架构族（示例） | v1.0 决议 |
|---|---|---|
| D | GPT-2 style（全 Post-Norm + GELU + learned PE） | v1.1+：若发现命中率显著，再独立拆出；否则继续走 Template G |
| E | Hybrid Mamba / Jamba（attention + SSM 混合） | v1.1+：需引入 SSMDetector（见 09 §5.2 补充修订 #24） |
| F | Encoder-Decoder（T5 / Flan-T5 / BART） | v1.1+：需扩展 DataFlowDirection 与 ModuleGraph 双分支表示 |

**v1.0 兜底规则**：任何不满足 Template A 三特征（RoPE + RMSNorm + SwiGLU/GatedMLP）的模型一律走 Template G（见 09 §5.1.20），**不得**默认回退至 Template A（原则 10）。

---

## [v1.1+ PARKING] §5.2.11 `services/parallel/` — 并行策略模块

> 对应 v1.0 范围：09 **不保留** `ParallelismStrategy` 接口与 registry（参见 11 §4）。以下为 v1.2+ 预研草案。

| 文件 | 职责 |
|------|------|
| `megatron.py` | Megatron-LM 策略：TP（Attention/MLP 切分）、PP（层分配）、DP、SP（序列并行）、CP（上下文并行）、EP（专家并行） |
| `fsdp.py` | FSDP 策略：参数分片、梯度分片、通信模式 |
| `pp_scheduler.py` | PP 调度器模拟：1F1B、Interleaved 1F1B、Zero-bubble（生成每个 micro-batch 的时间步序列） |

**接口统一**：

```python
class ParallelStrategy(ABC):
    @abstractmethod
    def partition(self, graph: ModuleGraph, config: ParallelConfig) -> PartitionedGraph:
        """将 ModuleGraph 按并行策略切分，每个节点标记所属 rank/device"""
        ...

    @abstractmethod
    def communication_pattern(self, partitioned: PartitionedGraph) -> list[CommOp]:
        """生成通信操作序列（All-reduce, P2P 等）"""
        ...
```

---

## [v1.1+ PARKING] §5.3.4 Diffusers 模型解析流程

```
入口判断:
  if "_class_name" in config:  # Diffusers pipeline 标志
      → diffusers_parser

diffusers_parser 流程:
  1. 识别 pipeline 类型
     class_name = config["_class_name"]  # e.g. "StableDiffusionXLPipeline"

  2. 找到 denoiser 组件
     denoiser = find_denoiser(config)
     # 按 class_name 匹配: UNet2DConditionModel / Transformer2DModel / ...
     # 不硬编码 "unet" key

  3. 并行 fetch 所有组件 config  # [P1-27]
     components = [k for k, v in config.items() if isinstance(v, list) and len(v) == 2]
     component_configs = await asyncio.gather(*[
         hf_client.fetch_config(repo_id, subfolder=name)
         for name in components
     ])  # 5×350ms → ~350ms

  4. 按组件解析 (非递归 admission)  # [P0-2]
     # 所有组件打包为单个 sandbox 任务
     if need_phase_b:
         admission.acquire()  # 模型级别 acquire 一次
         all_enhanced = await sandbox.run(
             load_all_components_meta,  # 单个函数加载全部组件
             repo_id, component_configs
         )
         admission.release()

  5. safetensors 按组件目录解析  # [P0-4]
     for comp_name in components:
         try:
             meta = await hf_client.fetch_safetensors_meta(
                 repo_id, subfolder=comp_name  # 子目录级别
             )
         except NotASafetensorsRepoError:
             # fallback: 扫描该子目录下的文件列表
             meta = await scan_component_files(repo_id, comp_name)

  6. 构建复合 ModuleGraph
     root = ModuleNode(name=class_name, type="pipeline")
     for comp_name, comp_graph in component_graphs.items():
         root.children.append(comp_graph.root)
     # denoiser 组件标记为 primary
```

---

## [v1.1+ PARKING] §5.3.6 并行策略计算流程

```
Client ──POST /api/v1/models/{id}/parallel──▶ Router
  body: { tp: 4, pp: 2, dp: 8, ep: 4, hardware: "A100-80G" }

strategy_composer 流程:
  1. 加载基础 ModuleGraph (from cache or pipeline)

  2. 硬件规格查询
     hw = hw_specs.get("A100-80G") → HardwareProfile(
         memory_gb=80, bandwidth_gbps=600,
         nvlink=True, ib_bandwidth_gbps=200
     )

  3. 策略组合与约束校验
     strategy = strategy_composer.compose(
         tp=4, pp=2, dp=8, ep=4,
         total_gpus=tp*pp*dp,  # 64
         hw=hw
     )
     # 约束: tp*pp*dp == total_gpus
     # 约束: ep ≤ num_experts, ep | num_experts

  4. 分区计算 → PartitionedGraph + CommOps  # [P2-33]
     partition_result = partition(graph, strategy)
     # 输出:
     #   PartitionedGraph: 每个 node 标注 (tp_rank, pp_stage, dp_rank, ep_rank)
     #   CommOps: List[CommOp(type, src_ranks, dst_ranks, tensor_shape, estimated_time)]
     #     type ∈ {AllReduce, AllGather, ReduceScatter, P2P, AllToAll}

  5. 返回 JSON for 3D rendering
```

---

## [v1.1+ PARKING] §5.3.7 PP 调度器流程

```
scheduler.py:

ScheduleStep = NamedTuple('ScheduleStep', [
    ('stage_id', int),
    ('micro_batch_id', int),
    ('action', Literal['F', 'B', 'W', 'IDLE']),
    ('gpu_rank', int),  # 注意：v1.1+ 并行调度中的 GPU 物理 rank；与 v1.0 白名单字段 `gpu_id`（GPU spec ID, string）为不同概念
    ('time_slot', int),
])

支持调度策略:
  - "1f1b": 1F1B pipeline schedule
  - "interleaved_1f1b": Interleaved 1F1B (Megatron)
  - "zero_bubble": Zero Bubble (F/B/W separation)

generate_schedule(pp_stages, num_micro_batches, strategy) → List[ScheduleStep]:
  # 输出每个 GPU 在每个 time_slot 的动作
  # 用于前端甘特图渲染
```

---

## [v1.1+ PARKING] §7.4 多模型对比接口

> v1.0 冻结范围明确「模型对比分屏 → v1.1+」。以下 schema 原样保留供 v1.1 重新对齐 04 后复用。

**`POST /api/v1/compare`**

```python
from pydantic import BaseModel, Field, model_validator, ConfigDict

class CompareRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    repo_ids: list[str] = Field(..., min_length=2, max_length=4)
    mode: Literal["quick", "deep"] = "quick"
    revision: str = "main"

    @model_validator(mode="after")
    def validate_deep_limit(self) -> "CompareRequest":
        if self.mode == "deep" and len(self.repo_ids) > 2:
            raise ValueError("deep compare supports max 2 models")
        for rid in self.repo_ids:
            if not REPO_ID_RE.fullmatch(rid):
                raise ValueError(f"Invalid repo_id: {rid}")
        return self

class CompareResponse(BaseModel):
    models: list[ModelSummary]
    diff: ComparisonDiff

class ModelSummary(BaseModel):
    repo_id: str
    commit_sha: str
    model_type: str
    architecture: str
    num_parameters: int
    num_hidden_layers: int
    hidden_size: int
    features: list[str]
    revision_used: int

class ComparisonDiff(BaseModel):
    parameter_comparison: ParameterComparison
    feature_diff: dict[str, list[bool]]
    structure_diff: StructureDiff

class ParameterComparison(BaseModel):
    model_level: dict[str, list[int | None]]
    per_layer: list[LayerComparison]

class LayerComparison(BaseModel):
    layer_pattern: str
    values: list[int | None]
    ratio: float | None = None

class StructureDiff(BaseModel):
    shared_patterns: list[str]
    unique_to: dict[str, list[str]]
    alignment: list[AlignmentEntry]
    layout_pair: LayoutPair | None = None

class AlignmentEntry(BaseModel):
    model_a_node: str | None
    model_b_node: str | None
    match_type: Literal["exact", "similar", "unique_a", "unique_b"]

class LayoutPair(BaseModel):
    model_a_offset: list[float]
    model_b_offset: list[float]
```

---

## [v1.1+ PARKING] §7.5 并行策略接口

> v1.0 冻结范围：`GET /api/v1/parallelism-strategies` 返回空数组（见 04 §4.9.4）；具体并行计算端点推迟至 v1.2+。

**`GET /api/v1/models/{repo_id:path}/parallel`**

```
请求:
  GET /api/v1/models/meta-llama/Llama-3.1-70B/parallel?tp=4&pp=2&dp=8&hardware=A100-80G

查询参数:
  tp: int = Query(1, ge=1, le=128)
  pp: int = Query(1, ge=1, le=64)
  dp: int = Query(1, ge=1, le=1024)
  ep: int = Query(1, ge=1, le=64)       # MoE only
  cp: int = Query(1, ge=1, le=32)
  sp: int = Query(1, ge=1, le=32)
  hardware: Literal["A100-40G", "A100-80G", "H100-80G", "H200", "A10G"] = "A100-80G"
  micro_batches: int = Query(8, ge=1, le=64)
  schedule: Literal["1f1b", "interleaved_1f1b", "zero_bubble"] = "1f1b"
```

```python
class ParallelResponse(BaseModel):
    strategy: ParallelStrategy
    partition: PartitionedGraph
    comm_ops: list[CommOp]
    schedule: list[ScheduleStep] | None = None
    hardware: HardwareProfile
    estimated_memory_per_gpu: ByteSize
    estimated_comm_overhead: ByteSize

class ParallelStrategy(BaseModel):
    tp: int
    pp: int
    dp: int
    ep: int
    cp: int
    sp: int
    total_gpus: int
    constraints_satisfied: bool
    constraint_violations: list[str]
    supported_dimensions: list[str]

class PartitionedGraph(BaseModel):
    nodes: list[PartitionedNode]

class PartitionedNode(BaseModel):
    node_id: str
    tp_shard: Literal["col", "row"] | None = None
    pp_stage: int
    dp_rank: int
    ep_rank: int | None = None

class CommOp(BaseModel):
    op_type: Literal["AllReduce", "AllGather", "ReduceScatter", "P2P", "AllToAll"]
    source_ranks: list[int]
    dest_ranks: list[int]
    tensor_shape: list[int]
    estimated_bytes: int
    estimated_time_ms: float
    start_ms: float
    end_ms: float

class ScheduleStep(BaseModel):
    gpu_rank: int  # v1.1+ 并行调度 GPU 物理 rank；与 v1.0 白名单 `gpu_id`（spec ID, string）为不同概念
    time_slot: int
    start_ms: float
    end_ms: float
    action: Literal["F", "B", "W", "IDLE", "COMM"]
    stage_id: int
    micro_batch_id: int
    memory_usage_bytes: int | None = None
    comm_op_ids: list[str] | None = None
```

---

## [v1.1+ PARKING] §7.11 Rate Limiting（冻结说明）

> 对齐原则 1（非商业化、内部工具）：Rate Limiting 在当前产品定位下**不进入路线图**。
>
> 本节仅保留冻结声明，不提供具体限额与实现细节；只有当产品定位发生变更（例如开放公网）并完成原则变更评审后，才允许重新引入独立设计文档。

---

## [v1.1+ PARKING] 已废弃 schema 名 — 迁移说明

> 09 §四 名词解释处保留了短迁移指针；本节提供完整的历史→新名称映射，便于阅读旧文档或代码残留。

| 旧名称（≤ 2026-04-24） | 新名称（2026-04-25 起） | 位置 |
|---|---|---|
| `TreeNode` | `ModuleNode`（节点） + `HierarchyTree`（父子关系） | `models/graph.py` |
| `FlowStep` | `DataEdge`（数据流边） + `ModuleNode.metadata`（计算描述） | `models/graph.py` |
| `ParamStats` | `ModuleNode.params`（精确）+ `EstimateResult`（估算，含 `Provenance.confidence=ESTIMATED`） | `models/graph.py`, `models/contracts.py` |
| `MoEInfo` / `MLAInfo` / `QuantInfo` / `KeyConfig` | `ArchitectureProfile.features[]` + `ArchitectureProfile.config_summary` | `models/profile.py` |
| `SubConfig` | 子模型 namespace 隔离的嵌套 dict（保持原始结构，不扁平化，见 09 §5.2 补充修订 #19） | `services/parsing/config_normalizer.py` |

**使用规则**：任何新建模块/PR 中**禁止**再出现上述旧名称；审核时发现即驳回。

---

## 回收指南（当 v1.1 启动时）

1. 按版本冻结 04/08/11 的当日快照，审视本 parking 文件的每一节是否仍然适用（API 路径 / schema 字段可能已演进）。
2. 回取章节内容时，先在 09 内新增对应编号（不要复用旧 §x.y，建议递增），再删除 parking 中对应段落。
3. Rate limiting 回取前必须同时改 04 §4.12（新增 RATE_LIMITED / ADMISSION_REJECTED）与 09 §7.9.2 错误码映射。
4. 模型对比分屏回取前需与前端 Template 对比屏 PRD 同步（05-visualization-design）。
5. 并行策略回取前需对齐 11 §4（ParallelismStrategy）+ ADR-020 + 附录 B（并行可视化设计）。

---

## 变更日志

- 2026-04-25：首版创建。从 09-backend-detailed-design.md 剥离 v1.1+ 候选内容（扩散/VLA/世界模型检测需求、AST 源码解析、Template D/E/F、§5.2.11 并行模块、§5.3.4/§5.3.6/§5.3.7 相关流程、§7.4/§7.5/§7.11 接口）。
