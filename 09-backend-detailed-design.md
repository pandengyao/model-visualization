# HF Model Visualizer — 后端 API 详细设计文档

> HuggingFace 模型结构 3D 交互式可视化 Web 服务 — 后端模块详细设计 | v1.0

> **文档权威性（2026-04-25 归一）**：
> - 本文档是**后端实现的权威源**（pipeline 细节、缓存实现、错误处理内部流程）。
> - **扩展点契约**（Adapter / Template / AnimationLayer / ParallelismStrategy / MemoryEstimator / GPU Catalog / DataFlowDirection / ConfigEdit / Pipeline 签名与注册机制）以 [11-extension-points.md](11-extension-points.md) 为准，与 11 冲突时以 11 为准。
> - **API 表层**（HTTP / SSE / WebSocket 端点、对外 Schema、响应头、错误码、交互响应预算）以 [04-api-design.md](04-api-design.md) 为准，与 04 冲突时以 04 为准。
> - **产品原则**见 [README](README.md)：非商业化（P1）/ 精美 3D（P2）/ 结构与数据流 100% 正确（P3）/ 教学深度超越竞品（P4）/ 前期不做性能优化但交互延迟是硬约束（P5）/ 架构可扩展（P6）/ 真实模型优先（P7）/ 架构广度底线（P8）/ Provenance 可追溯（P9）。本文档所有设计决策与原则冲突时，**原则优先、方案让步**。

> **v1.0 范围冻结（2026-04-25 最终决议）**：按 README 产品原则重新划定。
>
> **v1.0 必交付**（对齐 README / 11）：
> - HF 模型 → 3D 结构可视化（Template A/B/C/G 四选一，**未识别架构不得默认回退 A**）
> - Stage-1 结构动画 + Stage-2 数据流动画最小子集（Attention Q/K/V 分解 / MoE 路由 / Residual flow）
> - `PATCH /config` 动态编辑 + 端到端 < 300ms、后端 < 200ms 热更新（原则 5 交互硬约束）
> - `MemoryEstimator` **推理版**（weights + KV cache + activations，不含 gradients/optimizer_states/comm_buffer）
> - `GPU Catalog` 数据表（≥ 9 款，含国产卡），位置 `backend/data/gpu-catalog.yaml`，严禁代码硬编码
> - Provenance 为**全局强制字段**（原则 9）
> - 扩展接口就位：`ArchitectureAdapter` / `TemplateContract` / `AnimationLayer` / `ParallelismStrategy` / `MemoryEstimator` / `DataFlowDirection.backward`（接口定义 + 空实现占位）
>
> **v1.0 不交付**（推迟到 v1.1+/v2.0，保留接口留空实现）：
> - 并行策略可视化（TP/PP/DP/EP/CP/SP）→ v1.2+
> - 训练数据流与反向图生成（`direction="backward"` / `gradient_flow` / `activation_checkpoint` / `gradient_accumulation`）→ v1.1+
> - 训练版显存估算（Megatron-LM / FSDP）→ v1.1+
> - 源码 AST 解析 → v1.1+
> - 模型对比分屏 → v1.1+
>
> **本文档历史章节中与上述冻结范围冲突的内容（并行策略实现细节、训练反向流、rate limiting 等），已通过本轮修订加入弃用/软化标注；正文具体实现若与 11/04/README 冲突，以权威文档为准。**

---

## 一、项目背景

本模块为 HF Model Visualizer 的 FastAPI 后端服务，负责从 HF Hub 获取模型信息、检测架构特征、构建结构树、估算参数、生成推理数据流，以 JSON 提供给前端渲染。见 PRD §03、§04、§06。

| # | 需求点 | 优先级 | 现有能力 | 差距 | 涉及代码 |
|---|-------|-------|---------|------|---------|
| 1 | 下载并解析 HF 模型配置（config.json / model_index.json） | P0 | 不支持 | 需实现 async 下载 + 框架判定 + AutoConfig 加载 | `[新增] services/config_parser.py` |
| 2 | 模型框架判定（transformers / diffusers / 其他） | P0 | 不支持 | 需作为所有后续检测的入口分发 | `[新增] services/detectors/registry.py` |
| 3 | LLM 架构特征检测（注意力变体 / 位置编码 / 归一化 / FFN 变体 / 逐层调度 / tied weights / 模板分类 A-G） | P0 | 不支持 | 7 模板覆盖 95%+ transformers 模型 | `[新增] services/detectors/llm.py` |
| 4 | MoE + MLA + 量化检测 | P0 | 不支持 | 含 DeepSeek-MoE 共享专家 + MLA 压缩维度 + 多量化格式 | `[新增] services/detectors/moe.py` `mla.py` `quantization.py` |
| 5 | 全模态结构检测（视觉/音频编码器、投影器、token 策略、跨模态注入） | P0 | 不支持 | 视觉 v1.0 基础支持，音频 v1.1 | `[新增] services/detectors/multimodal.py` |
| 6 | 扩散模型结构检测（UNet/DiT、VAE、文本编码器、调度器、ControlNet） | P1 | 不支持 | diffusers 模型解析链路独立于 transformers | `[新增] services/detectors/diffusion.py` |
| 7 | VLA 结构检测（动作头、本体感知编码器、动作 tokenizer） | P1 | 不支持 | 新兴架构，需灵活适配 | `[新增] services/detectors/vla.py` |
| 8 | 世界模型结构检测（帧编码器、动力学模型、时序建模、动作条件注入） | P1 | 不支持 | 新兴架构，需灵活适配 | `[新增] services/detectors/world_model.py` |
| 9 | **检测器注册机制**（可插拔，新增模型类别无需改动主流程） | P0 | 不支持 | 注册表 + 自动发现，预留扩展点 | `[新增] services/detectors/registry.py` |
| 10 | 模型结构树构建（meta-device 真实树 + config 合成树降级） | P0 | 不支持 | 双模式 + 自动降级 | `[新增] services/tree_builder.py` |
| 11 | 参数量估算（精确统计 + config 估算双模式 + tied weights） | P0 | 不支持 | 含准确性测试 ground truth | `[新增] services/param_estimator.py` |
| 12 | 推理数据流生成（模板选择 + FlowStep 序列 + SafeTensors 验证） | P0 | 不支持 | v1.0 实现模板 A/B/C | `[新增] services/flow_generator.py` |
| 13 | 模型卡片信息获取（model_info + README） | P1 | 不支持 | HF Hub API 调用 | `[新增] services/model_card.py` |
| 14 | 两层缓存（L0 内存 TTL + L1 文件 JSON + 原子写入） | P0 | 不支持 | 含并发安全 + HF 降级策略 | `[新增] cache.py` |
| 15 | REST API 路由（6 端点 + SSE 进度推送） | P0 | 不支持 | 含错误统一格式 | `[新增] routers/model.py` `compare.py` |
| 16 | Pydantic 响应模型（11+ Schema 类） | P0 | 不支持 | 前后端契约 | `[新增] models/schemas.py` |
| 17 | 安全规则（model_id 校验 + Rate limit + trust_remote_code 隔离） | P1 | 不支持 | slowapi 限流 + 正则校验 | `[新增] main.py` |

| 18 | GPU 显存估算（推理 + 训练双模式，支持 Megatron 和 FSDP 两种并行框架） | P0 | 不支持 | 含 Megatron（TP/PP/DP/CP/EP）和 FSDP（Full/Grad/No Shard）两种框架，推理含 KV Cache 和量化感知 | `[新增] services/memory_estimator.py` |
| 19 | 并行策略计算引擎（Device Mesh / TP 切分 / PP 调度 / DP 梯度同步 / 通信量估算） | P1 | 不支持 | 覆盖 Megatron-style 全并行维度 | `[新增] services/parallel_engine.py` |
| 20 | 训练数据流生成（宏观 pipeline 调度 + 并行通信 + 梯度累积） | P1 | 不支持 | PP 1F1B 时序、DP 梯度同步、梯度累积 | `[新增] services/training_flow_generator.py` |
| 21 | 层内前向/反向数据流生成（单层内部的计算步骤和梯度回传路径） | P1 | 不支持 | 覆盖所有模型类别的层内 forward/backward 细节 | `[新增] services/flow_generator/backward.py` |
| 22 | transformers 模型结构源码解析（解析 `modeling_*.py` 中的类层级、`forward()` 调用链、模块注册关系，作为结构树/数据流/并行策略的 ground truth） | P0 | 不支持 | AST 解析 + meta-device 实例化双路径 | `[新增] services/model_inspector.py` |

### 架构设计约束：面向变化的能力

| 变化维度 | 具体场景 | 设计应对 |
|---------|---------|---------|
| **新模型架构** | transformers/diffusers 每月新增 model_type | 检测器注册机制（#9）+ 模板可扩展 + 未知架构 graceful fallback |
| **新并行策略** | 如 ZeRO++、Tensor+Sequence 混合新范式 | 并行引擎（#19）策略可插拔，通信原语抽象为接口 |
| **新量化方法** | 如 FP4、新的量化库 | 量化检测器（#4）按 method 注册，与核心逻辑解耦 |
| **新模态** | 如触觉、3D 点云、视频理解 | 模态检测器（#5）可插拔，子配置解析泛化 |
| **新模型类别** | 超出当前 7 类的全新范式 | 框架判定（#2）设兜底分支 + 手动注册入口 |
| **transformers 内部重构** | modeling 文件结构/命名变化 | 模型源码解析（#22）基于 AST/反射而非硬编码路径 |
| **HF Hub API 变更** | 新的文件格式、API 版本 | 文件下载（#1）抽象为适配层，隔离 HF SDK 细节 |
| **训练框架演进** | Megatron/FSDP 之外的新框架（如 DeepSpeed Ulysses） | 显存估算（#18）和训练流（#20）按框架注册，公式可配置 |
| **响应结构演进** | 前端需要新字段 | Schema（#16）使用版本化 + `extra="allow"` + 向后兼容策略 |

**落地到代码层面的 3 条硬性规则**：

1. **接口隔离** — 每个检测器/生成器/估算器实现统一抽象接口（`BaseDetector` / `BaseFlowGenerator` / `BaseMemoryEstimator`），新增类别只需新增实现类
2. **配置驱动** — 模板映射、切分规则、估算公式等可变逻辑外置为配置/注册表，而非硬编码在业务逻辑中
3. **Graceful degradation** — 遇到未识别的架构/策略/格式时，返回尽可能多的信息 + 明确标注"未完全支持"，而非报错

---

---

## 二、需求分析

| # | 需求点 | PRD 引用 | 目标代码路径 | 关键函数/类 | 前置依赖 | 版本 | 复杂度 |
|---|-------|---------|-------------|------------|---------|------|-------|
| 1 | HF 仓库文件下载 | §03 数据流, §04 §4.4 | `[新增] services/hub_client.py` | `list_repo_files()`, `async download_file()`, `download_all_metadata()` | 无 | v1.0 | M |
| 2 | 模型框架判定 | §06 Step 1.7 模板选择 | `[新增] services/detectors/registry.py` | `detect_framework(repo_files) → "transformers" \| "diffusers" \| "unknown"` | #1 | v1.0 | S |
| 3 | LLM 架构特征检测 | §06 Step 1.3, §06 §7b 模板选择算法 | `[新增] services/detectors/llm.py` | `detect_attention_variant()`, `detect_position_encoding()`, `detect_norm_type()`, `detect_ffn_variant()`, `compute_layer_schedule()`, `select_template()` | #1, #2, #9 | v1.0 | H |
| 4 | MoE + MLA + 量化检测 | §06 Step 1.3 detectors, §04 §4.2.5 | `[新增] services/detectors/moe.py` `mla.py` `quantization.py` | `detect_moe() → MoEInfo`, `detect_mla() → MLAInfo`, `detect_quantization() → QuantInfo` | #1, #9 | v1.0 | M |
| 5 | 全模态结构检测 | §06 Step 1.3 sub_configs | `[新增] services/detectors/multimodal.py` | `detect_vision_encoder()`, `detect_audio_encoder()`, `detect_projector_type()`, `detect_token_strategy()`, `detect_cross_modal_injection()` | #1, #2, #9 | v1.0 视觉 / v1.1 音频 | H |
| 6 | 扩散模型结构检测 | 新增（PRD 未覆盖） | `[新增] services/detectors/diffusion.py` | `detect_denoiser_type()`, `detect_vae()`, `detect_text_encoders()`, `detect_scheduler()`, `detect_controlnet()` | #1, #2, #9 | v1.1 | H |
| 7 | VLA 结构检测 | 新增 | `[新增] services/detectors/vla.py` | `detect_action_head()`, `detect_proprio_encoder()`, `detect_action_tokenizer()` | #1, #2, #9 | v1.1 | M |
| 8 | 世界模型结构检测 | 新增 | `[新增] services/detectors/world_model.py` | `detect_frame_encoder()`, `detect_dynamics_model()`, `detect_temporal_modeling()` | #1, #2, #9 | v1.1 | M |
| 9 | 检测器注册机制 | 架构设计约束 | `[新增] services/detectors/registry.py` | `class DetectorRegistry`, `register()`, `detect_all(config, files) → DetectionResult` | 无 | v1.0 | M |
| 10 | 模型结构树构建 | §06 Step 1.4, §04 §4.2.3 TreeNode | `[新增] services/tree_builder.py` | `load_model_meta(config) → TreeNode`, `build_synthetic_tree(config) → TreeNode`, `tree_to_text()` | #1, #22 | v1.0 | H |
| 11 | 参数量估算 | §06 Step 1.5, §06 §7h tied weights, §04 §4.2.4 | `[新增] services/param_estimator.py` | `count_parameters(model) → ParamStats`, `estimate_params_from_config(config) → ParamStats` | #1, #10 | v1.0 | H |
| 12a | LLM 推理数据流生成 | §06 Step 1.7 §7a-§7g | `[新增] services/flow_generator/llm.py` | `class LLMFlowGenerator(BaseFlowGenerator)`, `generate(config) → list[FlowStep]` | #1, #3, #4, #9, #22 | v1.0 | H |
| 12b | 多模态推理数据流生成 | §05 §5.5 阶段一多模态分支 | `[新增] services/flow_generator/multimodal.py` | `class MultimodalFlowGenerator(BaseFlowGenerator)` | #5, #12a | v1.0 视觉 / v1.1 全模态 | H |
| 12c | 扩散模型数据流生成 | 新增 | `[新增] services/flow_generator/diffusion.py` | `class DiffusionFlowGenerator(BaseFlowGenerator)` | #6 | v1.1 | H |
| 12d | VLA 数据流生成 | 新增 | `[新增] services/flow_generator/vla.py` | `class VLAFlowGenerator(BaseFlowGenerator)` | #7 | v1.1 | M |
| 12e | 世界模型数据流生成 | 新增 | `[新增] services/flow_generator/world_model.py` | `class WorldModelFlowGenerator(BaseFlowGenerator)` | #8 | v1.1 | M |
| 13 | 模型卡片信息获取 | §06 Step 1.6, §04 §4.2.6 | `[新增] services/model_card.py` | `async fetch_model_card(model_id) → ModelCard` | #1 | v1.0 | S |
| 14 | 两层缓存 | §06 Step 1.9, §04 §4.3 | `[新增] cache.py` | `get_cached()`, `set_cached()`, `safe_write_cache()`, HF 降级策略 | 无 | v1.0 | M |
| 15 | REST API 路由 | §04 §4.1 六端点 + §04 §4.5 SSE | `[新增] routers/model.py` `compare.py` | 6 个端点 + SSE progress 端点 | #14, #16, #17 | v1.0 | M |
| 16 | Pydantic 响应模型 | §04 §4.2 全部 Schema | `[新增] models/schemas.py` | `ModelVisualization`, `TreeNode`, `FlowStep`, `ParamStats`, `MoEInfo`, `MLAInfo`, `QuantInfo` 等 11+ 类 | 无 | v1.0 | M |
| 17 | 安全规则 | §04 §4.4 | `[新增] main.py` `middleware/security.py` | `validate_model_id()`, slowapi RateLimiter, trust_remote_code 警告 | 无 | v1.0 | S |
| 18 | GPU 显存估算 | 新增（参考 megatron_memory_estimator） | `[新增] services/memory_estimator.py` | `class MegatronEstimator(BaseEstimator)`, `class FSDPEstimator(BaseEstimator)`, `class InferenceEstimator(BaseEstimator)` | #1, #11 | v1.0 | H |
| 19 | 并行策略计算引擎 | §并行可视化设计 §10.1-§10.7 | `[新增] services/parallel_engine.py` | `compute_device_mesh()`, `compute_tp_sharding()`, `compute_pp_schedule()`, `estimate_comm_volume()` | #1, #3, #22 | v1.1 | H |
| 20 | 训练数据流生成（宏观） | §并行可视化设计 §10.8 A-F | `[新增] services/training_flow_generator.py` | `generate_training_step_flow()`, `generate_pp_schedule()`, `generate_grad_sync_flow()` | #12a, #19 | v1.1 | H |
| 21 | 层内前向/反向数据流生成 | §并行可视化设计 §10.8 E | `[新增] services/flow_generator/backward.py` | `generate_backward_flow(forward_steps) → list[FlowStep]`，覆盖所有模型类别 | #12a-e, #22 | v1.1 | H |
| 22 | transformers 模型结构源码解析 | 架构设计约束 | `[新增] services/model_inspector.py` | `inspect_model_class(config) → ModuleHierarchy`, `extract_forward_chain()`, `map_parallel_points()` | #1 | v1.0 | H |

---

## 三、设计目标

| 类别 | 目标 | 度量标准 |
|------|------|---------|
| **用户价值** | 任意 HF Hub 模型输入即可获得有价值的分析结果 | HF Hub Downloads Top-500 模型成功率 ≥ 95%；未适配模型仍返回基础结构树 + 参数量（graceful degradation） |
| **用户价值** | 交互式架构探索：不只是"看"，还能"改"和"试" | 支持 config 实时修改 → 结构/参数/显存/FLOPS 联动更新 + What-if diff |
| **用户价值** | 端到端可视化覆盖模型全生命周期（推理 + 训练 + 部署规划） | 每种模型类别提供：推理数据流 + 训练前向/反向流 + 并行策略 + 显存/FLOPS 估算 |
| **架构** | 可插拔注册机制，新增模型类别零改动主流程 | 新增 1 种模型类别仅需 1 个文件 + 注册 1 行 |
| **架构** | 增量重算引擎，config 变更仅触发受影响模块 | 依赖图驱动脏标记传播，非全量重跑 |
| **准确性** | 参数量基于 transformers 真实模型结构精确统计 | meta-device 加载成功时零误差；降级路径（AST 源码解析）同样 100% 精确；仅运行时相关数据（显存/FLOPS/延迟）标注 `is_estimated=true` |
| **准确性** | 结构树/数据流以 transformers `modeling_*.py` 为唯一权威来源 | 类层级、forward() 调用链、模块注册关系均来自源码解析，杜绝硬编码臆造 |
| **容错** | 任何单点失败不导致整体不可用 | HF Hub 不可用 → 返回缓存 + stale 标记；meta 加载失败 → 降级合成树；未知架构 → 返回通用基础分析 + "未完全支持"标注 |
| **安全** | 零信任：不执行任何模型代码，不下载权重文件 | trust_remote_code 默认关闭；下载除权重文件外的所有仓库文件；model_id 严格校验 |
| **扩展性** | Roadmap R1-R33 全部有明确扩展点，无需重构即可接入 | 每个预留项可在现有接口/注册表上扩展 |

---

## 四、名词解释

> ⚠ **术语迁移（2026-04-25 文档归一）**：本章及 §二 需求表中早期沿用的 `TreeNode` / `FlowStep` / `ParamStats` / `MoEInfo` / `MLAInfo` / `QuantInfo` / `KeyConfig` 等名称，在 §5.1.2 已被统一替换为 `ModuleGraph` + `ModuleNode` + `DataEdge` + `ArchitectureProfile` + `EstimateResult`。阅读时请按以下映射理解：
> - `TreeNode` → `ModuleNode`（节点） + `HierarchyTree`（父子关系）
> - `FlowStep` → `DataEdge`（数据流边）+ `ModuleNode.metadata`（计算描述）
> - `ParamStats` → `ModuleNode.params`（精确）+ `EstimateResult`（估算，带 `Provenance.confidence=ESTIMATED`）
> - `MoEInfo` / `MLAInfo` / `QuantInfo` / `KeyConfig` → `ArchitectureProfile.features[]` + `ArchitectureProfile.config_summary`
> 新建模块时请直接使用 §5.1.2 的新名称；旧名称仅作为阅读历史章节的注释残留。

仅列本文档新引入的术语：

| 术语 | 定义 | 对应代码 |
|------|------|---------|
| Meta-device 加载 | 使用 PyTorch meta device 实例化模型，仅创建模块结构和 shape 元信息，不分配实际显存/内存 | `[新增] services/tree_builder.py` |
| 模型结构树（Model Tree） | 从 transformers `modeling_*.py` 获取的 100% 精确模块层级树。主路径通过 meta-device 实例化获取，降级路径通过 AST 源码解析获取，两者结果一致 | `[新增] services/tree_builder.py` |
| 源码解析（Source Parsing） | 对 `modeling_*.py` 进行 AST 分析，提取 `__init__` 模块注册关系和 `forward()` 调用链，作为 meta-device 加载失败时的降级方案 | `[新增] services/model_inspector.py` |
| FlowStep | 数据流的最小单元，描述一个计算步骤（模块路径、输入/输出 shape、计算描述），支持递归嵌套 sub_steps | `[新增] models/schemas.py#FlowStep` |
| 架构模板（Template A-G） | 按模型架构特征分类的 7 种数据流生成模板（A=LLaMA Decoder, B=LLaMA-MoE, C=DeepSeek-MoE, D=GPT-2, E=Hybrid Mamba, F=Encoder-Decoder, G=Multimodal VLM） | `[新增] services/flow_generator/` |
| 检测器注册表 | 可插拔的架构特征检测机制，每个检测器实现 `BaseDetector` 接口并注册到全局表，主流程按表遍历执行 | `[新增] services/detectors/registry.py` |
| Config Patch | 用户对模型 config 的局部修改（JSON Patch 格式），作为增量重算的输入 | `[新增] services/config_mutator.py` |
| 增量重算（Incremental Recompute） | 基于字段→计算模块依赖图的脏标记传播机制，config 变更仅触发受影响模块重新计算 | `[新增] services/incremental_engine.py` |
| L0/L1 缓存 | L0=进程内内存 LRU 缓存，L1=文件系统 JSON 持久缓存，逐级回源到 HF Hub | `[新增] cache.py` |
| 硬件规格数据库 | 预置主流 GPU 完整参数（显存/带宽/SM/Tensor Core/FLOPS/互联），供显存/算力/延迟估算参数化使用 | `[新增] services/hardware_registry.py` |
| Graceful Degradation | 容错降级策略：任何子模块失败时返回尽可能多的有效信息 + 明确标注降级原因，而非整体报错 | 全局设计约束 |

---

## 五、方案设计



---

### 5.1 整体架构设计（终稿）

### 5.1.1 分层架构

```
┌─────────────────────────────────────────────────────┐
│                   Interface Layer                    │
│  routers/model.py  routers/compare.py  routers/explore.py │
│  (FastAPI endpoints, SSE streaming, WebSocket v1.x) │
├─────────────────────────────────────────────────────┤
│                  Application Layer                   │
│  pipeline.py — 5-stage 管线编排                      │
│  admission.py — 全局准入控制器（信号量+内存水位）      │
│  sandbox.py — ProcessPoolExecutor(spawn) + 超时      │
├─────────────────────────────────────────────────────┤
│                    Domain Layer                      │
│  parsing/    — 4 层解析策略（L0→L1→L2→L3）           │
│  detectors/  — 特征检测器注册表（按特征注册）          │
│  flow_gen/   — 数据流生成器注册表                     │
│  estimators/ — 资源估算器注册表                       │
│  parallel/   — 并行策略模块                          │
│  layout/     — 3D 布局计算                           │
├─────────────────────────────────────────────────────┤
│                 Infrastructure Layer                 │
│  cache.py    — L0 内存(TTLCache+Lock) + L1 分段文件   │
│  hf_client.py — HuggingFace Hub 交互（白名单下载）    │
│  hw_specs.py  — GPU 硬件规格数据库                    │
│  metrics.py   — Prometheus 指标（v1.1）               │
└─────────────────────────────────────────────────────┘
```

**依赖方向**：Interface → Application → Domain ← Infrastructure，Domain 层零外部依赖。

### 5.1.2 核心数据模型

```python
# --- Provenance（统一溯源） ---
class Confidence(str, Enum):
    EXACT = "exact"           # 来自 transformers 模型文件，100% 准确
    INFERRED = "inferred"     # 来自 safetensors 元数据推断
    ESTIMATED = "estimated"   # 计算估算（显存/FLOPS/延迟）

class Provenance(BaseModel):
    source: str               # "meta_device" | "safetensors_metadata" | "config_json" | "ast_parse"
    confidence: Confidence
    caveats: list[str] = []   # 例如 ["trust_remote_code=False, 部分自定义层可能缺失"]

# --- ModuleGraph（DAG 核心） ---
class ModuleNode(BaseModel):
    id: str                   # 唯一路径，如 "model.layers.0.self_attn"
    class_name: str           # 如 "LlamaAttention"
    params: int               # 精确参数量
    dtype: str                # "float16" | "bfloat16" | "int8" | ...
    tensor_shapes: dict[str, list[int]]  # {"weight": [4096, 4096], "bias": [4096]}
    metadata: dict            # 扩展字段（量化 bits、MoE num_experts 等）
    provenance: Provenance

class DataEdge(BaseModel):
    source: str               # 源节点 id
    target: str               # 目标节点 id
    # [MODIFIED 2026-04-25] 对齐 11 §7 + ADR-018：DataFlowDirection 一等公民
    edge_type: Literal[
        "data_flow",             # 前向激活（v1.0）
        "residual",              # 残差连接（v1.0）
        "skip_connection",       # 跨层跳跃（v1.0）
        "branch_merge",          # 分支合并（v1.0）
        "gradient_flow",         # 反向梯度（v1.1 启用，v1.0 不产出）
        "activation_checkpoint", # 激活值缓存（v1.1）
        "gradient_accumulation", # 梯度累积（v1.1）
    ]
    direction: Literal["forward", "backward", "bidirectional"]  # v1.0 仅产出 "forward"
    tensor_shape: list[int] | None
    provenance: Provenance   # [MANDATORY] 对齐 ADR-016 / 原则 9，任何边都必须携带 Provenance

class HierarchyTree(BaseModel):
    """前端 expand/collapse 用"""
    id: str
    children: list["HierarchyTree"] = []

class ModuleGraph(BaseModel):
    nodes: dict[str, ModuleNode]
    edges: list[DataEdge]
    hierarchy: HierarchyTree
    
    def to_tree(self) -> dict: ...      # DAG → 嵌套树（前端 3D 渲染用）
    def to_flat(self) -> dict: ...      # 扁平 nodes+edges（前端 2D 拓扑用）

# --- ArchitectureProfile（特征画像） ---
class ArchitectureProfile(BaseModel):
    model_type: str                           # "llama" | "qwen2_moe" | "stable_diffusion" | ...
    features: list[str]                       # ["moe", "mla", "gqa", "sliding_window", ...]
    config_summary: dict                      # 关键 config 字段标准化后的摘要
    parsing_layers_used: list[str]            # ["config_json", "safetensors_metadata", "meta_device"]
    provenance: Provenance

# --- Stage 契约类型 ---
class ParseResult(BaseModel):
    graph: ModuleGraph
    profile_hint: dict                        # 给 Stage 2 的提示信息

class DetectResult(BaseModel):
    profile: ArchitectureProfile

class FlowResult(BaseModel):
    graph: ModuleGraph                        # edges 已填充

class EstimateResult(BaseModel):
    memory: dict                              # 显存估算
    flops: dict                               # 计算量估算
    kv_cache: dict | None
    communication: dict | None                # 并行通信量
    latency: dict | None
    provenance: Provenance                    # 整体置信度 = ESTIMATED

class LayoutResult(BaseModel):
    positions: dict[str, dict]                # node_id → {x, y, z, width, height, depth}
    camera: dict                              # 推荐相机位置
    bounds: dict                              # 场景包围盒
```

### 5.1.3 五阶段纯函数管线

> **纯函数契约（对齐 11 §9）**：以下五个 Stage **全部是纯函数**——无副作用、无隐式 I/O、无全局状态依赖；所有 I/O（HF Hub 下载、L0/L1 缓存读写、meta-device 实例化、文件系统访问）**必须在 S1 入口前完成或在 S1 独立 I/O 前置步骤中完成**，S1 返回 `RawStructure` 后的全部阶段均接收不可变 Pydantic/dataclass 输入并返回新实例。

> **Pipeline 顶层签名（支持增量重跑，对齐 PATCH /config 热更新路径）**：
>
> ```python
> def run_pipeline(
>     raw_state: RawStructure,              # S1 产物（含原始 config + safetensors_meta + 可选 meta_state_dict）
>     overrides: dict | None = None,        # 来自 PATCH /config 的 user override
>     start_stage: Literal["S2", "S3", "S4", "S5"] = "S2",  # 热更新路径从 S2 起跳
> ) -> PipelineResult:
>     """
>     - 首次请求：S1（含 I/O）→ 调 run_pipeline(raw_state, overrides=None, start_stage="S2")
>     - PATCH /config：查 raw_state 缓存 → merge overrides → 直接 run_pipeline(..., start_stage="S2")
>     - 全过程纯函数，可重入；raw_state 本身不可变
>     """
> ```

```
Stage 1: parse_structure(repo_id, config) → ParseResult
         ┌─ L0: parse_config_json()
         ├─ L1: parse_safetensors_metadata()    ← 新增关键层
         ├─ L2: load_meta_device()               ← async 增强（可能慢）
         └─ L3: parse_ast() [v1.1]
         结果逐层合并，高层覆盖低层，每个字段带 Provenance
         [纯函数性声明] 本阶段是 I/O 的"汇聚点"。HF Hub 下载、缓存读写、meta-device 实例化均在 S1
         入口独立处理，S1 产出 RawStructure 后即锁定，S2-S5 不再触达任何 I/O 或全局状态。

Stage 2: detect_features(graph, config) → DetectResult
         [纯函数] 无副作用、无 I/O、无全局状态。
         遍历所有已注册的 ArchitectureAdapter（对齐 11 §1）；adapter.detect(config) 强校验。
         严禁 `if model_type == "llama"` 分支（对齐 11 §1.3 硬性验收）——必须通过 Adapter dispatch。
         汇总为 ArchitectureProfile（含 template_id: A/B/C/G）。

Stage 3: synthesize_flows(graph, profile) → FlowResult
         [纯函数] 无副作用、无 I/O、无全局状态。
         根据 profile.features 触发对应 FlowGenerator。
         v1.0 只生成 direction="forward" 的 DataEdge；direction="backward" / gradient_flow /
         activation_checkpoint / gradient_accumulation 推迟到 v1.1。

Stage 4: estimate_resources(graph, profile, training_config?, gpu_id?) → EstimateResult
         [纯函数] 无副作用、无 I/O、无全局状态。
         v1.0 仅触发 InferenceMemoryEstimator（id="inference_v1"），估算 weights + KV cache + activations。
         GPU 规格从 backend/data/gpu-catalog.yaml 读取（在 S1 前加载到 AppContext，S4 只消费只读副本）。

Stage 5: compute_layout(graph, template_id) → LayoutResult
         [纯函数] 无副作用、无 I/O、无全局状态。
         3D 空间布局计算，根据 template_id ∈ {A, B, C, G} 分派到对应布局子策略。
```

**管线编排**（pipeline.py）：

```python
async def analyze_model(repo_id: str, options: AnalyzeOptions) -> SSE stream:
    # Phase A: 快速响应（L0 + L1，< 1s）
    config = await fetch_config(repo_id)
    parse_result = parse_config_and_safetensors(repo_id, config)
    detect_result = detect_features(parse_result.graph, config)
    flow_result = synthesize_flows(parse_result.graph, detect_result.profile)
    estimate_result = estimate_resources(flow_result.graph, detect_result.profile)
    layout_result = compute_layout(flow_result.graph, options.layout)
    yield sse_segment(data=full_result, revision=1, is_final=False)

    # Phase B: 异步增强（L2 meta-device，可能需要数秒）
    if should_try_meta_device(detect_result.profile):
        enhanced = await run_in_executor(meta_device_enhance, repo_id, config)
        if enhanced:
            # 重跑 Stage 2-5，用更精确的数据
            detect2 = detect_features(enhanced.graph, config)
            flow2 = synthesize_flows(enhanced.graph, detect2.profile)
            estimate2 = estimate_resources(flow2.graph, detect2.profile)
            layout2 = compute_layout(flow2.graph, options.layout)
            yield sse_segment(data=full_result_v2, revision=2, is_final=True)
        else:
            yield sse_segment(revision=1, is_final=True)  # L1 结果即最终结果
```

### 5.1.4 四层解析策略

| 层 | 数据源 | 覆盖率 | 精度 | Provenance | 耗时 |
|---|--------|--------|------|------------|------|
| L0 | config.json / model_index.json | 100% | 配置级 | source="config_json", confidence=EXACT | ~100ms |
| L1 | safetensors 元数据 | ~90% | 精确参数+shapes | source="safetensors_metadata", confidence=EXACT(参数量) / INFERRED(层结构) | ~200ms |
| L2 | meta-device 加载 | ~55-65% | 完整模块树 | source="meta_device", confidence=EXACT | 2-10s |
| L3 | AST 解析 (v1.1) | 取决于代码复杂度 | forward 数据流 | source="ast_parse", confidence=INFERRED | ~500ms |

**合并规则**：高层结果覆盖低层同名字段；每个字段保留最高精度来源的 Provenance。

**diffusers 独立路径**：当 L0 检测到 `model_index.json` 时，走 diffusers 专用解析流程（解析 pipeline components 配置，每个子模型独立走 L0-L2）。

### 5.1.5 特征检测器注册表

```python
# detectors/registry.py
_registry: list[type[FeatureDetector]] = []

def register(cls):
    """装饰器注册"""
    _registry.append(cls)
    return cls

class FeatureDetector(ABC):
    """所有检测器的基类"""
    feature_name: str                      # 如 "moe", "mla", "gqa"
    
    @abstractmethod
    def detect(self, graph: ModuleGraph, config: dict) -> FeatureResult | None:
        """返回 None 表示未检测到该特征"""
        ...

# detectors/moe.py
@register
class MoEDetector(FeatureDetector):
    feature_name = "moe"
    def detect(self, graph, config):
        # 检查 config 中的 num_experts / num_local_experts
        # 检查 graph 中是否存在 MoE 相关模块名
        ...
```

**v1.0 内置检测器清单**：

| 检测器 | 检测特征 | 关键信号 |
|--------|---------|---------|
| MoEDetector | MoE 路由 | num_experts, MoE 类名 |
| MLADetector | Multi-Latent Attention | MLA 类名, q_lora_rank |
| GQADetector | Grouped Query Attention | num_key_value_heads ≠ num_attention_heads |
| SlidingWindowDetector | 滑动窗口注意力 | sliding_window 配置 |
| QuantizationDetector | 量化配置 | quantization_config, bits |
| RoPEDetector | 旋转位置编码 | rope_type, rope_scaling |
| VisionDetector | 视觉编码器 | vision_config, image_size |
| AudioDetector | 音频编码器 | audio_config |
| CrossAttentionDetector | 交叉注意力 | cross_attention 类名/配置 |
| DiffusionDetector | 扩散模型 | model_index.json, scheduler_config |
| VLADetector | VLA 架构 | action_head, robot_config |
| TiedWeightsDetector | 权重共享 | tie_word_embeddings, id(p) 去重 |

### 5.1.6 准入控制与进程隔离

> **v1.0 修订（原则 1 + 原则 10）**：本节描述的 `AdmissionController`（信号量 + 内存水位 + 拒绝请求）与 `CircuitBreaker` 属于 public-launch 级过度工程。
> - **v1.0 不引入 AdmissionController 与 CircuitBreaker**（内部工具单机单 worker，不需要）。
> - **v1.0 仅保留** `sandbox.py` 的 `ProcessPoolExecutor(spawn)` + `asyncio.wait_for(timeout)` 作为最小隔离机制。
> - 涉及 `admission.acquire/release` 的调用位置（`pipeline.py` / `sandbox.py` / `diffusers_parser`）在 v1.0 实现中可删除或置为 no-op。
> - v1.1+ 若出现并发压力或开放公网，再按新 ADR 评估是否引入。以下代码保留为参考。

```python
# admission.py
class AdmissionController:
    def __init__(self, max_concurrent: int = 2, memory_threshold: float = 0.8):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._memory_threshold = memory_threshold
    
    async def acquire(self):
        await self._semaphore.acquire()
        mem = psutil.virtual_memory()
        if mem.percent / 100 > self._memory_threshold:
            self._semaphore.release()
            raise MemoryPressureError(f"内存使用 {mem.percent}% 超过阈值")
    
    def release(self):
        self._semaphore.release()

# sandbox.py
_executor = ProcessPoolExecutor(
    max_workers=2,                          # 限制并发 meta-device 加载
    mp_context=multiprocessing.get_context("spawn"),  # 必须 spawn 模式
)

async def run_meta_device(repo_id: str, config: dict, timeout: float = 30.0) -> ParseResult:
    """在独立进程中执行 meta-device 加载，返回序列化后的数据（非 nn.Module）"""
    admission = get_admission_controller()
    await admission.acquire()
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _meta_device_worker, repo_id, config),
            timeout=timeout,
        )
        return ParseResult.model_validate(result)
    finally:
        admission.release()
```

### 5.1.7 分段缓存策略

```python
# cache.py
import threading
from cachetools import TTLCache

class SegmentedCache:
    """分段缓存：每个段独立 TTL，L0 内存 + L1 文件"""
    
    SEGMENTS = {
        "config":    {"l0_ttl": 3600, "l1_ttl": 86400},    # config 变化少
        "tree":      {"l0_ttl": 1800, "l1_ttl": 43200},    # 模型树
        "params":    {"l0_ttl": 1800, "l1_ttl": 43200},    # 参数统计
        "flow":      {"l0_ttl": 900,  "l1_ttl": 21600},    # 数据流
        "estimate":  {"l0_ttl": 600,  "l1_ttl": 7200},     # 估算结果变化较快
        "layout":    {"l0_ttl": 300,  "l1_ttl": 3600},     # 布局受配置影响
    }
    
    def __init__(self):
        self._lock = threading.Lock()                        # TTLCache 非线程安全
        self._l0: dict[str, TTLCache] = {
            seg: TTLCache(maxsize=128, ttl=cfg["l0_ttl"])
            for seg, cfg in self.SEGMENTS.items()
        }
    
    def get(self, segment: str, key: str):
        with self._lock:
            val = self._l0[segment].get(key)
        if val is not None:
            return val
        return self._l1_get(segment, key)  # 文件缓存 fallback
    
    def set(self, segment: str, key: str, value):
        with self._lock:
            self._l0[segment][key] = value
        self._l1_set(segment, key, value)  # 异步写文件
```

### 5.1.8 SSE 流式推送协议

```json
// SSE segment 格式
{
  "segment": "full",           // "config" | "tree" | "flow" | "estimate" | "layout" | "full"
  "revision": 1,               // 递增版本号，1=L0+L1 快速结果，2=L2 增强结果
  "is_final": false,           // true 表示不会再有后续 revision
  "data": { ... },             // 实际数据（非进度，是真实结构化数据）
  "provenance_summary": {      // 本次数据的整体溯源
    "layers_used": ["config_json", "safetensors_metadata"],
    "overall_confidence": "inferred",
    "caveats": ["meta-device 加载进行中，稍后将推送更精确结果"]
  }
}
```

### 5.1.9 Config 字段标准化

```python
# parsing/config_normalizer.py
FIELD_ALIASES = {
    "num_attention_heads": ["n_head", "num_heads", "n_heads", "attention_heads"],
    "hidden_size": ["n_embd", "d_model", "embed_dim", "hidden_dim"],
    "num_hidden_layers": ["n_layer", "num_layers", "n_layers", "depth"],
    "intermediate_size": ["n_inner", "ffn_dim", "mlp_dim"],
    "num_key_value_heads": ["num_kv_heads", "kv_heads", "n_head_kv"],
    # ... 持续扩展
}

def normalize_config(raw_config: dict) -> dict:
    """将各模型的非标准字段名统一为 transformers 标准名"""
    normalized = {}
    for standard_name, aliases in FIELD_ALIASES.items():
        for alias in [standard_name] + aliases:
            if alias in raw_config:
                normalized[standard_name] = raw_config[alias]
                break
    # 保留未匹配的原始字段
    for k, v in raw_config.items():
        if k not in normalized and not _is_alias(k):
            normalized[k] = v
    return normalized
```

### 5.1.10 最终目录结构

```
backend/app/
├── main.py                         # FastAPI app 入口
├── pipeline.py                     # 5-stage 管线编排 + SSE 流控
├── admission.py                    # 全局准入控制器（信号量+内存水位）
├── sandbox.py                      # ProcessPoolExecutor(spawn) + 超时
├── cache.py                        # L0 内存(TTLCache+Lock) + L1 分段文件
│
├── routers/
│   ├── model.py                    # GET /model/{org}/{repo}, SSE /stream/{org}/{repo}
│   ├── compare.py                  # POST /compare
│   └── explore.py                  # WebSocket /ws/explore (v1.x)
│
├── models/
│   ├── graph.py                    # ModuleGraph, ModuleNode, DataEdge, HierarchyTree
│   ├── profile.py                  # ArchitectureProfile
│   ├── provenance.py               # Provenance, Confidence
│   ├── contracts.py                # Stage I/O 契约类型：ParseResult, DetectResult, ...
│   └── schemas.py                  # API 响应 Pydantic 模型
│
├── services/
│   ├── parsing/
│   │   ├── config_parser.py        # L0: config.json / model_index.json
│   │   ├── config_normalizer.py    # config 字段别名标准化
│   │   ├── safetensors_parser.py   # L1: safetensors 元数据解析（NEW）
│   │   ├── meta_loader.py          # L2: meta-device 加载（id(p) 去重 tied weights）
│   │   ├── ast_parser.py           # L3: AST 解析 (v1.1)
│   │   ├── diffusers_parser.py     # diffusers 专用路径（model_index.json）
│   │   └── merger.py               # 多层结果合并 + Provenance 标记
│   │
│   ├── detectors/
│   │   ├── registry.py             # 装饰器注册表
│   │   ├── base.py                 # FeatureDetector ABC
│   │   ├── moe.py                  # MoE 路由检测
│   │   ├── mla.py                  # Multi-Latent Attention
│   │   ├── gqa.py                  # Grouped Query Attention
│   │   ├── sliding_window.py       # 滑动窗口注意力
│   │   ├── quantization.py         # 量化配置
│   │   ├── rope.py                 # 旋转位置编码
│   │   ├── vision.py               # 视觉编码器
│   │   ├── audio.py                # 音频编码器
│   │   ├── cross_attention.py      # 交叉注意力
│   │   ├── diffusion.py            # 扩散模型
│   │   ├── vla.py                  # VLA 架构
│   │   └── tied_weights.py         # 权重共享
│   │
│   ├── flow_generators/
│   │   ├── registry.py             # 装饰器注册表
│   │   ├── base.py                 # FlowGenerator ABC
│   │   ├── inference.py            # 推理数据流
│   │   ├── training_forward.py     # 训练前向传播
│   │   ├── training_backward.py    # 训练反向传播
│   │   ├── multimodal.py           # 端到端多模态数据流
│   │   ├── diffusion.py            # 扩散模型数据流（去噪循环）
│   │   └── vla.py                  # VLA 数据流（感知→决策→动作）
│   │
│   ├── estimators/
│   │   ├── registry.py             # 装饰器注册表
│   │   ├── base.py                 # Estimator ABC
│   │   ├── memory.py               # 显存估算
│   │   ├── flops.py                # 计算量估算
│   │   ├── kv_cache.py             # KV Cache 估算
│   │   ├── communication.py        # 并行通信量估算
│   │   └── latency.py              # 延迟估算
│   │
│   ├── parallel/
│   │   ├── megatron.py             # Megatron-LM (TP/PP/DP/SP/CP/EP)
│   │   ├── fsdp.py                 # FSDP
│   │   └── pp_scheduler.py         # PP 调度器（1F1B / Interleaved / Zero-bubble）
│   │
│   └── layout/
│       └── engine.py               # 3D 空间布局计算
│
└── infra/
    ├── hf_client.py                # HuggingFace Hub 交互（白名单下载）
    ├── hw_specs.py                 # GPU 硬件规格数据库
    └── metrics.py                  # Prometheus 指标（v1.1）
```

---

### 5.1.11 版本路线图（架构演进）

| 版本 | 架构变更 |
|------|---------|
| v1.0 | 5-stage 纯函数管线 + SSE 双阶段推送 + 准入控制 + spawn 进程池 |
| v1.1 | + AST 解析(L3) + Prometheus 指标 + Worker 安全加固 + 前后端镜像拆分 |
| v1.2 | + ReactiveGraph 包装（零重写，同一套纯函数）+ L2 对象存储 + WebSocket 探索模式 |
| v2.0 | + 并行策略可视化（TP/PP/DP/EP/CP/SP）+ 训练 pipeline 数据流动画 |

---

以上是 Section 5.1 整体架构设计终稿，融合了两轮评审团全部 19 项建议。请确认。

---

### 5.1.12 ArchitectureAdapter 注册表（契约对齐 11 §1）

> **上级契约**：11 §1「ArchitectureAdapter」。本节是落地实现。
>
> **硬性验收（对齐 11 §1.3）**：`detect_features` / `synthesize_flows` / `compute_layout` **不得**出现 `if model_type == "..."` 分支；所有模型族分派**必须**走 Adapter registry。违反即视为架构违规，拒绝合入。

#### 接口签名

```python
# backend/adapters/base.py
from typing import Protocol, Literal

class ArchitectureAdapter(Protocol):
    """每种模型架构族对应一个 Adapter；v1.0 与 11 §1.1 一致。"""

    name: str                       # 如 "llama", "llama_moe", "deepseek_moe", "generic"
    matches: list[str]              # config.model_type 可匹配值列表
    confidence: Confidence          # EXACT（识别确定）/ INFERRED（回退到 Template G）

    def detect(self, config: dict) -> bool:
        """强校验：该 config 是否真的属于本架构族（非单字段匹配，需多特征交叉验证）。"""

    def build_graph(
        self,
        config: dict,
        safetensors_meta: dict | None,
        meta_state_dict: dict | None,
    ) -> ModuleGraph:
        """构建 ModuleGraph（节点+边+层级），所有 node/edge 均携带 Provenance。"""

    def template_id(self) -> Literal["A", "B", "C", "G"]:
        """返回前端渲染模板 ID（v1.0 固定四选一）。"""
```

#### v1.0 内置四个 Adapter

| Adapter | 文件 | matches | template_id | 识别特征 |
|---|---|---|---|---|
| `LlamaAdapter` | `backend/adapters/llama.py` | `["llama", "llama2", "llama3", "qwen2", "mistral"]` | A | RoPE + RMSNorm + SwiGLU/GatedMLP（三特征必须同时命中，见 §5.1.18 Template G 算法） |
| `LlamaMoEAdapter` | `backend/adapters/llama_moe.py` | `["mixtral", "qwen2_moe", "qwen3_moe"]` | B | LlamaAdapter 三特征 + `num_local_experts > 0` + router 为标准 top-k |
| `DeepseekMoEAdapter` | `backend/adapters/deepseek_moe.py` | `["deepseek_v2", "deepseek_v3"]` | C | MLA（`q_lora_rank` / `kv_lora_rank` 存在）+ MoE（可含 shared_expert） |
| `GenericAdapter` | `backend/adapters/generic.py` | `["*"]`（兜底） | G | 所有上面 detect 均返回 False 时命中；**v1.0 原则 8 强制：不得默认回退 A** |

**真实模型引用**：
- Template A 对应 Qwen2-0.5B / Llama-3-8B / Llama-3-70B / Mistral-7B
- Template B 对应 Mixtral-8x7B / Qwen2-57B-A14B
- Template C 对应 DeepSeek-V2 / DeepSeek-V3
- Template G 兜底：Mamba / RWKV / ViT / 任何不满足 Template A 三特征的架构

#### 注册方式（显式 register，禁止自动发现）

```python
# backend/adapters/__init__.py
from .registry import register, get_all, dispatch
from .llama import LlamaAdapter
from .llama_moe import LlamaMoEAdapter
from .deepseek_moe import DeepseekMoEAdapter
from .generic import GenericAdapter

# 注意：显式调用，严禁 entry_points / pluggy / importlib 自动发现（违反"1 文件 + 1 注册"原则）
register(LlamaAdapter())
register(LlamaMoEAdapter())
register(DeepseekMoEAdapter())
register(GenericAdapter())           # 必须最后注册；dispatch 未匹配其他时兜底
```

```python
# backend/adapters/registry.py
_registry: list[ArchitectureAdapter] = []
_generic: ArchitectureAdapter | None = None

def register(adapter: ArchitectureAdapter) -> None:
    if adapter.name == "generic":
        global _generic
        _generic = adapter
    else:
        _registry.append(adapter)

def dispatch(config: dict) -> ArchitectureAdapter:
    """按注册顺序遍历，首个 detect(config)==True 命中；均不命中 → GenericAdapter（Template G）。"""
    for a in _registry:
        if a.detect(config):
            return a
    assert _generic is not None, "GenericAdapter 必须注册，兜底 Template G"
    return _generic
```

#### 新增架构的扩展成本（对齐 11 §0.1）
新增一个架构族 = 新建 `backend/adapters/<name>.py` + 在 `__init__.py` 追加 1 行 `register(...)`。**任何改动超过"1 文件 + 1 注册"视为架构违规**。

---

### 5.1.13 MemoryEstimator 插件接口（契约对齐 11 §5）

> **上级契约**：11 §5「MemoryEstimator」。本节是落地实现。

#### 接口签名

```python
# backend/estimators/base.py
from typing import Protocol, Literal

class MemoryEstimator(Protocol):
    id: Literal["inference_v1", "megatron_tp_pp_sp", "fsdp_zero1", "fsdp_zero2", "fsdp_zero3"]
    scope: Literal["inference", "training"]
    supports: list[str]   # 如 ["weights", "kv_cache", "activations"]（v1.0）

    def estimate(
        self,
        graph: ModuleGraph,
        training_config: TrainingConfig,    # batch / seq_len / dtype / optimizer / grad_accum / ckpt
        parallelism: ParallelismPlan,       # v1.0 仅 tp=pp=dp=ep=1
        gpu: GPUSpec,                       # 来自 gpu-catalog.yaml
    ) -> MemoryBreakdown:
        """返回按类别的 per-device 显存消耗 + 总量 + GPU 容量占比。"""
```

#### v1.0 实现：`InferenceMemoryEstimator`（id="inference_v1"）

**作用域**：推理。**覆盖**：`weights` + `kv_cache` + `activations`。
**不覆盖**（v1.0 均返回 `None` 或 `0`）：`gradients` / `optimizer_states` / `comm_buffer`（留给 v1.1 的 `MegatronEstimator` / `FSDPZero1/2/3Estimator`）。

**计算公式**（SI 字节单位；`bytes_per_dtype` 表如下）：

| dtype | bytes_per_dtype |
|---|---|
| `float32` | 4 |
| `bfloat16` / `float16` | 2 |
| `float8_e4m3` / `float8_e5m2` | 1 |
| `int8` | 1 |
| `int4` | 0.5 |

```
weights_bytes
  = Σ (module.params × bytes_per_dtype(module.dtype))   遍历 graph.nodes
    - tied_weights_overcounting                          （按 id(p) 去重）

kv_cache_bytes
  = 2                                                    （K + V 两张）
  × num_kv_heads                                         （GQA 时 = num_key_value_heads；MLA 时 = 1 对 latent）
  × head_dim                                             （MLA 时 = kv_lora_rank）
  × seq_len                                              （= max_position_embeddings 或 TrainingConfig.seq_len）
  × batch                                                （推理 batch）
  × bytes_per_dtype(kv_dtype)                            （通常与 activation dtype 一致）
  × num_hidden_layers

activations_bytes
  = layer_activation_footprint × num_hidden_layers × (1 if activation_checkpointing else grad_accum_factor)
  layer_activation_footprint
    = batch × seq_len × hidden_size × bytes_per_dtype(activation_dtype) × activation_multiplier
    activation_multiplier 默认 = 4（attention scores + FFN intermediate + residual + layernorm 中间态）
    若 `activation_checkpointing=True`，只保留 checkpoint 边界激活，乘系数 1/sqrt(num_hidden_layers)
    （对齐 Megatron recomputation 论文近似，v1.0 按 sqrt 近似；v1.1 精细化）

per_device_total_bytes = weights + kv_cache + activations
utilization_ratio      = per_device_total_bytes / (gpu.memory_gb × 1024**3)
```

**Provenance**：整体 `confidence=ESTIMATED`；`caveats` 含 `"activation 使用 multiplier=4 近似"`、`"kv_cache 按最大 seq_len 按满计算"` 等。

#### v1.1+ 预留接口

- `MegatronEstimator`（id="megatron_tp_pp_sp"）：含 gradients / optimizer_states / comm_buffer；分片按 TP+PP+SP 计算。
- `FSDPZero1Estimator` / `FSDPZero2Estimator` / `FSDPZero3Estimator`：FSDP ZeRO-1/2/3 分片策略。

v1.0 registry 中仅注册 `InferenceMemoryEstimator`；`GET /api/v1/memory-estimators` 返回单条 `[{"id":"inference_v1",...}]`。

---

### 5.1.14 GPU Catalog 数据表（契约对齐 11 §6）

> **上级契约**：11 §6「GPU Catalog」。
> **硬性规则**：**严禁**在任何代码中硬编码 GPU 规格（memory、带宽、tflops）；所有估算通过 `backend/data/gpu-catalog.yaml` 消费；新增 GPU = 改 yaml，无需改代码。

#### 文件位置

`backend/data/gpu-catalog.yaml`

#### Schema（每条记录字段）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | 是 | 主键，如 `a100_80g_sxm`、`h100_80g_sxm5`、`kunlun_p800` |
| `vendor` | string | 是 | `nvidia` / `huawei` / `baidu` / `cambricon` / `amd` |
| `arch` | string | 是 | `ampere` / `hopper` / `ada_lovelace` / `blackwell` / 国产架构名 |
| `memory_gb` | int | 是 | 显存容量 |
| `memory_bandwidth_gbps` | int | 是 | HBM/GDDR 带宽 |
| `fp16_tflops` | float | 是 | FP16 峰值算力 |
| `bf16_tflops` | float | 是 | BF16 峰值算力 |
| `fp8_tflops` | float \| null | 否 | 不支持 FP8 时填 `null`（Ampere / 消费卡） |
| `nvlink_gbps` | int \| null | 否 | NVLink 带宽；消费卡 / 国产卡按实际填写或 null |
| `tdp_w` | int | 是 | TDP 瓦数 |
| `release_year` | int | 是 | 发布年份 |

#### v1.0 必含 9 款（对齐 README「GPU 选型 8 款 + 国产 1 款补足」与 11 §6.3 精选集）

```yaml
- id: a100_40g_sxm
  vendor: nvidia
  arch: ampere
  memory_gb: 40
  memory_bandwidth_gbps: 1555
  fp16_tflops: 312
  bf16_tflops: 312
  fp8_tflops: null
  nvlink_gbps: 600
  tdp_w: 400
  release_year: 2020

- id: a100_80g_sxm
  vendor: nvidia
  arch: ampere
  memory_gb: 80
  memory_bandwidth_gbps: 2039
  fp16_tflops: 312
  bf16_tflops: 312
  fp8_tflops: null
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
  arch: hopper
  memory_gb: 141
  memory_bandwidth_gbps: 4800
  fp16_tflops: 989
  bf16_tflops: 989
  fp8_tflops: 1979
  nvlink_gbps: 900
  tdp_w: 700
  release_year: 2024

- id: rtx_4090_24g
  vendor: nvidia
  arch: ada_lovelace
  memory_gb: 24
  memory_bandwidth_gbps: 1008
  fp16_tflops: 165
  bf16_tflops: 165
  fp8_tflops: 330
  nvlink_gbps: null
  tdp_w: 450
  release_year: 2022

- id: rtx_3090_24g
  vendor: nvidia
  arch: ampere
  memory_gb: 24
  memory_bandwidth_gbps: 936
  fp16_tflops: 71
  bf16_tflops: 71
  fp8_tflops: null
  nvlink_gbps: null
  tdp_w: 350
  release_year: 2020

- id: l40s_48g
  vendor: nvidia
  arch: ada_lovelace
  memory_gb: 48
  memory_bandwidth_gbps: 864
  fp16_tflops: 362
  bf16_tflops: 362
  fp8_tflops: 733
  nvlink_gbps: null
  tdp_w: 350
  release_year: 2023

- id: ascend_910b_64g
  vendor: huawei
  arch: ascend
  memory_gb: 64
  memory_bandwidth_gbps: 1600
  fp16_tflops: 376
  bf16_tflops: 376
  fp8_tflops: null
  nvlink_gbps: 392      # HCCS
  tdp_w: 400
  release_year: 2023

- id: kunlun_p800
  vendor: baidu
  arch: xpu3
  memory_gb: 64
  memory_bandwidth_gbps: 1600
  fp16_tflops: 345
  bf16_tflops: 345
  fp8_tflops: null
  nvlink_gbps: null
  tdp_w: 400
  release_year: 2024
```

#### 加载与端点

- 进程启动时由 `AppContext` 一次性加载到内存，S4 只消费只读副本（保持纯函数）。
- `GET /api/v1/gpus` 直接返回该数组（见 04 §4.9.1）。
- 未知 `gpu_id` → `404 GPU_SPEC_NOT_FOUND`（见 04 §4.4）。

---

### 5.1.15 ParallelismStrategy 插件接口（v1.0 仅定义，契约对齐 11 §4 + ADR-020）

> **v1.0 定位**：**仅定义接口 + 注册空数组**，无任何内置实现。TP / PP / DP / EP / CP / SP 及其组合**全部推迟到 v1.1+**（对应 README v1.0 范围冻结：并行策略可视化 → v1.2+）。

#### 接口签名

```python
# backend/parallelism/base.py
from typing import Protocol, Literal

class ParallelismStrategy(Protocol):
    id: Literal["TP", "PP", "DP", "EP", "CP", "SP", "TP+PP", "TP+PP+DP", ...]

    def partition(self, graph: ModuleGraph, world_size: int, config: dict) -> PartitionedGraph:
        """声明切分 / 复制规则；每个节点标注 (tp_rank, pp_stage, dp_rank, ep_rank)。"""

    def communication_ops(self, partitioned: PartitionedGraph) -> list[CommOp]:
        """返回通信原语序列：AllReduce / AllGather / ReduceScatter / All2All / P2P。"""

    def animation_script(self, partitioned: PartitionedGraph) -> ParallelismAnimationScript:
        """前端动画脚本：时序 + 设备布局 + 数据流箭头。"""
```

#### Registry（v1.0 为空）

```python
# backend/parallelism/__init__.py
from .registry import register, get_all

# v1.0：不注册任何实现。GET /api/v1/parallelism-strategies 返回 []。
# v1.1+ 起逐步加入 TensorParallel() / PipelineParallel() / ...
```

#### v1.1+ 规划

对齐 11 §4.3 验收：必须支持 Megatron-LM 论文 5 种基础策略（TP/PP/DP/SP/CP）+ 3 种组合（TP+PP、TP+PP+DP、TP+PP+DP+EP），且 DeepSeek-V3 4D 并行（TP+PP+DP+EP）必须可用插件实现。

---

### 5.1.16 DataFlowDirection 一等公民（契约对齐 11 §7 + ADR-018）

> DataEdge 的 `edge_type` 与 `direction` 字段已在 §5.1.2 更新。此处规定 S3 `synthesize_flows` 的产出边界。

#### v1.0 产出

- S3 **只生成** `direction="forward"` 的 `DataEdge`。
- 允许 `edge_type ∈ {"data_flow", "residual", "skip_connection", "branch_merge"}`。
- 禁止在 v1.0 产出 `gradient_flow` / `activation_checkpoint` / `gradient_accumulation` 及 `direction="backward"` 的边——前端如收到应按 schema 向前兼容忽略，不得崩溃。

#### v1.1+ 产出

- 启用 `direction="backward"`：S3 新增子模块 `backward_synthesizer`，基于 forward graph 镜像生成 gradient edges。
- 启用 `activation_checkpoint` / `gradient_accumulation` 边类型以可视化训练时序。
- 动画模式四选一：`forward_only` / `backward_only` / `forward_backward_split_screen` / `forward_backward_overlay`（见 11 §7.2）。

---

### 5.1.17 热更新链路（PATCH /config，契约对齐 04 §4.6 + 11 §8）

> **硬约束**：后端处理 < **200ms**（原则 5 交互硬约束例外条款，不因"前期不做性能优化"豁免）。端到端 < 300ms（含 WS 往返 + 前端重渲染）。

#### 链路时序

```
1. Client                  PATCH /api/v1/stream/{org}/{repo}/config
                           Body: { overrides: {...}, session_id: "<uuid>" }
                           
2. FastAPI route           校验 overrides 字段级约束（见 schemas）
                             ├─ 类型/范围越界 → 400 CONFIG_OVERRIDE_INVALID（RFC 7807）
                             └─ 跨字段冲突（如 num_experts_per_tok > num_experts）
                                → 422 CONFIG_OVERRIDE_IMPOSSIBLE

3. SessionStore            raw_state = raw_state_cache.get(session_id)
                             ├─ 命中 → 立即复用（跳过 HF Hub / L1 磁盘缓存）
                             └─ 未命中 / 已过期 → 404 SESSION_NOT_FOUND（需重开 SSE）

4. Merge overrides         merged_config = deep_merge(raw_state.config, overrides)

5. Pipeline                run_pipeline(
                             raw_state.with_config(merged_config),
                             overrides=overrides,
                             start_stage="S2",
                           )
                           ── 跳过 S1 I/O，直接从 S2 detect → S3 synthesize →
                              S4 estimate → S5 layout。全程纯函数。

6. WS emit                 WebSocketHub.publish(session_id, {
                             type: "graph_update",
                             revision: raw_state.revision + 1,
                             source: "config_override",
                             data: PipelineResult.to_schema(),
                             provenance_summary: {...}
                           })
                           revision += 1 且持久化到 SessionStore

7. HTTP response           204 No Content（立即返回，不阻塞 WS 推送）
```

#### raw_state 常驻内存缓存

```python
from cachetools import TTLCache
import threading

class SessionStore:
    """
    存放 PATCH /config 所需的 raw_state（S1 产物）；
    key = session_id (UUID)；TTL 30 min；maxsize 256（内部工具单机足够）。
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cache: TTLCache[str, RawStructure] = TTLCache(maxsize=256, ttl=30 * 60)

    def put(self, session_id: str, raw: RawStructure) -> None:
        with self._lock: self._cache[session_id] = raw

    def get(self, session_id: str) -> RawStructure | None:
        with self._lock: return self._cache.get(session_id)
```

**写入时机**：初始 SSE `GET /api/v1/stream/{org}/{repo}` 在 `is_final=true` 前将 S1 产物（含解析后的 config + safetensors_meta，**不含** meta-device state_dict 以控内存）写入 `SessionStore`。

#### 错误契约（RFC 7807）

- `400 CONFIG_OVERRIDE_INVALID`：overrides 单字段越界（type/range）。
- `422 CONFIG_OVERRIDE_IMPOSSIBLE`：跨字段语义冲突（如 `num_experts_per_tok > num_experts`、`num_attention_heads % num_key_value_heads != 0`）。
- `404 SESSION_NOT_FOUND`：session_id 不存在或 TTL 过期。
- 校验失败**不推 WS 事件**。

#### 预算拆解（硬约束 < 200ms）

| 子步骤 | 预算 |
|---|---|
| overrides 字段级 + 跨字段校验 | < 5ms |
| SessionStore 查找 + deep_merge | < 5ms |
| S2 detect_features | < 30ms |
| S3 synthesize_flows | < 50ms |
| S4 estimate_resources（InferenceMemoryEstimator） | < 30ms |
| S5 compute_layout（Block 级，约数百节点） | < 50ms |
| WS publish（本机） | < 30ms |
| **合计** | **< 200ms** |

若任一项超预算，属于"交互响应延迟"硬约束违规，必须立即修复（不延期到 Phase N 性能优化阶段）。

---

### 5.1.18 WebSocket 增量推送（契约对齐 04 §4.7）

#### 端点

```
WS /api/v1/stream/{org}/{repo}/updates?session_id={uuid}
```

#### 消息 schema（Server → Client）

```json
{
  "type": "graph_update",
  "revision": 3,
  "source": "config_override",
  "data": {
    "graph":    { "nodes": {...}, "edges": [...], "hierarchy": {...}, "provenance": {...} },
    "profile":  { "model_type": "...", "features": [...], "template_id": "A|B|C|G", ... },
    "estimate": { "memory": {...}, "flops": {...}, "provenance": {...} },
    "layout":   { "positions": {...}, "camera": {...} }
  },
  "provenance_summary": {
    "layers_used": ["config_json", "config_override"],
    "overall_confidence": "inferred",
    "caveats": ["参数由用户覆盖，非源自 HF Hub"]
  }
}
```

与 SSE `segment` 帧的 `data` 字段 **1:1 对齐**，前端 store 可复用 `replaceSnapshot()` 逻辑。

#### 连接管理

- 每个 `session_id` 对应至多 1 条活跃 WS；重复建立 → 后建替换前者（发送 1001 going-away 给旧连接）。
- 断线重连：客户端使用相同 `session_id` 重连；服务端根据 `revision` 自动对齐，无需客户端显式 resync。
- 心跳：服务端 15s 间隔下发 ping 帧；客户端未回应 pong 连续 2 次 → 服务端主动关闭。

#### 生命周期

```python
class WebSocketHub:
    """单进程内 session_id → WebSocket 路由表。"""
    def __init__(self):
        self._conns: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def register(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            prev = self._conns.pop(session_id, None)
            if prev is not None:
                await prev.close(code=1001)
            self._conns[session_id] = ws

    async def publish(self, session_id: str, payload: dict) -> None:
        ws = self._conns.get(session_id)
        if ws is not None:
            await ws.send_json(payload)
```

---

### 5.1.19 Provenance 全字段强制（契约对齐 ADR-016 / 原则 9）

**硬性规则**：以下 schema 每一个实例，**必须**携带 `Provenance` 字段或引用根级 `Provenance`，任何裸数据属契约违规：

| Schema | Provenance 位置 |
|---|---|
| `ModuleNode` | 节点自带 `provenance: Provenance` |
| `DataEdge` | 边自带 `provenance: Provenance` |
| `ArchitectureProfile` | 对象自带 `provenance: Provenance` |
| `MemoryBreakdown` | 对象自带 `provenance: Provenance` |
| `EstimateResult` | 对象自带 `provenance: Provenance` |
| `ModuleGraph` | 根自带 `provenance: Provenance`（刻画整图可信度） |

**响应头**：所有主路径响应（SSE `segment` / WS `graph_update` / `GET /model/...`）必须携带：

```
X-Provenance-Summary: layers_used=config_json,safetensors_metadata;confidence=inferred
```

前端徽标展示可**只依赖响应头**，无需解析 body（原则 9 落地的工程约束）。

---

### 5.1.20 Template G 判定算法（契约对齐 ADR-015 + 原则 8）

> **原则 8**：对不认识的架构走 Template G 通用回退并明确打 `INFERRED` 徽标，**绝不默认回退成 LLaMA 骗用户**。

#### `LlamaAdapter.detect()` 实现（三特征必须同时命中）

```python
def _matches_llama_family(config: dict) -> bool:
    """Template A 三特征检测：RoPE + RMSNorm + SwiGLU/GatedMLP 必须同时命中。"""
    has_rope = (
        config.get("rope_type") is not None
        or config.get("rope_scaling") is not None
        or config.get("rope_theta") is not None
    )
    has_rmsnorm = (
        config.get("rms_norm_eps") is not None
        or config.get("layer_norm_type", "").lower() in {"rmsnorm", "rms"}
    )
    has_gated_mlp = (
        config.get("hidden_act", "").lower() in {"silu", "swiglu", "gelu_pytorch_tanh"}
        and config.get("intermediate_size") is not None
    )
    return has_rope and has_rmsnorm and has_gated_mlp

class LlamaAdapter:
    name = "llama"
    matches = ["llama", "llama2", "llama3", "qwen2", "mistral"]
    confidence = Confidence.EXACT

    def detect(self, config: dict) -> bool:
        return (
            config.get("model_type") in self.matches
            and _matches_llama_family(config)        # 关键：不仅看 model_type，还验三特征
        )
```

#### dispatch 策略（兜底必为 G）

```
for adapter in [LlamaAdapter, LlamaMoEAdapter, DeepseekMoEAdapter]:
    if adapter.detect(config):
        return adapter
# 任一未命中 → GenericAdapter（Template G）
# 即使 config.model_type == "llama" 但三特征缺失 → 仍走 Template G（不得伪装 EXACT）
return GenericAdapter()
```

`GenericAdapter` 返回的 `ArchitectureProfile.provenance.confidence = INFERRED`，前端据此显示"通用回退（未完全识别）"徽标。

---


---

### 5.2 模块职责划分

按目录结构逐模块说明：**职责边界 → 对外接口 → 依赖关系 → 内部关键设计**。

---

### 5.2.1 `routers/` — 接口层

#### `routers/model.py`

| 项 | 说明 |
|---|------|
| **职责** | 单模型分析入口：同步快速查询 + SSE 流式推送 |
| **端点** | `GET /api/v1/model/{org}/{repo}` — 返回 L0+L1 快照（JSON）<br>`GET /api/v1/stream/{org}/{repo}` — SSE 流，推送 revision 1（快速）→ revision 2（增强） |
| **依赖** | → `pipeline.analyze_model()` / `pipeline.analyze_model_stream()` |
| **不做** | 不含任何业务逻辑，仅参数校验 + 调用 pipeline + 格式化响应 |

#### `routers/compare.py`

| 项 | 说明 |
|---|------|
| **职责** | 多模型对比 |
| **端点** | `POST /api/v1/compare` — body: `{models: [repo_id, ...], aspects: [...]}` |
| **依赖** | → `pipeline.analyze_model()` × N，并行调用后做 diff |
| **内部逻辑** | 参数量差异、层结构差异、特征差异、估算对比表 |

#### `routers/explore.py`（v1.x）

| 项 | 说明 |
|---|------|
| **职责** | 交互式模型探索（修改 config → 增量重算 → 推送更新） |
| **端点** | `WebSocket /api/v1/ws/explore` |
| **协议** | 客户端发送 config patch → 后端校验约束 → 重跑 Stage 2-5 → 推送增量结果 |
| **依赖** | → `pipeline.incremental_recompute()` |

---

### 5.2.2 `pipeline.py` — 管线编排

| 项 | 说明 |
|---|------|
| **职责** | 编排 5 个 Stage 的执行顺序、SSE 双阶段推送、缓存查询/回填 |
| **对外接口** | `analyze_model(repo_id, options) → FullResult`<br>`analyze_model_stream(repo_id, options) → AsyncGenerator[SSESegment]`<br>`incremental_recompute(graph, config_patch, options) → FullResult`（v1.x） |
| **依赖** | → `cache.SegmentedCache`<br>→ `sandbox.run_meta_device()`<br>→ `admission.AdmissionController`<br>→ 各 Stage 纯函数 |
| **关键设计** | 1. 每个 Stage 调用前先查缓存，命中则跳过<br>2. Phase A（L0+L1）同步完成后立即 yield revision=1<br>3. Phase B（L2）通过 `run_in_executor` 异步执行，完成后 yield revision=2<br>4. 任何 Stage 失败不中断管线，降级返回已有数据 + Provenance 标记 caveats |

---

### 5.2.3 `admission.py` — 准入控制器

| 项 | 说明 |
|---|------|
| **职责** | 限制并发 meta-device 加载数量，防止内存爆炸 |
| **对外接口** | `acquire() → None`（阻塞等待信号量 + 检查内存水位）<br>`release() → None` |
| **配置** | `max_concurrent: int = 2`，`memory_threshold: float = 0.8` |
| **依赖** | → `psutil`（内存水位检测） |
| **单例** | 全局单例，`pipeline.py` 和 `sandbox.py` 共享同一实例 |

---

### 5.2.4 `sandbox.py` — 进程隔离

| 项 | 说明 |
|---|------|
| **职责** | 在独立 spawn 进程中执行 meta-device 加载，超时控制，结果序列化 |
| **对外接口** | `run_meta_device(repo_id, config, timeout=30) → ParseResult` |
| **关键约束** | 1. 必须 `spawn` 模式（不继承父进程 GPU 状态）<br>2. Worker 函数返回序列化后的 `dict`（非 `nn.Module`）<br>3. Worker 内部用 `id(p)` 做 tied weights 去重<br>4. 超时后进程强制终止 |
| **依赖** | → `admission.AdmissionController`<br>→ `services.parsing.meta_loader._meta_device_worker()` |

---

### 5.2.5 `cache.py` — 分段缓存

| 项 | 说明 |
|---|------|
| **职责** | 6 段独立缓存（config/tree/params/flow/estimate/layout），L0 内存 + L1 文件 |
| **对外接口** | `get(segment, key) → value \| None`<br>`set(segment, key, value) → None`<br>`invalidate(segment, key) → None`<br>`invalidate_model(repo_id) → None`（清除该模型所有段） |
| **线程安全** | `threading.Lock` 保护所有 L0 TTLCache 操作 |
| **L1 文件格式** | `{cache_dir}/{segment}/{sha256(key)}.json`，用 orjson 序列化 |
| **L1 过期** | 文件 mtime + TTL 判断，惰性清理 + 定期 GC |

---

### 5.2.6 `models/` — 领域模型

| 文件 | 职责 | 关键类 |
|------|------|--------|
| `graph.py` | 核心 DAG 数据结构 | `ModuleGraph`, `ModuleNode`, `DataEdge`, `HierarchyTree` |
| `profile.py` | 架构特征画像 | `ArchitectureProfile` |
| `provenance.py` | 统一溯源 | `Provenance`, `Confidence` |
| `contracts.py` | Stage 管线 I/O 契约 | `ParseResult`, `DetectResult`, `FlowResult`, `EstimateResult`, `LayoutResult` |
| `schemas.py` | API 响应模型 | `ModelResponse`, `CompareResponse`, `SSESegment`, `ErrorResponse` |

**设计原则**：所有模型均为 Pydantic `BaseModel`，不可变（`frozen=True` 或仅通过 `model_copy(update=...)` 产生新实例），Stage 之间通过契约类型传递。

---

### 5.2.7 `services/parsing/` — 解析层（Stage 1）

#### `config_parser.py` — L0

| 项 | 说明 |
|---|------|
| **职责** | 下载并解析 config.json / model_index.json，构建初始 ModuleGraph 骨架 |
| **输入** | repo_id |
| **输出** | `ParseResult`（基础节点 + 空 edges + 初步 hierarchy） |
| **依赖** | → `infra.hf_client`（下载）→ `config_normalizer`（字段标准化） |

#### `config_normalizer.py`

| 项 | 说明 |
|---|------|
| **职责** | 将各模型非标准 config 字段名映射为 transformers 标准名 |
| **输入** | `raw_config: dict` |
| **输出** | `normalized_config: dict` |
| **维护** | `FIELD_ALIASES` 映射表，新模型只需追加别名 |

#### `safetensors_parser.py` — L1（新增关键模块）

| 项 | 说明 |
|---|------|
| **职责** | 通过 `huggingface_hub.get_safetensors_metadata()` 获取精确参数量、tensor shapes、推断层结构 |
| **输入** | repo_id |
| **输出** | `ParseResult`（精确参数量、每层 tensor shapes、推断的层级结构） |
| **核心逻辑** | 1. 解析 safetensors index → 得到所有 tensor name + shape + dtype<br>2. 从 tensor naming pattern 推断层结构（如 `model.layers.31.self_attn.q_proj.weight` → 32 层）<br>3. 精确计算每层参数量（shape 相乘再求和）<br>4. 参数量标记 `confidence=EXACT`，层结构推断标记 `confidence=INFERRED` |
| **覆盖率** | ~90%+ 模型有 safetensors 格式，无需执行任何代码 |

#### `meta_loader.py` — L2

| 项 | 说明 |
|---|------|
| **职责** | `AutoModel.from_pretrained(repo_id, device_map="meta")` 加载完整模块树 |
| **输入** | repo_id, config |
| **输出** | `ParseResult`（完整 nn.Module 树序列化、tied weights、所有子模块类名） |
| **关键细节** | 1. `trust_remote_code=False`（安全约束）<br>2. Tied weights 用 `id(p)` 判断（`data_ptr()` 在 meta 设备返回 0）<br>3. 返回序列化 dict，不返回 nn.Module 对象<br>4. 在 sandbox spawn 进程中执行 |

#### `diffusers_parser.py`

| 项 | 说明 |
|---|------|
| **职责** | diffusers 模型专用路径：解析 `model_index.json` 中的 pipeline components |
| **输入** | repo_id, model_index_config |
| **输出** | `ParseResult`（多组件组合图：UNet/VAE/TextEncoder/Scheduler 等） |
| **逻辑** | 遍历 model_index.json 的每个 component，对每个子模型独立调用 L0→L1→L2 |

#### `merger.py`

| 项 | 说明 |
|---|------|
| **职责** | 合并多层解析结果，执行 Provenance 标记 |
| **规则** | 同一字段：L2 > L1 > L0 覆盖，保留最高精度来源的 Provenance<br>不同字段：合并取并集<br>冲突字段：保留高层值，低层值记入 `caveats` |

#### `ast_parser.py`（v1.1）

| 项 | 说明 |
|---|------|
| **职责** | AST 解析 `modeling_*.py`，提取 forward() 数据流 |
| **版本** | v1.1 实现 |

---

### 5.2.8 `services/detectors/` — 特征检测（Stage 2）

| 项 | 说明 |
|---|------|
| **整体职责** | 遍历注册表中所有 FeatureDetector，对 ModuleGraph + config 进行特征匹配，汇总为 ArchitectureProfile |
| **注册机制** | `@register` 装饰器，无需修改注册表代码即可新增检测器 |
| **扩展方式** | 新增检测器：创建文件 → 实现 `FeatureDetector.detect()` → 加 `@register` → 完成 |

**每个检测器职责边界**：

| 检测器 | 输入信号（config 字段 / graph 模块名） | 输出 metadata |
|--------|--------------------------------------|---------------|
| MoEDetector | `num_experts`, `num_local_experts`, MoE 类名 | `{num_experts, top_k, capacity_factor, shared_expert}` |
| MLADetector | MLA 类名, `q_lora_rank`, `kv_lora_rank` | `{q_lora_rank, kv_lora_rank, rope_head_dim}` |
| GQADetector | `num_key_value_heads ≠ num_attention_heads` | `{num_kv_heads, num_groups}` |
| SlidingWindowDetector | `sliding_window`, `max_window_layers` | `{window_size, window_layers}` |
| QuantizationDetector | `quantization_config` | `{method, bits, group_size, sym}` |
| RoPEDetector | `rope_type`, `rope_scaling`, `rope_theta` | `{type, scaling_type, theta, max_position}` |
| VisionDetector | `vision_config`, `image_size` | `{patch_size, image_size, vision_hidden_size}` |
| AudioDetector | `audio_config` | `{sample_rate, feature_size}` |
| CrossAttentionDetector | cross_attention 类名 | `{cross_attn_layers}` |
| DiffusionDetector | `model_index.json` 存在 | `{pipeline_class, schedulers, components}` |
| VLADetector | `action_head`, `robot_config` | `{action_dim, action_space}` |
| TiedWeightsDetector | `tie_word_embeddings`, graph 中 `id(p)` 相同节点 | `{tied_pairs: [[node_a, node_b], ...]}` |

---

### 5.2.9 `services/flow_generators/` — 数据流生成（Stage 3）

| 项 | 说明 |
|---|------|
| **整体职责** | 根据 ArchitectureProfile.features 触发对应的 FlowGenerator，填充 ModuleGraph.edges |
| **注册机制** | 同 detectors，`@register` 装饰器 |
| **触发条件** | 每个 generator 声明 `required_features: list[str]`，profile.features 满足时触发 |

| 生成器 | 触发特征 | 生成的 edges |
|--------|---------|-------------|
| `inference.py` | 所有模型 | 推理时 token 从 embedding → layers → head 的完整前向数据流 |
| `training_forward.py` | 所有模型 | 训练前向传播数据流（含 loss 计算） |
| `training_backward.py` | 所有模型 | 反向传播梯度流（edges 反向 + gradient 标记） |
| `multimodal.py` | `["vision"]` 或 `["audio"]` | 端到端多模态流：图像/音频编码 → 投影 → 融合 → 语言模型 |
| `diffusion.py` | `["diffusion"]` | 去噪循环：noise → UNet(timestep,condition) → denoise → decode |
| `vla.py` | `["vla"]` | 感知 → 推理 → 决策 → 动作空间映射 |

---

### 5.2.10 `services/estimators/` — 资源估算（Stage 4）

| 项 | 说明 |
|---|------|
| **整体职责** | 计算显存占用、FLOPS、KV Cache、通信量、延迟估算 |
| **所有输出** | `confidence=ESTIMATED`，附带计算公式说明 |
| **硬件参数化** | 所有估算通过 `infra.hw_specs` 获取 GPU 规格，非硬编码 |

| 估算器 | 输入 | 输出 |
|--------|------|------|
| `memory.py` | graph（参数量+dtype）, parallel_config | 模型权重显存、优化器状态显存、激活值显存、总显存 |
| `flops.py` | graph（tensor shapes）, batch_size, seq_len | 前向 FLOPS、反向 FLOPS、MFU 估算 |
| `kv_cache.py` | profile（GQA/MLA 特征）, seq_len, batch_size | KV Cache 显存、per-token 增量 |
| `communication.py` | graph, parallel_config | All-reduce / All-gather / Reduce-scatter / P2P 通信量 |
| `latency.py` | flops, communication, hw_specs | 计算延迟、通信延迟、流水线气泡率 |

---

### 5.2.11 `services/parallel/` — 并行策略

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

### 5.2.12 `services/layout/engine.py` — 3D 布局（Stage 5）

| 项 | 说明 |
|---|------|
| **职责** | 将 ModuleGraph 映射到 3D 空间坐标 |
| **输入** | `ModuleGraph`, `LayoutConfig`（模式、间距、分组方式） |
| **输出** | `LayoutResult`（每个节点的 x/y/z/尺寸 + 相机位置 + 包围盒） |
| **布局模式** | `by_layer`（按层纵向堆叠）、`by_type`（按模块类型分区）、`by_parallel`（按并行分片分区） |
| **尺寸映射** | 节点体积 ∝ 参数量对数，颜色 ∝ 模块类型 |

---

### 5.2.13 `infra/` — 基础设施层

#### `hf_client.py`

| 项 | 说明 |
|---|------|
| **职责** | HuggingFace Hub 文件下载，白名单过滤 |
| **白名单** | `config.json`, `model_index.json`, `tokenizer_config.json`, `generation_config.json`, `preprocessor_config.json`, `*.safetensors.index.json`, `modeling_*.py`（v1.1 AST 用）等 |
| **黑名单** | `*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`, `*.gguf`（所有权重文件） |
| **接口** | `fetch_config(repo_id) → dict`<br>`fetch_safetensors_index(repo_id) → dict \| None`<br>`fetch_file(repo_id, filename) → bytes`<br>`list_repo_files(repo_id) → list[str]` |

#### `hw_specs.py`

| 项 | 说明 |
|---|------|
| **职责** | GPU 硬件规格数据库，参数化所有估算公式 |
| **内置规格** | A100-40G, A100-80G, H100-80G, H800, L40S, RTX 4090, ... |
| **字段** | `memory_gb`, `bandwidth_gbps`, `flops_fp16`, `flops_bf16`, `flops_fp32`, `nvlink_bandwidth`, `pcie_bandwidth` |
| **扩展** | 支持用户自定义 GPU 规格（通过 API 参数传入） |

#### `metrics.py`（v1.1）

| 项 | 说明 |
|---|------|
| **职责** | Prometheus 指标收集 |
| **指标** | `model_parse_duration_seconds`, `meta_device_load_duration_seconds`, `cache_hit_total`, `cache_miss_total`, `admission_rejected_total`, `active_meta_device_loads` |

---

### 5.2.14 模块依赖关系总览

```
routers/*
  └→ pipeline.py
       ├→ cache.py
       ├→ admission.py
       ├→ sandbox.py
       │    └→ admission.py
       │    └→ services/parsing/meta_loader.py
       ├→ services/parsing/*        (Stage 1)
       │    └→ infra/hf_client.py
       ├→ services/detectors/*      (Stage 2)
       ├→ services/flow_generators/* (Stage 3)
       ├→ services/estimators/*     (Stage 4)
       │    └→ infra/hw_specs.py
       ├→ services/parallel/*       (Stage 4 可选)
       └→ services/layout/*         (Stage 5)

models/*  ← 被所有层引用，自身零依赖
infra/*   ← 被 Domain 层引用，自身仅依赖外部库
```

**依赖规则**：
- `models/` 不依赖任何其他内部模块
- `services/` 之间不互相依赖（通过 `pipeline.py` 编排串联）
- `infra/` 不依赖 `services/` 或 `routers/`
- `routers/` 仅依赖 `pipeline.py` + `models/schemas.py`

---

以上是 Section 5.2 模块职责划分。请确认。

---

### 5.2 补充 — 模块评审修订（46 条建议 + 修订后目录结构）

四位评审者共产出 **46 条建议**，按主题整理如下：

---

### 一、模块边界与拆分（架构师 + Python 工程师共识）

| # | 问题 | 建议 | 优先级 |
|---|------|------|--------|
| 1 | **pipeline.py 职责过重**（编排 + SSE + 缓存 = god object） | 拆为 `pipeline.py`（纯编排）+ `sse_emitter.py`（SSE 协议）+ `cache_strategy.py`（缓存策略）| P0 |
| 2 | **Stage 1 缺少内部编排器**（6 文件协作无显式 orchestrator） | 在 `parsing/` 内增加 `orchestrator.py`，纯函数 `build_layer_plan()` + `execute_layer_plan()` | P1 |
| 3 | **merger.py 角色混乱**（既是 peer 又是 coordinator） | 提升为 Stage 1 的 orchestrator，或重命名为 `layer_merger.py` 并作为唯一入口 | P2 |
| 4 | **compare 逻辑放在 router 中违反职责边界** | Application 层新增 `comparison.py`，router 只做参数解析 | P1 |
| 5 | **config_normalizer.py 粒度过细** | 可合入 config_parser 或改为子包 `config_parser/_aliases.py` | P3 |

---

### 二、契约与类型安全（架构师 + Python 工程师共识）

| # | 问题 | 建议 | 优先级 |
|---|------|------|--------|
| 6 | **Stage 间缺少降级类型表达** | 引入 `StageOutcome[T]`（value + degraded_layers + diagnostics）+ `PipelineTrace` | P0 |
| 7 | **contracts.py 与 schemas.py 边界模糊** | schemas.py 定义 `from_stage_results()` 工厂方法作为 API 兼容性防火墙 | P1 |
| 8 | **features 是 `set[str]`，隐式契约** | 定义 `FeatureTag` 枚举，检测器/生成器均引用枚举值 | P1 |
| 9 | **热路径性能**：Pydantic v2 对数千节点的序列化约 50-80ms | 内部管线用 `dataclass(slots=True)`，API 边界用 Pydantic `BaseModel(from_attributes=True)` | P2 |

---

### 三、依赖规则保障（架构师 + Python 工程师共识）

| # | 问题 | 建议 | 优先级 |
|---|------|------|--------|
| 10 | **sandbox.py 直接 import meta_loader = 穿透依赖** | sandbox 只接收 `Callable`，由 pipeline 传入 worker 函数 | P0 |
| 11 | **meta_loader 可能向上依赖 admission** | admission 控制在 pipeline 层执行，meta_loader 保持纯 Domain Service | P0 |
| 12 | **hw_specs 与 estimators 隐式耦合** | 在 models/ 中定义 `HardwareProfile` 领域模型，estimators 依赖它而非 infra | P1 |
| 13 | **cache.py 策略/机制混合** | infra/cache.py 只做纯机制（get/put bytes），application/cache_strategy.py 做业务策略 | P1 |
| 14 | **CI 架构守护测试** | 10 行 import 检查脚本，每次 CI 运行 <1s | P1 |

---

### 四、parallel/ 模块归属（Python 工程师 + 架构师共识）

| # | 问题 | 建议 | 优先级 |
|---|------|------|--------|
| 15 | **parallel/ 同时被 Stage 4 和 Stage 5 消费，归属不清** | 提升为与 models/ 同级的独立领域计算模块；pipeline 先调 parallel → 输出供 estimator 和 layout 共享 | P1 |
| 16 | **megatron.py 承载 6 种策略过重** | 拆为 tensor_parallel.py / pipeline_parallel.py / data_parallel.py / sequence_parallel.py / context_parallel.py / expert_parallel.py + strategy_composer.py | P1 |

---

### 五、注册表模式优化（Python 工程师）

| # | 问题 | 建议 | 优先级 |
|---|------|------|--------|
| 17 | **三份几乎相同的 registry.py** | 抽取泛型 `Registry[T]`（~40 行），放 `services/_registry.py` | P2 |
| 18 | **parsing 层无注册表** | 引入 `@register_parser(level=N)` 优先级注册，merger 按 level 排序合并 | P2 |

---

### 六、ML 正确性风险（ML 工程师 — 破坏"100% 正确"原则）

| # | 风险 | 影响范围 | 建议 | 优先级 |
|---|------|---------|------|--------|
| 19 | **嵌套 config 扁平化导致字段冲突**（text/vision hidden_size 互覆盖） | 所有多模态模型 | 保持子模型 namespace 隔离，normalize 返回树状结构 | P0 |
| 20 | **MLA KV cache 用错 GQA 公式** | DeepSeek-V2/V3 | kv_cache.py 按检测结果分支：MHA / GQA / MLA 三套公式 | P0 |
| 21 | **Pre-norm vs Post-norm 未检测** | 所有模型数据流 | config_normalizer 增加 norm_position 字段；L2 可从模块树 100% 确定 | P0 |
| 22 | **DiT/Flux/SD3 被当 UNet 处理** | 新一代扩散模型 | diffusers_parser 按 `_class_name` 路由，不硬编码 `unet` key | P0 |
| 23 | **DBRX 合并 expert tensor 推断失败** | DBRX 类模型 | safetensors_parser 双策略：regex 索引 + 3D tensor shape[0] | P1 |
| 24 | **SSM/Mamba 完全无法处理** | Mamba/Jamba/Zamba | 新增 SSMDetector + SSM flow generator | P1 |
| 25 | **Falcon `multi_query: true` 无法映射** | Falcon 系列 | normalizer 支持 bool→int 语义转换（`multi_query=True` → `num_kv_heads=1`） | P1 |
| 26 | **tie_word_embeddings 默认值随 model_type 变化** | 参数量统计 | 建立 per-model-type 默认值表 | P1 |
| 27 | **GGUF 模型完全不支持** | 量化发行者模型 | 新增 `gguf_parser.py` 或 L1 适配层 | P2 |
| 28 | **MoE flow 需参数化 block template** | MoE 可视化 | 接收 MoEDetectorResult 动态生成 router→expert→combine 流 | P1 |
| 29 | **MLA/GQA 检测器可能同时触发**（互斥关系） | DeepSeek-V2 | 加入 post-processing 互斥逻辑 | P1 |
| 30 | **tensor name→语义角色映射不可靠** | safetensors 推断 | 参数量标 EXACT，层结构推断标 INFERRED/HEURISTIC | P1 |
| 31 | **缺少 LoRA/Adapter 检测器** | LoRA 模型参数统计 | adapter_config.json 检测 + 参数量拆分 | P2 |
| 32 | **PP Zero-Bubble W 阶段建模复杂** | PP 调度可视化 | 独立的 ScheduleStep 类型区分 F/B/W/IDLE | P2 |

---

### 七、生产稳定性（基础设施工程师）

| # | 问题 | 建议 | 优先级 |
|---|------|------|--------|
| 33 | **多 worker Semaphore 失效**（4 workers × 2 = 8 并发） | v1.0 单 worker + 水平扩容 | P0 |
| 34 | **psutil 容器内读宿主机内存** | 直接读 cgroup v1/v2 文件 | P0 |
| 35 | **OOM 后 Executor 变 broken 不恢复** | ResilientProcessPool 检测 broken 并重建 | P0 |
| 36 | **asyncio.wait_for 超时不杀子进程** | 改用 multiprocessing.Process + 显式 kill | P0 |
| 37 | **SSE 客户端断连后 Phase B 仍运行** | request.is_disconnected() + task.cancel() | P0 |
| 38 | **HF Hub 无重试/限流处理** | tenacity 指数退避 + 429 降级返回 stale 缓存 | P0 |
| 39 | **无健康检查端点** | 三级 /health/live + /health/ready + /health/startup | P0 |
| 40 | **无优雅关闭** | lifespan + SIGTERM handler + graceful timeout | P0 |
| 41 | **缓存损坏无容错** | try-except + 静默删除 + CRC32 校验（v1.1） | P0 |
| 42 | **TOCTOU 竞态** | 内存检查纳入 Semaphore 临界区内 | P1 |
| 43 | **spawn 冷启动 5-8s** | startup 预热 + initializer 预加载 | P1 |
| 44 | **缓存雪崩** | TTL jitter ±10% | P1 |
| 45 | **SSE 无 heartbeat** | 15s 间隔 SSE 注释行 | P1 |
| 46 | **日志无结构化** | structlog JSON + request_id 中间件 | P1 |

---

### 修订后的目录结构（融合全部共识）

```
backend/app/
├── main.py
├── application/                     # Application 层（原散落的几个文件）
│   ├── pipeline.py                  # 纯管线编排（不含 SSE/缓存策略）
│   ├── sse_emitter.py               # SSE 协议：revision 拆分、heartbeat
│   ├── cache_strategy.py            # 业务缓存策略：key 生成、分段、失效判定
│   ├── comparison.py                # 多模型对比编排
│   ├── admission.py                 # 准入控制器（信号量 + cgroup 内存检查）
│   └── sandbox.py                   # 进程隔离（接收 Callable，不 import services）
│
├── routers/
│   ├── model.py                     # GET + SSE
│   ├── compare.py                   # POST /compare → comparison.py
│   ├── explore.py                   # WebSocket (v1.x)
│   └── health.py                    # /health/live, /health/ready, /health/startup
│
├── models/                          # 领域模型（零依赖，dataclass 为主）
│   ├── graph.py                     # ModuleGraph (纯数据容器 + 基本查询)
│   ├── profile.py                   # ArchitectureProfile + FeatureTag 枚举
│   ├── provenance.py                # Provenance, Confidence
│   ├── contracts.py                 # StageOutcome[T] + PipelineTrace + 各 Stage I/O
│   ├── hardware.py                  # HardwareProfile 领域模型（NEW）
│   └── schemas.py                   # API 响应 Pydantic 模型 (from_stage_results)
│
├── services/
│   ├── _registry.py                 # 泛型 Registry[T]
│   ├── parsing/
│   │   ├── orchestrator.py          # Stage 1 内部编排：build_layer_plan + execute
│   │   ├── config_parser.py         # L0 (含 normalizer + 嵌套 config 树状处理)
│   │   ├── safetensors_parser.py    # L1 (后缀模式匹配 + 统计推断)
│   │   ├── meta_loader.py           # L2 (纯 worker 函数, 不依赖 admission)
│   │   ├── diffusers_parser.py      # diffusers (按 _class_name 路由, 不硬编码 unet)
│   │   ├── gguf_parser.py           # GGUF 元数据解析 (v1.1)
│   │   ├── merger.py                # 多层合并 + Provenance
│   │   └── ast_parser.py            # L3 (v1.1)
│   │
│   ├── detectors/
│   │   ├── base.py + 14 个检测器    # +SSM +LoRA(v1.1) + post-processing 互斥逻辑
│   │   └── post_process.py          # 检测结果互斥/组合逻辑（MLA↔GQA 等）
│   │
│   ├── flow_generators/
│   │   ├── base.py + 生成器         # +ssm.py
│   │   └── blocks/                  # 参数化 block template (moe_block, ssm_block)
│   │
│   ├── estimators/
│   │   └── kv_cache.py              # MHA/GQA/MLA 三套公式分支
│   │
│   └── layout/
│       └── engine.py
│
├── parallel/                        # 独立领域计算模块（与 models/ 同级）
│   ├── tensor_parallel.py
│   ├── pipeline_parallel.py
│   ├── data_parallel.py
│   ├── sequence_parallel.py
│   ├── context_parallel.py
│   ├── expert_parallel.py
│   ├── scheduler.py                 # F/B/W/IDLE ScheduleStep
│   └── strategy_composer.py
│
└── infra/
    ├── cache.py                     # 纯机制：TTLCache+Lock + 原子文件写 + 损坏容错
    ├── hf_client.py                 # httpx + tenacity 重试 + 429 降级 + 大小限制
    ├── hw_specs.py                  # GPU 规格查表 → 返回 HardwareProfile
    └── metrics.py                   # Prometheus 指标 (v1.1)
```

---

以上 46 条建议，建议全部采纳（P2/P3 按标注版本分期）。是否同意？

---

### 5.3 流程设计

### 5.3.1 核心 SSE 双阶段推送流程

```
┌──────────┐     ┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Client   │────▶│  SSE Endpoint │────▶│  Pipeline        │────▶│  SSE Emitter │
│  (GET)    │◀────│  /models/{id} │◀────│  (Orchestrator)  │◀────│  (Queue)     │
└──────────┘     └───────────────┘     └──────────────────┘     └──────────────┘
                        │                       │
                   Connection           ┌───────┴────────┐
                   Semaphore(50)        │                │
                                   Phase A          Phase B
                                   (sync+fast)      (async+heavy)
```

**完整流程（含全部修正）：**

```
Client ──GET /api/v1/models/{repo_id}──▶ Router

Router:
  1. SSE 连接限制检查
     ├─ sse_semaphore.acquire(timeout=0)  # 全局 Semaphore(50)
     └─ 失败 → 返回 429 "Too many SSE connections"

  2. 全局超时包裹  # [P0-9]
     async with asyncio.timeout(45):  # 45s 全局兜底

  3. 缓存查询（segmented）
     ├─ cache_key = f"{repo_id}:{commit_sha[:8]}:{segment}"  # [P1-21]
     ├─ L0 hit → 直接 yield, skip pipeline
     └─ L0 miss → 检查 single-flight  # [P1-20]
         ├─ in-flight → await event, 复用结果
         └─ not in-flight → register flight, 进入 Pipeline

  4. Pipeline 执行
     ├─ Phase A ──────────────────────────────────────────────
     │   a. async fetch (FetchedModelData 模式)  # [P0-8]
     │      fetched = FetchedModelData(
     │        config = await hf_client.fetch_config(repo_id),
     │        safetensors_meta = await hf_client.fetch_safetensors_meta(repo_id),
     │        model_info = await hf_client.fetch_model_info(repo_id),
     │      )
     │      ├─ 404/private → yield error event MODEL_NOT_FOUND, return  # [P1-15]
     │      └─ config=None → StageOutcome(data=None, degraded=["L0"])  # [P0-14]
     │
     │   b. Stage 1: parse_structure(fetched) → StageOutcome[ModuleGraph]
     │      ├─ L0: config_parser.parse(fetched.config)
     │      │   └─ normalize: two-phase strategy  # [P0-11]
     │      │       ① model_type routing (DBRX→flatten ffn_config)
     │      │       ② generic recursive prefix flattening fallback
     │      │   └─ encoder-decoder: preserve both branches  # [P0-5]
     │      ├─ L1: safetensors_parser.parse(fetched.safetensors_meta)
     │      └─ merge: field-level source priority  # [P0-3]
     │          ├─ vocab_size → L0 wins (padding)
     │          ├─ num_attention_heads → L0 wins
     │          ├─ actual_param_count → L1 wins
     │          └─ hidden_size → cross-validate (must match)
     │
     │   c. Stage 2-5: adaptive offload  # [P0-6]
     │      if graph.node_count > 500:
     │          result = await loop.run_in_executor(thread_pool, run_stages_2_to_5, graph)
     │      else:
     │          result = run_stages_2_to_5(graph)
     │
     │      Stage 2: detect_features(graph) → StageOutcome[Set[FeatureTag]]
     │          ├─ 串行执行 14 detectors（总计 <15ms）  # [P2-40 确认]
     │          ├─ 每个 detector 独立 try/except  # [P1-25]
     │          │   └─ 单个异常 → diagnostics 记录, 其余正常
     │          └─ post_process: mutual exclusion rules
     │
     │      Stage 3: synthesize_flows(graph, features)
     │      Stage 4: estimate_resources(graph, features)
     │      Stage 5: compute_layout(graph)
     │
     │   d. yield revision=1 via Queue  # [P0-7]
     │      queue.put(SSEEvent(
     │          data=full_result,
     │          revision=1,
     │          is_final=not should_try_phase_b,
     │          confidence_summary=compute_confidence(result)  # [P1-19]
     │      ))
     │
     │   e. cache backfill (非阻塞)  # [P1-29]
     │      asyncio.create_task(backfill_cache(result))
     │
     ├─ Phase B ──────────────────────────────────────────────
     │   f. 断路器检查  # [P0-13]
     │      if circuit_breaker.is_open():
     │          → skip Phase B, revision=1 is_final=True
     │
     │   g. 大模型预判  # [P1-17]
     │      estimated_mem = estimate_meta_device_memory(profile)
     │      if estimated_mem > available_memory * 0.8:
     │          → skip Phase B, diagnostics += "model_too_large"
     │
     │   h. admission.acquire()  # Semaphore(2)
     │      └─ memory check: read cgroup v1/v2 files  # [P0-12]
     │
     │   i. meta-device loading (sandbox)
     │      enhanced = await sandbox.run(
     │          meta_device_enhance, repo_id, fetched.config,
     │          timeout=30
     │      )
     │      ├─ ResilientProcessPool: Process.kill(SIGKILL) on timeout  # [P0-10]
     │      ├─ trust_remote_code=False
     │      │   └─ 降级时 diagnostics += "remote_code_skipped"  # [P1-31]
     │      └─ 失败 → circuit_breaker.record_failure()  # [P0-13]
     │
     │   j. Phase B re-run (幂等)  # [P0-1]
     │      graph_v2 = clone_structure_only(graph_v1)  # 保留节点+层级, 清空 flow edges
     │      graph_v2 = merge_enhanced(graph_v2, enhanced)  # field-level priority
     │      result_v2 = run_stages_2_to_5(graph_v2)  # 从零重建 flows
     │
     │   k. yield revision=2 via Queue
     │      queue.put(SSEEvent(data=full_result_v2, revision=2, is_final=True))
     │
     │   l. admission.release()
     │
     └─ SSE Consumer (单一消费者)  # [P0-7]
         async for event in queue:
             yield format_sse(event)
         # heartbeat producer 也写入同一 Queue
         # → 消除帧交错风险

  5. finally 清理  # [P2-38]
     ├─ cancel Phase B task (if running)
     ├─ admission.release() (if held)
     ├─ sse_semaphore.release()
     └─ single-flight: notify waiters + deregister
```

**代码映射表：**

| 流程步骤 | 文件 | 函数/类 |
|----------|------|---------|
| SSE 连接限制 | `routers/model.py` | `sse_connection_guard()` |
| 全局超时 | `routers/model.py` | `asyncio.timeout(45)` middleware |
| Single-flight | `application/cache_strategy.py` | `SingleFlight.execute()` |
| FetchedModelData 组装 | `application/pipeline.py` | `fetch_model_data()` |
| Pipeline 编排 | `application/pipeline.py` | `run_pipeline()` |
| SSE Queue merger | `application/sse_emitter.py` | `SSEMerger` |
| 断路器 | `application/sandbox.py` | `CircuitBreaker` |
| Admission | `application/admission.py` | `AdmissionController` |
| 缓存策略 | `application/cache_strategy.py` | `SegmentedCacheStrategy` |
| 字段级合并 | `services/parsing/merger.py` | `merge_with_priority()` |
| 两阶段 normalize | `services/parsing/config_parser.py` | `TwoPhaseNormalizer` |
| clone_structure_only | `models/graph.py` | `ModuleGraph.clone_structure_only()` |

---

### 5.3.2 Stage 内部执行流程

```
StageOutcome[T] 统一契约:
┌─────────────────────────────────────┐
│  data: T                            │  # 本 Stage 产出
│  provenance: Dict[str, Provenance]  │  # 字段级来源
│  degraded_layers: List[str]         │  # 降级的层
│  diagnostics: List[Diagnostic]      │  # 问题记录
│  duration_ms: float                 │  # 耗时
└─────────────────────────────────────┘

Stage 2 (detect_features) 详细流程:
  for detector in registry.get_all():      # 14 detectors, 串行
      try:
          tags = detector.detect(graph, profile)
          all_tags.update(tags)
          trace.record(detector.name, duration, tags)
      except Exception as e:               # [P1-25] detector 级容错
          diagnostics.append(Diagnostic(
              stage="detect_features",
              component=detector.name,
              error=str(e),
              severity="warning"
          ))
  # post_process: mutual exclusion
  all_tags = post_process_tags(all_tags)    # e.g. MHA ∩ GQA → keep GQA
  return StageOutcome(data=all_tags, diagnostics=diagnostics)
```

**PipelineTrace 轻量化设计**  `[P1-28]`：

```python
@dataclass
class PipelineTrace:
    stages: List[StageTrace]

@dataclass
class StageTrace:
    name: str               # e.g. "detect_features"
    duration_ms: float
    node_count: int         # 当前 graph 节点数
    diagnostics: List[Diagnostic]
    # 不存储 result data → 避免内存膨胀
```

---

### 5.3.3 多模型对比流程

```
Client ──POST /api/v1/compare──▶ Router
  body: { repo_ids: ["A", "B", "C", "D"], mode: "quick" | "deep" }

Router:
  1. 校验: len(repo_ids) ≤ 4
  2. mode 路由:  # [P1-26]
     ├─ "quick" → 全部走 Phase A only (最多 4 模型)
     └─ "deep"  → 最多 2 模型允许 Phase B
         if len(repo_ids) > 2 and mode == "deep":
             return 400 "deep compare supports max 2 models"

  3. 并发执行:
     results = await asyncio.gather(*[
         run_pipeline(repo_id, phase_b=is_deep)
         for repo_id in repo_ids
     ], return_exceptions=True)

  4. 对比分析:
     comparison = comparison.compare(results)
     # 结构差异 / 参数对比 / feature 差集

  5. 返回: { models: [...], diff: {...} }
```

---

### 5.3.4 Diffusers 模型解析流程

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

### 5.3.5 MoE 数据流生成流程

```
触发条件: FeatureTag.MOE in features

flow_generators/moe_flow.py:
  1. 从 graph 提取 MoE 参数
     num_experts = profile.num_local_experts
     top_k = profile.num_experts_per_tok
     router_type = profile.router_type  # "top_k" | "expert_choice"

  2. 使用参数化模板生成 flow  # blocks/moe_block.py
     moe_flow = MoEBlockTemplate(
         num_experts=num_experts,
         top_k=top_k,
         router_type=router_type,
         has_shared_expert=profile.has_shared_expert,  # DeepSeek-V2
     ).generate()

  3. 输出 DataEdge 列表:
     edges = [
         DataEdge(src="input", dst="router", tensor_shape=[B, S, D]),
         DataEdge(src="router", dst="gate", tensor_shape=[B, S, num_experts]),
         # top_k selection
         DataEdge(src="gate", dst=f"expert_{i}", tensor_shape=[B, S/top_k, D])
         for i in range(top_k)
         # + shared expert if applicable
         DataEdge(src="shared_expert", dst="combine", ...),
         DataEdge(src="combine", dst="output", tensor_shape=[B, S, D]),
     ]
```

---

### 5.3.6 并行策略计算流程

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

### 5.3.7 PP 调度器流程

```
scheduler.py:

ScheduleStep = NamedTuple('ScheduleStep', [
    ('stage_id', int),
    ('micro_batch_id', int),
    ('action', Literal['F', 'B', 'W', 'IDLE']),
    ('gpu_id', int),
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

### 5.3.8 encoder-decoder 特殊处理流程

```
config_normalizer 流程:  # [P0-5]

  if config.get("is_encoder_decoder", False):
      # 保留双分支
      normalized = {
          "encoder": {
              "num_hidden_layers": config.get("encoder_layers", config.get("num_hidden_layers")),
              "hidden_size": config.get("encoder_ffn_dim", config.get("d_model")),
              ...
          },
          "decoder": {
              "num_hidden_layers": config.get("decoder_layers"),
              "hidden_size": config.get("decoder_ffn_dim", config.get("d_model")),
              ...
          },
          "is_encoder_decoder": True,
      }
      # BartConfig(encoder_layers=6, decoder_layers=12)
      # → encoder.num_hidden_layers=6, decoder.num_hidden_layers=12
      # 不再丢失 decoder 层数
```

---

### 5.3.9 HF Hub 通信流程

```
hf_client.py (httpx + tenacity):

  1. 正常请求
     @retry(
         stop=stop_after_attempt(3),
         wait=wait_exponential(min=1, max=10),
         retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
         before_sleep=log_retry
     )
     async def fetch_config(repo_id, subfolder=None):
         resp = await self.client.get(
             f"{HF_API}/{repo_id}/resolve/main/{subfolder or ''}/config.json",
             timeout=10
         )

  2. 429 降级
     if resp.status_code == 429:
         retry_after = int(resp.headers.get("Retry-After", 60))
         self._rate_limited_until = now() + retry_after
         raise RateLimited(retry_after)

  3. Hub 不可达 3 级降级  # [P1-24]
     except (httpx.ConnectError, httpx.TimeoutException):
         level_1: return L1_file_cache.get(key)       # 文件缓存
         level_2: return stale_cache.get(key, stale=True)  # 上次成功 + stale 标记
         level_3: raise HubUnreachableError()          # 无缓存可用

  4. Model 404 处理  # [P1-15]
     if resp.status_code == 404:
         raise ModelNotFoundError(repo_id)
     if resp.status_code == 401:  # private/gated
         raise ModelAccessDeniedError(repo_id)
```

---

### 5.3.10 缓存生命周期流程

```
6-segment 缓存设计:

Segment          │ TTL    │ 大小估算 │ 说明
─────────────────┼────────┼──────────┼──────────
raw_config       │ 24h    │ ~2KB     │ 原始 config.json
structure        │ 12h    │ ~50KB    │ ModuleGraph
features         │ 12h    │ ~1KB     │ Set[FeatureTag]
flows            │ 6h     │ ~20KB    │ DataEdge list
resources        │ 6h     │ ~5KB     │ ResourceEstimate
layout           │ 6h     │ ~30KB    │ 3D positions

Cache Key 格式:  # [P1-21]
  f"{repo_id}:{commit_sha[:8]}:{segment}"
  # commit_sha 从 model_info() 获取，模型更新后自动失效

L0 (内存): TTLCache + threading.Lock (per segment)
L1 (文件): atomic write (.tmp → rename)  # [P2-35]

写入流程 (非阻塞):  # [P1-29]
  async def backfill_cache(key, data, segment):
      # L0: 直接写入 (in event loop, <1ms)
      l0_cache[segment].set(key, data)
      # L1: offload to thread
      await run_in_executor(thread_pool, l1_cache.write, key, data)

GC 流程:
  定时任务 (每 10 分钟):
      await run_in_executor(thread_pool, gc_expired_files)  # [P2-34]
      # 使用 os.scandir (非 os.listdir) 避免大目录性能问题

  Emergency GC (内存 > 80%):
      # 按 LRU 删除 L0 entries
      # L1: 保留 .tmp 写入中的文件  # [P2-35]

手动刷新:  # [P1-16]
  DELETE /api/v1/models/{repo_id}/cache
  → 清除该模型全部 6 segment (L0 + L1)
```

---

### 5.3.11 ResilientProcessPool 生命周期

```
ResilientProcessPool:

  初始化 (startup):
      pool_size = min(cpu_count(), 2)  # 默认 2 workers
      for i in range(pool_size):
          worker = Process(target=sandbox_worker, args=(task_queue, result_queue))
          worker.start()
      # warmup: 每个 worker 执行 import torch, transformers
      warmup_result = await ping_all_workers(timeout=30)  # [P1-23]
      if warmup_result.has_permanent_failure:
          self.phase_b_available = False
          logger.error("Phase B disabled: torch/transformers not available")
      elif warmup_result.has_transient_failure:
          retry(3)  # transient → retry

  执行任务:
      async def run(fn, *args, timeout=30):
          worker = self._get_idle_worker()
          worker.task_queue.put((fn, args))
          try:
              result = await asyncio.wait_for(
                  self._wait_result(worker), timeout=timeout
              )
          except asyncio.TimeoutError:
              worker.process.kill()  # SIGKILL  # [P0-10]
              self._rebuild_worker(worker)
              circuit_breaker.record_failure()
              raise

  异常检测与重建:
      if worker.process.exitcode is not None:  # worker 已死
          if worker.process.exitcode == -9:  # OOM killed
              logger.warning("Worker OOM killed, rebuilding")
          self._rebuild_worker(worker)

  [P2-41] 健康检查:
      # 每 30s ping workers, broken → 提前 rebuild

  [TODO v1.1] 监控指标:  # [P2-36]
      pool_worker_restarts_total
      pool_task_duration_seconds
      pool_timeout_total
```

---

### 5.3.12 Admission Control 流程

```
AdmissionController:

  semaphore = asyncio.Semaphore(2)  # 最多 2 个并发 meta-device 加载
  memory_watermark = 0.85           # 85% 内存水位线

  async def acquire(estimated_mem_mb: int = 0):
      await semaphore.acquire()
      # 在临界区内检查内存  # [P0-12]
      try:
          available = read_cgroup_memory()  # 直接读 cgroup files
          # cgroup v2: /sys/fs/cgroup/memory.max, memory.current
          # cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes, memory.usage_in_bytes
          # host fallback: psutil.virtual_memory()
          if available.used_ratio > memory_watermark:
              semaphore.release()
              raise MemoryPressureError(f"Memory at {available.used_ratio:.0%}")
          if estimated_mem_mb > 0 and estimated_mem_mb > available.free_mb:  # [P1-17]
              semaphore.release()
              raise ModelTooLargeError(f"Need {estimated_mem_mb}MB, only {available.free_mb}MB free")
      except:
          semaphore.release()
          raise

  release():
      semaphore.release()
```

---

### 5.3.13 SSE 重连流程 `[P1-18]`

```
Client ──GET /api/v1/models/{id} [Last-Event-ID: "rev1-abc123"]──▶ Router

Router:
  1. 解析 Last-Event-ID
     last_id = request.headers.get("Last-Event-ID")
     if last_id:
         parsed = parse_event_id(last_id)  # → {revision: 1, request_hash: "abc123"}

  2. 断点续传判断
     cached_result = cache.get(repo_id, segment="full_result")
     if cached_result and cached_result.revision > parsed.revision:
         # 已有更新结果 → 直接发送
         yield sse_event(data=cached_result, id=make_event_id(cached_result))
     elif cached_result and cached_result.revision == parsed.revision:
         # Phase B 可能还在进行中
         if phase_b_in_progress(repo_id):
             → 等待 Phase B 完成后发送 revision=2
         else:
             yield sse_event(data=cached_result, is_final=True)
     else:
         # 无缓存 → 重新执行完整 pipeline
         → 进入 5.3.1 正常流程
```

---

### 5.3.14 冷启动预热流程 `[P1-32]`

```
startup_event (FastAPI lifespan):

  1. Process Pool 初始化 (5.3.11)
  2. L1 → L0 缓存预热
     top_models = scan_l1_cache_by_access_time(limit=100)
     for model_key in top_models:
         data = l1_cache.read(model_key)
         if data and not data.is_expired:
             l0_cache.set(model_key, data)
     logger.info(f"Warmed up {len(top_models)} models from L1 cache")

  3. K8s probe 状态  # [P1-23]
     /health/ready → 200 only after:
       ├─ pool warmup complete (or marked unavailable)
       └─ cache warmup complete
     /health/live → 200 always (process alive)
     /health/startup → 200 after basic init (before warmup)
```

---

### 5.3.15 日志与可观测性 `[P2-37]`

```
[TODO v1.1] 结构化日志设计:

  request_id: 全请求生命周期透传
    ├─ Router → Pipeline → Services: contextvars.ContextVar
    ├─ → ThreadPoolExecutor: copy_context()
    └─ → subprocess: sandbox 函数参数传递

  日志格式: JSON structured
  {
      "timestamp": "...",
      "level": "INFO",
      "request_id": "req-abc123",
      "stage": "detect_features",
      "duration_ms": 12.5,
      "model": "meta-llama/Llama-3-70B",
      "message": "..."
  }

[TODO v1.1] 监控指标:  # [P2-36]
  sse_connections_active          (Gauge)
  phase_b_duration_seconds        (Histogram)
  cache_hit_ratio_by_segment      (Counter per segment)
  pool_worker_restarts_total      (Counter)
  admission_wait_seconds          (Histogram)
  hub_request_duration_seconds    (Histogram, label=endpoint)
  circuit_breaker_state           (Gauge, 0=closed, 1=open)
```

---

### 流程交叉引用矩阵

| 流程 | 依赖的流程 | 涉及的 P0 修正 |
|------|-----------|---------------|
| 5.3.1 核心 SSE | 5.3.2, 5.3.10, 5.3.11, 5.3.12 | P0-1,6,7,8,9,10,12,13,14 |
| 5.3.2 Stage 内部 | — | P0-6, P1-25 |
| 5.3.3 多模型对比 | 5.3.1 (Phase A) | P1-26 |
| 5.3.4 Diffusers | 5.3.1, 5.3.12 | P0-2,4,5 |
| 5.3.5 MoE 数据流 | 5.3.2 (Stage 3) | P0-11 |
| 5.3.8 encoder-decoder | 5.3.2 (Stage 1) | P0-5 |
| 5.3.9 HF Hub | — | P0-14, P1-15,24 |
| 5.3.10 缓存 | — | P1-16,20,21,29 |
| 5.3.11 ProcessPool | 5.3.12 | P0-10, P0-13, P1-23 |
| 5.3.12 Admission | — | P0-2,12, P1-17 |
| 5.3.13 SSE 重连 | 5.3.1, 5.3.10 | P1-18 |
| 5.3.14 冷启动 | 5.3.10, 5.3.11 | P1-23,32 |

---

Section 5.3 流程设计修订完成，共 **15 条流程**（原 12 条 + 新增 5.3.13 SSE 重连 + 5.3.14 冷启动预热 + 5.3.15 日志可观测性），全部 42 项评审意见已内联。请确认后进入 Section 5.4 或 Section 6。

---

## 六、存储设计

> 本项目无传统数据库，全部持久化通过两级缓存实现（L0 内存 + L1 文件）。

### 6.1 存储架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│                                                                  │
│  cache_strategy.py          single_flight.py                     │
│  ┌─ SegmentCacheStrategy    ┌─ SingleFlightController            │
│  ├─ RevisionFenceManager    └─ FlightResult + timeout + errors   │
│  ├─ InvalidationStrategy                                         │
│  └─ GCPolicy (水位线/TTL)                                        │
│                        │                                         │
│              ┌─────────┴──────────┐                              │
│              ▼                    ▼                               │
│  ┌──────────────────┐  ┌────────────────────┐                    │
│  │  Infrastructure  │  │  Infrastructure     │                    │
│  │  L0MemoryStore   │  │  L1FileStore        │                    │
│  │  (pure KV:       │  │  (pure file I/O:    │                    │
│  │   get/put/del)   │  │   read/write/del)   │                    │
│  │  不知 segment/   │  │  不知 revision/     │                    │
│  │  revision 概念   │  │  segment 概念       │                    │
│  └──────────────────┘  └────────────────────┘                    │
│      ~ms 级读取              ~10ms 级读取                          │
│      容量: ~200MB            容量: ~5GB (可配置)                    │
└─────────────────────────────────────────────────────────────────┘
```

**层职责边界** `[#17]`：

| 层 | 职责 | 不知道的概念 |
|---|------|------------|
| **infra/cache.py** | 纯 KV 存取：`get(key) → bytes`, `put(key, data, ttl)`, `delete(key)` | segment, revision, provenance |
| **application/cache_strategy.py** | 组合 L0+L1, read-through/write-through, RevisionFence, GCPolicy, SEGMENTS 定义 | HTTP, SSE, pipeline |
| **application/single_flight.py** | 并发请求去重协调 | cache, pipeline |

---

### 6.2 Cache Key 设计

```
格式: v{schema}:{repo_id_normalized}:{commit_sha_prefix}:{component}:{segment}

规则:                                                    # [#11] [#28] [#5]
  schema_version   = CACHE_SCHEMA_VERSION (int)          # 路径隔离新旧版本
  repo_id_normalized = validate_and_normalize(repo_id)   # 安全校验 + 归一化
  commit_sha_prefix  = commit_sha[:12]                   # 48 bits, 碰撞阈值 ~16M
  component          = "__root__"                         # Transformers 单模型
                     | "unet" | "vae" | "text_encoder"   # Diffusers 组件级
  segment            ∈ SEGMENT_NAMES                     # 6 个固定值

示例:
  Transformers: "v1:meta-llama--llama-3-70b:a1b2c3d4e5f6:__root__:structure"
  Diffusers:    "v1:stabilityai--sdxl:e5f6g7h8i9j0:unet:structure"
              "v1:stabilityai--sdxl:e5f6g7h8i9j0:__pipeline_index__:raw_config"
```

**输入校验** `[#8]`：

```python
import re
from pathlib import Path

def validate_and_normalize(repo_id: str) -> str:
    # Layer 1: 正则白名单
    if not re.match(r'^[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)?$', repo_id):
        raise ValueError(f"Invalid repo_id: {repo_id}")
    normalized = repo_id.replace("/", "--").lower()
    # Layer 2: 黑名单
    if ".." in normalized or normalized.startswith("/"):
        raise ValueError(f"Suspicious repo_id: {repo_id}")
    return normalized

def validate_sha(sha: str) -> str:
    if not re.match(r'^[0-9a-f]{7,40}$', sha):
        raise ValueError(f"Invalid commit SHA: {sha}")
    return sha

def safe_cache_path(cache_dir: Path, *parts: str) -> Path:
    path = cache_dir.joinpath(*parts)
    # Layer 3: resolve 后确认在 cache_dir 内
    resolved = path.resolve()
    if not resolved.is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Path escape: {path} -> {resolved}")
    return resolved
```

---

### 6.3 六段缓存数据模型（修订后上限）

```python
CACHE_SCHEMA_VERSION = 1

SEGMENTS: Dict[str, SegmentConfig] = {
    "raw_config": SegmentConfig(
        ttl=86400,                # 24h
        max_entry_bytes=100_000,  # 100KB [#19] — Diffusers 多组件 config
        serializer="json",
    ),
    "structure": SegmentConfig(
        ttl=43200,                # 12h
        max_entry_bytes=5_000_000, # 5MB [#4] — MoE/Diffusers 大模型
        serializer="msgpack",
        compression="zstd",       # [#32] 压缩比 3-5x
    ),
    "features": SegmentConfig(
        ttl=43200,                # 12h
        max_entry_bytes=10_000,
        serializer="json",
    ),
    "flows": SegmentConfig(
        ttl=21600,                # 6h
        max_entry_bytes=1_000_000, # 1MB [#4] — MoE fan-out/fan-in 数据流
        serializer="msgpack",
        compression="zstd",
    ),
    "resources": SegmentConfig(
        ttl=21600,                # 6h
        max_entry_bytes=200_000,  # 200KB [#4] — 量化元数据
        serializer="json",
    ),
    "layout": SegmentConfig(
        ttl=21600,                # 6h
        max_entry_bytes=2_000_000, # 2MB — 大模型 3D 坐标
        serializer="msgpack",
        compression="zstd",
    ),
}
# [#33] layout 只存逻辑拓扑坐标（相对位置），前端负责映射到 3D 空间
```

**CacheEntry（修正后）** `[#1]`：

```python
@dataclass(frozen=True)
class CacheEntry:
    key: str
    data: bytes               # 序列化后的 payload（可能含 zstd 压缩）
    created_at: float         # time.time() — wall clock [#1]
    expire_at: float          # created_at + ttl — wall clock [#1]
    ttl_seconds: int
    size_bytes: int
    compression: str          # "none" | "zstd" [#32]
    full_commit_sha: str      # 完整 40 字符 sha [#11]

    @property
    def is_expired(self) -> bool:
        now = time.time()
        if self.created_at > now + 60:  # 时钟回拨防御
            return True
        return now > self.expire_at

    @property
    def remaining_ttl(self) -> float:
        return max(0.0, self.expire_at - time.time())
```

> **注意**：`revision`、`provenance_summary` 等业务字段不在 CacheEntry 中 `[#17]`。它们被序列化进 `data` 内部，由 application 层解析。

---

### 6.4 Revision Fence — 跨 Segment 一致性 `[#3]`

```python
@dataclass
class RevisionFence:
    repo_key: str               # "v1:meta-llama--llama-3-70b:a1b2c3d4e5f6:__root__"
    committed_revision: int     # 最后一次完整写入的 revision (1 or 2)
    committed_at: float         # wall clock
    segment_revisions: Dict[str, int]  # 每个 segment 当前写入的 revision

class RevisionFenceManager:
    """保证读取端只看到完整 revision 的数据"""

    async def commit_revision(self, repo_key: str, revision: int,
                              segments: Dict[str, CacheEntry]):
        """原子提交：全部 segment 写入后才更新 fence"""
        # 1. 写入所有 segment (L0 + L1)
        for seg_name, entry in segments.items():
            self.l0.put(f"{repo_key}:{seg_name}", entry)
            await self._backfill_l1(f"{repo_key}:{seg_name}", entry)

        # 2. 更新 fence（原子写入）
        fence = RevisionFence(
            repo_key=repo_key,
            committed_revision=revision,
            committed_at=time.time(),
            segment_revisions={s: revision for s in segments},
        )
        self.l0.put(f"{repo_key}:__fence__", serialize(fence))
        await self._backfill_l1(f"{repo_key}:__fence__", serialize(fence))

    def read_consistent(self, repo_key: str) -> Tuple[int, Dict[str, bytes]]:
        """读取时只返回 committed_revision 对应的一致快照"""
        fence = self._read_fence(repo_key)
        if not fence:
            return 0, {}
        segments = {}
        for seg_name in SEGMENTS:
            entry = self.l0.get(f"{repo_key}:{seg_name}")
            if entry:
                entry_rev = self._extract_revision(entry)
                if entry_rev <= fence.committed_revision:
                    segments[seg_name] = entry
                # revision > committed 的是未完成写入，忽略
        return fence.committed_revision, segments
```

---

### 6.5 L0 内存缓存（infra 层，纯 KV）

```python
class L0MemoryStore:
    """纯 KV 内存存储，不知道 segment/revision 概念"""  # [#17]

    def __init__(self, max_entries: int = 10000, default_ttl: int = 43200):
        self._store = TTLCache(maxsize=max_entries, ttl=default_ttl)
        self._lock = threading.Lock()
        self._gc_cooldown_until: float = 0.0  # [#16]

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:  # 微秒级
            return self._store.get(key)

    def put(self, key: str, data: bytes, ttl: int = None) -> None:
        with self._lock:  # 微秒级
            self._store[key] = data

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def delete_prefix(self, prefix: str) -> int:
        """删除指定前缀的全部 key"""
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def memory_usage_bytes(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._store.values())

    def emergency_gc(self, target_bytes: int) -> int:
        """分批快照→排序→逐 key 删除，持锁时间极短""" # [#6]
        # Phase 1: 快照
        with self._lock:
            snapshot = list(self._store.items())  # [#12] list() 强制求值
        # Phase 2: 排序（无锁）
        snapshot.sort(key=lambda kv: kv[1].created_at if hasattr(kv[1], 'created_at') else 0)
        # Phase 3: 逐 key 删除
        evicted = 0
        for key, _ in snapshot:
            if self.memory_usage_bytes() <= target_bytes:
                break
            with self._lock:
                self._store.pop(key, None)
                evicted += 1
        # [#16] 设置冷却期
        self._gc_cooldown_until = time.time() + 30
        return evicted

    @property
    def in_gc_cooldown(self) -> bool:
        return time.time() < self._gc_cooldown_until
```

---

### 6.6 L1 文件缓存（infra 层，纯文件 I/O）

**文件系统布局** `[#23] [#28] [#10]`：

```
{CACHE_DIR}/                             # 默认 /data/cache (非 /tmp)
├── v1/                                  # schema version 隔离
│   ├── me/                              # repo_id[:2] hash 分桶
│   │   └── meta-llama--llama-3-70b/
│   │       ├── a1b2c3d4e5f6/            # commit_sha[:12]
│   │       │   ├── __root__/            # component
│   │       │   │   ├── raw_config.json
│   │       │   │   ├── structure.msgpack.zst
│   │       │   │   ├── features.json
│   │       │   │   ├── flows.msgpack.zst
│   │       │   │   ├── resources.json
│   │       │   │   └── layout.msgpack.zst
│   │       │   └── __fence__.json       # RevisionFence
│   │       └── prev_sha/               # 旧版本（标记 .stale 后 GC 优先回收）
│   ├── st/
│   │   └── stabilityai--sdxl/
│   │       └── e5f6g7h8i9j0/
│   │           ├── __pipeline_index__/  # pipeline 元数据
│   │           ├── unet/               # 组件级缓存 [#5]
│   │           ├── vae/
│   │           └── text_encoder/
│   └── _meta/
│       └── access_index.json           # AccessTracker [#15]
└── v0/                                  # 旧 schema，GC 自动清理
```

**原子写入** `[#7]`：

```python
import os, tempfile, contextlib

class L1FileStore:
    """纯文件 I/O，不知道 segment/revision 概念"""

    def __init__(self, cache_dir: Path, max_size_bytes: int):
        self._cache_dir = cache_dir
        self._max_size = max_size_bytes
        self._current_size = AtomicCounter(0)  # 启动时扫描初始化 [#9]

    def read(self, path: Path) -> Optional[bytes]:
        try:
            data = path.read_bytes()
            return data
        except FileNotFoundError:         # [#25] GC 竞争 → cache miss
            return None
        except OSError as e:
            logger.warning(f"L1 read error: {e}")
            return None

    def write(self, path: Path, data: bytes) -> bool:
        """多进程安全原子写入"""
        # [#9] 写入前 quota 检查
        if self._current_size.get() + len(data) > self._max_size:
            return False  # 超限拒绝

        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)  # [#10]
        fd, tmp_path_str = tempfile.mkstemp(      # [#7] 唯一临时文件名
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())              # 确保落盘
            fd = -1
            os.replace(tmp_path_str, str(path))   # 跨平台原子替换
            self._current_size.add(len(data))
            return True
        except BaseException:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path_str)
            return False                           # [#26] 写入失败 non-fatal

    def delete(self, path: Path) -> bool:
        try:
            size = path.stat().st_size
            path.unlink()
            self._current_size.sub(size)
            return True
        except OSError:
            return False

    def gc_expired(self, is_expired_fn) -> int:
        """必须在 executor 线程中调用""" # [#24]
        removed = 0
        empty_dirs = []
        schema_dir = self._cache_dir / f"v{CACHE_SCHEMA_VERSION}"
        if not schema_dir.exists():
            return 0
        for bucket_dir in schema_dir.iterdir():
            if bucket_dir.name == "_meta" or not bucket_dir.is_dir():
                continue
            for repo_dir in bucket_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                for sha_dir in repo_dir.iterdir():
                    if not sha_dir.is_dir():
                        continue
                    # [#11] .stale 标记的目录优先回收
                    if (sha_dir / ".stale").exists():
                        shutil.rmtree(sha_dir, ignore_errors=True)
                        removed += 1
                        continue
                    for comp_dir in sha_dir.iterdir():
                        if not comp_dir.is_dir():
                            continue
                        with os.scandir(comp_dir) as entries:  # [#24] with 保证 FD 释放
                            has_live = False
                            for entry in entries:
                                if entry.name.endswith(".tmp"):
                                    if time.time() - entry.stat().st_mtime > 300:
                                        with contextlib.suppress(OSError):
                                            os.unlink(entry.path)
                                    continue
                                if is_expired_fn(entry.path):
                                    with contextlib.suppress(OSError):
                                        os.unlink(entry.path)
                                        removed += 1
                                else:
                                    has_live = True
                            if not has_live:
                                empty_dirs.append(comp_dir)
        # 清理空目录
        for d in empty_dirs:
            with contextlib.suppress(OSError):
                d.rmdir()
        return removed

    def gc_old_schema(self, current_version: int):  # [#28]
        """清理旧 schema 版本目录"""
        for entry in self._cache_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("v"):
                try:
                    v = int(entry.name[1:])
                    if v < current_version:
                        shutil.rmtree(entry, ignore_errors=True)
                except ValueError:
                    pass
```

---

### 6.7 Diffusers 组件级缓存 `[#5]`

```python
# application/cache_strategy.py 中的 Diffusers 缓存逻辑

class ComponentCacheStrategy:
    """Diffusers pipeline 的组件级缓存"""

    def cache_key(self, repo_id: str, sha: str,
                  component: str, segment: str) -> str:
        norm = validate_and_normalize(repo_id)
        sha_v = validate_sha(sha)
        return f"v{CACHE_SCHEMA_VERSION}:{norm}:{sha_v[:12]}:{component}:{segment}"

    async def write_pipeline(self, repo_id, sha, pipeline_index, component_results):
        """分组件写入"""
        # 1. 写入 pipeline 索引
        idx_key = self.cache_key(repo_id, sha, "__pipeline_index__", "raw_config")
        await self._write(idx_key, pipeline_index)

        # 2. 逐组件写入全部 segment
        for comp_name, comp_data in component_results.items():
            for seg_name, seg_data in comp_data.items():
                key = self.cache_key(repo_id, sha, comp_name, seg_name)
                await self._write(key, seg_data)

        # 3. 更新 fence（pipeline 级）
        await self.fence_mgr.commit_revision(...)

    async def read_component(self, repo_id, sha, component, segment):
        """按需读取单个组件的单个 segment — SSE 可渐进推送"""
        key = self.cache_key(repo_id, sha, component, segment)
        return await self._read(key)
```

---

### 6.8 SingleFlight（修正后无死锁）`[#2]`

```python
@dataclass
class FlightResult:
    value: Any = None
    error: BaseException | None = None

class SingleFlightController:
    """并发请求去重。粒度: per (repo, sha, phase)""" # [#2]

    def __init__(self, default_timeout: float = 120.0):
        self._lock = asyncio.Lock()
        self._flights: Dict[str, asyncio.Event] = {}
        self._results: Dict[str, FlightResult] = {}
        self._timeout = default_timeout

    async def execute(self, key: str, fn: Callable) -> Any:
        # 判断 leader/follower（锁内只做字典查找，不 await）
        async with self._lock:
            if key in self._flights:
                event = self._flights[key]
                is_leader = False
            else:
                event = asyncio.Event()
                self._flights[key] = event
                is_leader = True

        if not is_leader:
            # follower: 在锁外等待 [#2 关键修正]
            try:
                await asyncio.wait_for(event.wait(), timeout=self._timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"SingleFlight wait timeout: {key}")
            result = self._results.get(key)
            if result is None:
                raise RuntimeError(f"Flight result cleaned: {key}")
            if result.error is not None:
                raise result.error  # 传播原始异常
            return result.value

        # leader: 执行实际工作
        flight_result = FlightResult()
        try:
            flight_result.value = await asyncio.wait_for(fn(), timeout=self._timeout)
            return flight_result.value
        except Exception as e:
            flight_result.error = e
            raise
        finally:
            self._results[key] = flight_result
            async with self._lock:
                self._flights.pop(key, None)
            event.set()  # 通知 followers
            # 延迟清理 [#14]
            asyncio.get_running_loop().call_later(
                5.0, lambda k=key: self._results.pop(k, None)
            )
```

---

### 6.9 序列化层 `[#13]`

```python
import msgpack, zlib, json, enum

# msgpack ext type codes
_EXT_FROZENSET = 1
_EXT_TUPLE = 2
_EXT_ENUM = 3

def _encode_hook(obj):
    if isinstance(obj, frozenset):
        return msgpack.ExtType(_EXT_FROZENSET,
            msgpack.packb(sorted(obj), default=_encode_hook))
    if isinstance(obj, tuple):
        return msgpack.ExtType(_EXT_TUPLE,
            msgpack.packb(list(obj), default=_encode_hook))
    if isinstance(obj, enum.Enum):
        return msgpack.ExtType(_EXT_ENUM,
            msgpack.packb({"cls": type(obj).__name__, "val": obj.value}))
    raise TypeError(f"Cannot serialize {type(obj)}")

def _decode_hook(code, data):
    if code == _EXT_FROZENSET:
        return frozenset(msgpack.unpackb(data, ext_hook=_decode_hook))
    if code == _EXT_TUPLE:
        return tuple(msgpack.unpackb(data, ext_hook=_decode_hook))
    if code == _EXT_ENUM:
        return msgpack.unpackb(data, ext_hook=_decode_hook)["val"]
    return msgpack.ExtType(code, data)

def serialize(segment_config: SegmentConfig, payload: Any) -> bytes:
    if segment_config.serializer == "json":
        data = json.dumps(payload, ensure_ascii=False, separators=(",",":")).encode()
    else:
        data = msgpack.packb(payload, default=_encode_hook, use_bin_type=True)
    if segment_config.compression == "zstd" and len(data) > 10240:  # [#32] >10KB 才压缩
        import zstandard
        data = zstandard.ZstdCompressor(level=3).compress(data)
    return data

def deserialize(segment_config: SegmentConfig, data: bytes) -> Any:
    if segment_config.compression == "zstd" and data[:4] == b'\x28\xb5\x2f\xfd':
        import zstandard
        data = zstandard.ZstdDecompressor().decompress(data)
    if segment_config.serializer == "json":
        return json.loads(data)
    return msgpack.unpackb(data, ext_hook=_decode_hook, raw=False)

# 数据完整性校验（文件头）[Infra 安全建议]
HEADER_FMT = "<BI"  # version(1B) + crc32(4B) = 5 bytes

def wrap_with_header(data: bytes) -> bytes:
    crc = zlib.crc32(data) & 0xFFFFFFFF
    header = struct.pack(HEADER_FMT, CACHE_SCHEMA_VERSION, crc)
    return header + data

def unwrap_and_verify(raw: bytes) -> Optional[bytes]:
    if len(raw) < 5:
        return None
    version, expected_crc = struct.unpack(HEADER_FMT, raw[:5])
    if version != CACHE_SCHEMA_VERSION:
        return None  # schema 不匹配 → cache miss [#28]
    data = raw[5:]
    if zlib.crc32(data) & 0xFFFFFFFF != expected_crc:
        return None  # 数据损坏 → cache miss
    return data
```

---

### 6.10 级联失效 + Volatile Repo 检测 `[#11] [#18]`

```python
class InvalidationStrategy:
    def __init__(self):
        self._sha_index: Dict[str, List[Tuple[float, str]]] = {}  # repo → [(created_at, sha)]

    async def on_new_commit(self, repo_id: str, old_sha: str, new_sha: str):
        norm = validate_and_normalize(repo_id)
        # 1. L0: 立即驱逐旧 SHA 全部 key
        self.l0.delete_prefix(f"v{CACHE_SCHEMA_VERSION}:{norm}:{old_sha[:12]}")
        # 2. L1: 标记旧 SHA 目录为 stale → GC 优先回收
        stale_dir = self._sha_dir(norm, old_sha[:12])
        if stale_dir.exists():
            (stale_dir / ".stale").touch()

    def detect_volatile(self, repo_id: str) -> bool:
        """TTL 窗口内 3+ 不同 sha → volatile"""
        norm = validate_and_normalize(repo_id)
        entries = self._sha_index.get(norm, [])
        cutoff = time.time() - 43200  # 12h window
        recent = [e for e in entries if e[0] > cutoff]
        return len(set(sha for _, sha in recent)) >= 3

    def effective_ttl(self, repo_id: str, base_ttl: int,
                      access_count: int = 0) -> int:
        """自适应 TTL""" # [#20]
        if self.detect_volatile(repo_id):
            return min(base_ttl, 3600)       # volatile → 1h 上限
        if access_count > 100:
            return min(base_ttl * 2, 172800) # 热门 → x2 (最大 48h)
        return base_ttl
```

---

### 6.11 AccessTracker（替代 access_log.jsonl）`[#15]`

```python
class AccessTracker:
    """内存记录 + 定期 compact 为 access_index.json"""

    INDEX_FILE = "_meta/access_index.json"

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir
        self._dirty: Dict[str, Tuple[float, int]] = {}  # key → (last_access, count)
        self._lock = asyncio.Lock()

    async def record(self, key: str):
        async with self._lock:
            prev = self._dirty.get(key, (0, 0))
            self._dirty[key] = (time.time(), prev[1] + 1)

    async def flush(self):
        """定期合并到 index 文件（原子写入）"""
        async with self._lock:
            if not self._dirty:
                return
            to_flush = dict(self._dirty)
            self._dirty.clear()
        existing = self._read_index()
        existing.update(to_flush)
        # 过滤已删除的条目
        existing = {k: v for k, v in existing.items()
                    if self._cache_exists(k)}
        self._atomic_write_index(existing)

    def top_n(self, n: int) -> List[str]:
        """冷启动预热 Top-N"""
        index = self._read_index()
        return sorted(index, key=lambda k: index[k][0], reverse=True)[:n]
```

---

### 6.12 容量规划（修订）

```
单模型缓存占用（压缩后估算）:
  dense 7B-13B:   ~50KB  (structure ~30KB compressed)
  MoE 8x22B:      ~400KB (structure ~200KB compressed, flows ~150KB)
  Diffusers SDXL:  ~300KB (5 组件，各自 ~60KB)
  DeepSeek-V2:     ~800KB (极端 case，structure ~500KB compressed)

容量估算:
  L0 上限:  10000 entries × 平均 50KB ≈ 500MB (需乘 worker 数)
  L1 上限:  5GB (可配置，emptyDir sizeLimit 设 6.5GB)

内存水位线:
  Normal:    < 60%  → 正常运行
  Warning:   60-80% → 日志 warning
  Critical:  > 80%  → emergency_gc + 停止 L1→L0 回填 [#16]
  Danger:    > 90%  → admission 拒绝新请求

多 Worker 注意 [Infra]:
  - 单 worker + async 并发（推荐）：L0 容量 = 上述值
  - 多 worker：每个 worker 独立 L0，总内存 = L0 × worker_count
```

---

### 6.13 K8s 部署配置 `[#10] [#34] [#37]`

```yaml
# Pod spec
volumes:
  - name: l1-cache
    emptyDir:
      sizeLimit: 6500Mi  # L1 5GB + 30% headroom

containers:
  - name: api
    volumeMounts:
      - name: l1-cache
        mountPath: /data/cache
    resources:
      requests:
        ephemeral-storage: "7Gi"
      limits:
        ephemeral-storage: "7Gi"
    env:
      - name: CACHE_DIR
        value: "/data/cache"
    securityContext:
      runAsNonRoot: true
      readOnlyRootFilesystem: true  # /data/cache 通过 volumeMount 可写

# Readiness probe 与预热结合 [#37]
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
```

---

### 6.14 可观测性指标 `[#27]`

```python
# Prometheus metrics 预定义
cache_ops     = Counter("cache_ops_total", "Cache operations",
                        ["level", "op", "segment"])  # level=l0|l1, op=hit|miss|write|delete
l0_memory     = Gauge("l0_memory_bytes", "L0 memory usage")
l1_disk       = Gauge("l1_disk_bytes", "L1 disk usage")
l1_file_count = Gauge("l1_file_count", "L1 file count")
gc_duration   = Histogram("gc_duration_seconds", "GC scan duration")
gc_deleted    = Counter("gc_deleted_total", "Items deleted by GC", ["level"])
l1_rejected   = Counter("l1_write_rejected_total", "L1 writes rejected by quota")
flight_waits  = Counter("singleflight_waits_total", "SingleFlight follower waits")
```

---

### 6.15 CacheConfig（修订）`[#35]`

```python
@dataclass
class CacheConfig:
    cache_dir: Path = Path("/data/cache")           # [#10] 非 /tmp
    schema_version: int = CACHE_SCHEMA_VERSION
    l0_max_entries: int = 10000
    l1_max_size_bytes: int = 5 * 1024**3            # 5GB（含压缩后的大模型）
    gc_interval_seconds: int = 600
    emergency_gc_threshold: float = 0.80
    gc_cooldown_seconds: int = 30                    # [#16]
    singleflight_timeout: float = 120.0
    volatile_sha_threshold: int = 3                  # [#18]

    # per-segment TTL (环境变量可覆盖)
    ttl_raw_config: int = 86400
    ttl_structure: int = 43200
    ttl_features: int = 43200
    ttl_flows: int = 21600
    ttl_resources: int = 21600
    ttl_layout: int = 21600

    @classmethod
    def from_env(cls) -> "CacheConfig":
        cache_dir = Path(os.environ.get("CACHE_DIR", "/data/cache"))
        resolved = cache_dir.resolve()
        # [#35] 禁止危险路径
        for prefix in ["/proc", "/sys", "/dev", "/etc", "/var/run"]:
            if str(resolved).startswith(prefix):
                raise ValueError(f"cache_dir must not be under {prefix}")
        return cls(cache_dir=resolved, ...)
```

---


- **P0**：wall clock 替代 monotonic、SingleFlight 死锁修复、RevisionFence 一致性、segment 上限 5x-10x 提升、组件级缓存、mkstemp 原子写入、路径注入防御、写入时 quota 检查、/data/cache 替代 /tmp、sha[:12]
- **P1**：TTLCache 快照迭代、msgpack ext_type hook、AccessTracker 替代 jsonl、GC cooldown、infra/application 职责分离、volatile repo 检测、自适应 TTL、schema 版本路径隔离、Prometheus 指标
- **P2**：zstd 压缩、hash 分桶、layout 逻辑坐标约束、CRC32 完整性校验（均已标注为 TODO 或直接内联）

---

## 七、接口设计

> 基于 Section 5 流程设计和 Section 6 存储设计，定义全部 REST/SSE 接口的请求响应规格。

### 7.1 接口总览

| # | 方法 | 路径 | 说明 | 认证 |
|---|------|------|------|------|
| 1 | `GET` | `/api/v1/models/{repo_id:path}/stream` | **核心接口** — SSE 双阶段推送模型结构 | 无 |
| 2 | `POST` | `/api/v1/compare` | 多模型对比（最多 4 模型） | 无 |
| 3 | `GET` | `/api/v1/models/{repo_id:path}/parallel` | 并行策略计算 | 无 |
| 4 | `GET` | `/api/v1/models` | 模型搜索/热门推荐 | 无 |
| 5 | `DELETE` | `/api/v1/models/{repo_id:path}/cache` | 手动刷新缓存 | API Key |
| 6 | `GET` | `/health/live` | K8s liveness probe | 无 |
| 7 | `GET` | `/health/ready` | K8s readiness probe | 无 |
| 8 | `GET` | `/health/startup` | K8s startup probe | 无 |

> **变更 [C1/M1/m1]**：SSE 端点从 `/models/{id}` 分离为 `/models/{id}/stream`，预留 `/models/{id}` 给未来同步 JSON；`/explore` 改为 `GET /api/v1/models`（集合资源）。

---

### 7.2 公共依赖

#### 7.2.1 repo_id 校验（M14）

```python
import re
from fastapi import Path, Depends, HTTPException

REPO_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")

def validated_repo_id(
    repo_id: str = Path(..., max_length=200),
) -> str:
    if not REPO_ID_RE.fullmatch(repo_id):
        raise HTTPException(400, detail=_err("INVALID_REPO_ID", "Invalid repo_id format"))
    if ".." in repo_id or "\x00" in repo_id:
        raise HTTPException(400, detail=_err("INVALID_REPO_ID", "Illegal characters"))
    return repo_id
```

#### 7.2.2 通用枚举 & 类型别名（M15/M16/m4）

```python
from enum import Enum
from typing import Literal

# 置信度（全局复用）
ConfidenceLevel = Literal["EXACT", "INFERRED", "ESTIMATED"]

class ModuleType(str, Enum):
    MODEL      = "model"
    LAYER      = "layer"
    ATTENTION  = "attention"
    FFN        = "ffn"
    EXPERT     = "expert"
    NORM       = "norm"
    EMBEDDING  = "embedding"
    LINEAR     = "linear"
    PROJECTION = "projection"
    LM_HEAD    = "lm_head"
    CONTAINER  = "container"
    OTHER      = "other"

class DetailLevel(str, Enum):
    FULL     = "full"
    OVERVIEW = "overview"

class EdgeType(str, Enum):
    DATA_FLOW       = "data_flow"
    RESIDUAL        = "residual"
    SKIP_CONNECTION  = "skip_connection"
    EXPERT_ROUTE    = "expert_route"

class GeometryHint(str, Enum):
    BOX      = "box"
    SPHERE   = "sphere"
    CYLINDER = "cylinder"
    CUSTOM   = "custom"

class DType(str, Enum):
    FLOAT16  = "float16"
    FLOAT32  = "float32"
    BFLOAT16 = "bfloat16"
    INT8     = "int8"
    INT4     = "int4"
```

#### 7.2.3 request_id 中间件（s3 相关）

```python
from contextvars import ContextVar
from uuid import uuid4

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="unknown")

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = rid
    token = request_id_ctx.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    request_id_ctx.reset(token)
    return response
```

#### 7.2.4 SSE Semaphore 封装（C5）

```python
import asyncio

class BoundedSSESemaphore:
    def __init__(self, max_val: int):
        self._sem = asyncio.Semaphore(max_val)
        self._max = max_val

    async def try_acquire(self) -> bool:
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=0)
            return True
        except asyncio.TimeoutError:
            return False

    def release(self):
        self._sem.release()

    @property
    def current(self) -> int:
        return self._max - self._sem._value
```

---

### 7.3 核心接口 — SSE 模型结构推送

**`GET /api/v1/models/{repo_id:path}/stream`**

> **变更 [M1]**：路径从 `/models/{id}` 分离为 `/models/{id}/stream`。

```
请求:
  GET /api/v1/models/meta-llama/Llama-3.1-70B/stream?revision=main&phase_b=true
  Accept: text/event-stream
  Last-Event-ID: 1:015              (可选，断线重连)

路径参数:
  repo_id: str          # Depends(validated_repo_id)

查询参数:
  revision: str = "main"
  phase_b: bool = true
  components: list[str] = null       # 重复参数 ?components=graph&components=flows (M15)
  detail_level: DetailLevel = "full" # Enum 约束 (M15)
```

#### 7.3.1 SSE Event 类型（C2 拆分 + M3/M5 语义增强）

| event 类型 | 含义 | 触发时机 | 单帧大小上限 |
|-----------|------|---------|------------|
| `progress` | 进度推送 | 每个 Stage 开始/结束 | ~1KB |
| `result:meta` | profile + confidence + features + resources + trace | Phase A/B 完成 | ~10KB |
| `result:graph` | nodes + edges **分 chunk** | Phase A/B 完成 | **≤200KB/chunk** |
| `result:flows` | flows 数据 | Phase A/B 完成 | ~50KB |
| `result:layout` | 3D 布局数据 | Phase A/B 完成 | ~100KB |
| `result:complete` | 标记当前 revision 所有 chunk 发送完毕 | 全部 result 子事件后 | ~200B |
| `heartbeat` | 保活 | 每 **10s**（无其他事件时） | ~100B |
| `error` | 错误信息（含 fatal 标记） | 404/403/500/timeout | ~500B |
| `done` | 流结束（含完成状态） | 全部 phase 完成/失败 | ~300B |

> **变更 [C2]**：result 拆为 5 个子事件，graph 分 chunk（≤200KB/chunk）。  
> **变更 [M6]**：heartbeat 间隔从 15s 缩短为 **10s**。

#### 7.3.2 Event ID 编码（C1 统一序列号）

```
全局单调递增序列号: {revision}:{global_seq}

示例:
  id: 1:001    event: progress
  id: 1:002    event: progress
  id: 1:003    event: result:meta
  id: 1:004    event: result:graph    (chunk 1/3)
  id: 1:005    event: result:graph    (chunk 2/3)
  id: 1:006    event: result:graph    (chunk 3/3)
  id: 1:007    event: result:flows
  id: 1:008    event: result:layout
  id: 1:009    event: result:complete
  id: 2:010    event: progress
  ...
  id: 2:018    event: done

heartbeat 复用上一个有意义事件的 ID，不推进序列号。
```

#### 7.3.3 断线重连协议（C1）

```
客户端重连:
  Last-Event-ID: 1:006

服务端逻辑:
  1. 解析 revision=1, seq=6
  2. 检查 ring buffer（容量 200 条）:
     a. 命中 → 从 seq=7 开始重放（跳过 progress，直送 result/done）
     b. 未命中（断线太久）→ 发送 event: reset，客户端需全量重新请求
  3. 若 revision 1 的 result 已全部缓存 → 直接从缓存重发
  4. SSE 流开头声明重连间隔: retry: 3000
```

#### 7.3.4 result:meta 事件示例

```json
{
  "revision": 1,
  "is_final": false,
  "update_mode": "full",
  "commit_sha": "a1b2c3d4e5f6g7h8i9j0k1l2",
  "pipeline_run_id": "run-uuid-001",
  "confidence_summary": {
    "overall": "INFERRED",
    "degraded_layers": [
      { "layer": "L2", "reason": "meta_device_unavailable", "fallback": "heuristic_estimate" }
    ],
    "remote_code_skipped": false
  },
  "profile": {
    "model_type": "llama",
    "architecture": "LlamaForCausalLM",
    "num_parameters": 70553706496,
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "is_encoder_decoder": false,
    "encoder_config": null,
    "decoder_config": null
  },
  "features": ["GQA", "RoPE", "RMSNorm", "SwiGLU"],
  "resources": {
    "memory": {
      "params_bf16": { "bytes": 140660178944, "display": "131.0 GB" },
      "params_fp32": { "bytes": 281320357888, "display": "262.0 GB" },
      "kv_cache_per_token": {
        "bf16": { "bytes": 1310720, "display": "1.25 MB" },
        "formula": "2 * n_layers * (n_kv_heads * d_head) * 2"
      },
      "estimated_inference_bf16": { "bytes": 150323855360, "display": "~140 GB", "is_estimated": true }
    },
    "quantization": null,
    "assumptions": ["activation memory assumes batch_size=1, seq_len=2048"]
  },
  "trace": {
    "stages": [
      { "name": "parse_structure", "duration_ms": 245, "node_count": 1247, "diagnostics": [] },
      { "name": "detect_features", "duration_ms": 12, "diagnostics": [] },
      { "name": "synthesize_flows", "duration_ms": 38, "diagnostics": [] },
      { "name": "estimate_resources", "duration_ms": 5, "diagnostics": [] },
      { "name": "compute_layout", "duration_ms": 67, "diagnostics": [] }
    ],
    "total_ms": 367,
    "phase": "A"
  }
}
```

> **变更 [M11]**：MemoryEstimate 所有值改为 `ByteSize{bytes, display}`。  
> **变更 [m6/前端#12]**：degraded_layers 改为对象数组含 reason + fallback。

#### 7.3.5 result:graph 事件示例（分 chunk）

```json
{
  "revision": 1,
  "chunk_index": 0,
  "total_chunks": 3,
  "nodes": [
    {
      "id": "model",
      "name": "LlamaForCausalLM",
      "module_type": "model",
      "parent_id": null,
      "children_ids": ["model.embed_tokens", "model.layers", "model.norm", "lm_head"],
      "incoming_edge_ids": [],
      "outgoing_edge_ids": ["e-001"],
      "template_id": null,
      "instance_index": null,
      "params": { "total": 70553706496 },
      "provenance": { "source": "L0+L1", "confidence": "EXACT" }
    },
    {
      "id": "model.layers.0.self_attn",
      "name": "LlamaSdpaAttention",
      "module_type": "attention",
      "parent_id": "model.layers.0",
      "children_ids": ["...q_proj", "...k_proj", "...v_proj", "...o_proj"],
      "incoming_edge_ids": ["e-010"],
      "outgoing_edge_ids": ["e-011"],
      "template_id": "tpl-attn-gqa",
      "instance_index": 0,
      "params": {
        "total": 109051904,
        "q_proj": { "shape": [8192, 8192], "dtype": "bfloat16" }
      },
      "provenance": { "source": "L1", "confidence": "EXACT" }
    }
  ],
  "edges": [
    {
      "id": "e-001",
      "source_id": "model.embed_tokens",
      "target_id": "model.layers.0",
      "source_port": "bottom",
      "target_port": "top",
      "tensor_shape": [null, null, 8192],
      "tensor_dim_names": ["batch", "seq_len", "hidden"],
      "dtype": "bfloat16",
      "edge_type": "data_flow",
      "label": "hidden_states",
      "estimated_bytes": 67108864,
      "flow_stage_ids": ["stage-embed"],
      "control_points": null
    }
  ],
  "block_templates": [
    {
      "template_id": "tpl-attn-gqa",
      "template": "GQA",
      "params": { "num_kv_heads": 8, "num_q_heads": 64 },
      "instance_count": 80,
      "representative_node_id": "model.layers.0.self_attn"
    },
    {
      "template_id": "tpl-ffn-swiglu",
      "template": "SwiGLU",
      "params": { "intermediate_size": 28672 },
      "instance_count": 80,
      "representative_node_id": "model.layers.0.mlp"
    }
  ]
}
```

> **变更 [M8]**：增加 `block_templates[]`，每 node 增加 `template_id` + `instance_index`。  
> **变更 [M9]**：DataEdge 增加 `label`、`estimated_bytes`、`source_port/target_port`、`flow_stage_ids[]`、`control_points`。  
> **变更 [m6]**：增加 `tensor_dim_names`。  
> **变更 [s1]**：每 node 增加 `incoming_edge_ids` / `outgoing_edge_ids`。

#### 7.3.6 result:flows 事件示例（M10）

```json
{
  "revision": 1,
  "macro": [
    {
      "stage_id": "stage-embed",
      "label": "Token Embedding",
      "description": "Convert token IDs to dense vectors",
      "node_ids": ["model.embed_tokens"],
      "edge_ids": ["e-001"],
      "input_shape": { "tokens": [null, null] },
      "output_shape": { "hidden_states": [null, null, 8192] },
      "order": 0,
      "is_repeating": false,
      "repeat_count": 1,
      "sub_stages": [],
      "duration_hint_ms": 300
    },
    {
      "stage_id": "stage-layer-group",
      "label": "Transformer Layers × 80",
      "node_ids": ["model.layers"],
      "edge_ids": ["e-002"],
      "order": 1,
      "is_repeating": true,
      "repeat_count": 80,
      "sub_stages": [
        {
          "stage_id": "stage-layer-N-attn",
          "label": "Self Attention (GQA)",
          "template_id": "tpl-attn-gqa",
          "node_ids": ["model.layers.{N}.self_attn"],
          "edge_ids": ["e-{N}-attn-in", "e-{N}-attn-out"],
          "order": 0,
          "duration_hint_ms": 400
        },
        {
          "stage_id": "stage-layer-N-ffn",
          "label": "FFN (SwiGLU)",
          "template_id": "tpl-ffn-swiglu",
          "node_ids": ["model.layers.{N}.mlp"],
          "edge_ids": ["e-{N}-ffn-in", "e-{N}-ffn-out"],
          "order": 1,
          "duration_hint_ms": 350
        }
      ],
      "duration_hint_ms": 800
    },
    {
      "stage_id": "stage-lm-head",
      "label": "LM Head",
      "node_ids": ["model.norm", "lm_head"],
      "edge_ids": ["e-norm-out", "e-lm-head"],
      "order": 2,
      "is_repeating": false,
      "repeat_count": 1,
      "sub_stages": [],
      "duration_hint_ms": 200
    }
  ],
  "blocks": {
    "attention": { "template": "GQA", "params": { "num_kv_heads": 8, "num_q_heads": 64 } },
    "ffn": { "template": "SwiGLU", "params": { "intermediate_size": 28672 } }
  }
}
```

#### 7.3.7 result:layout 事件示例（C4 完备化）

```json
{
  "revision": 1,
  "coordinate_system": "right_hand_y_up",
  "unit": "logical",
  "positions": {
    "model": {
      "x": 0, "y": 0, "z": 0,
      "level": 0,
      "scale": [3.0, 1.0, 3.0],
      "rotation": [0, 0, 0],
      "geometry": "box",
      "color": "#6B7280",
      "opacity": 1.0
    },
    "model.layers.0.self_attn": {
      "x": -1.5, "y": -3.0, "z": 0,
      "level": 2,
      "scale": [1.2, 0.8, 1.2],
      "rotation": [0, 0, 0],
      "geometry": "sphere",
      "color": "#4A90D9",
      "opacity": 1.0
    }
  },
  "bounds": { "min": [-5, -120, -3], "max": [5, 0, 3] },
  "suggested_camera": {
    "position": [0, -60, 50],
    "target": [0, -60, 0],
    "fov": 50,
    "near": 0.1,
    "far": 500,
    "up": [0, 1, 0],
    "controls": {
      "min_distance": 5,
      "max_distance": 200,
      "enable_damping": true
    }
  },
  "lod_levels": [
    { "level": 0, "label": "Model Overview", "visible_node_ids": ["model", "model.embed_tokens", "model.layers", "model.norm", "lm_head"], "camera_distance_threshold": 100 },
    { "level": 1, "label": "Layer Groups", "visible_node_ids": ["...layers.0", "...layers.1", "..."], "camera_distance_threshold": 50 },
    { "level": 2, "label": "Full Detail", "visible_node_ids": null, "camera_distance_threshold": 0 }
  ]
}
```

> **变更 [C4]**：Position3D 增加 `scale`、`rotation`、`geometry`、`color`、`opacity`；LayoutData 增加 `coordinate_system`、`unit`。  
> **变更 [M8]**：增加 `lod_levels[]`。  
> **变更 [m8]**：CameraConfig 增加 `fov`、`near/far`、`controls{min_distance, max_distance}`。

#### 7.3.8 result:complete 事件

```json
{
  "revision": 1,
  "is_final": false,
  "total_events_in_revision": 7,
  "pipeline_run_id": "run-uuid-001"
}
```

#### 7.3.9 rev2 增量更新策略（C3）

```
revision=2 策略:
  - result:meta → update_mode: "full" (meta 数据量小，始终全量)
  - result:graph → update_mode: "patch"
    payload 增加:
      "changed_node_ids": ["model.layers.0.self_attn", ...],
      "changed_edge_ids": ["e-010", ...],
      仅包含 changed nodes/edges（非全量）
  - result:flows → update_mode: "full" (通常不变或变化小)
  - result:layout → update_mode: "patch"
    payload 增加:
      "changed_position_ids": ["model.layers.0.self_attn", ...]
      仅包含位置变更的节点
  - result:complete → is_final: true

HTTP Header:
  X-Update-Mode: full | patch

短期方案: changed_*_ids + 仅含变更子集
中期演进: JSON Patch (RFC 6902) 格式
```

#### 7.3.10 progress 事件（s2 增强）

```json
{
  "stage": "parsing_structure",
  "stage_index": 1,
  "total_stages": 5,
  "progress": 0.3,
  "message": "Parsing L0+L1 (1,247 nodes)",
  "eta_ms": 2300,
  "nodes_discovered": 432,
  "edges_discovered": 856
}
```

#### 7.3.11 error 事件（M3 fatal 区分）

```json
{
  "code": "PHASE_B_TIMEOUT",
  "message": "Phase B meta-device loading timed out after 60s",
  "fatal": false,
  "phase": "B",
  "details": null,
  "request_id": "req-uuid-here"
}
```

> `fatal: false` → 客户端可继续使用已收到的 Phase A 数据。  
> `fatal: true` → 客户端应关闭连接。

#### 7.3.12 done 事件（M5 语义完善）

```json
{
  "status": "partial",
  "phases": {
    "a": "complete",
    "b": "timeout"
  },
  "total_events": 18,
  "elapsed_ms": 61234
}
```

> `status`: `"complete"` | `"partial"` | `"failed"`  
> `phases.{a,b}`: `"complete"` | `"skipped"` | `"timeout"` | `"error"`  
> **不变式**：任何 SSE 流必须以且仅以一个 `done` 事件结尾。

#### 7.3.13 SSE 帧格式与响应头（m7）

```
SSE 帧 (W3C EventSource 兼容):
  retry: 3000\n
  event: <event_type>\n
  id: <revision>:<global_seq>\n
  data: <json_payload>\n
  \n

必须响应头:
  Content-Type: text/event-stream
  Cache-Control: no-cache, no-store
  Connection: keep-alive
  X-Accel-Buffering: no          # 禁止 Nginx 缓冲
  X-Request-ID: req-uuid-here
```

#### 7.3.14 分阶段超时（M7）

```
Phase A: asyncio.timeout(15)   — L0+L1 快速解析
Phase B: asyncio.timeout(60)   — meta-device 加载

Phase A 超时 → error(fatal=true, code=ANALYSIS_TIMEOUT) + done(status=failed)
Phase B 超时 → error(fatal=false, code=PHASE_B_TIMEOUT) + done(status=partial, phases.b=timeout)
```

#### 7.3.15 SSE 端点实现（C5/C6/M6/M17 修复）

```python
@router.get("/api/v1/models/{repo_id:path}/stream")
async def stream_model(
    request: Request,
    repo_id: str = Depends(validated_repo_id),
    revision: str = Query("main", pattern=r"^[a-zA-Z0-9._/-]+$", max_length=100),
    phase_b: bool = True,
    components: list[str] | None = Query(None),
    detail_level: DetailLevel = DetailLevel.FULL,
    semaphore: BoundedSSESemaphore = Depends(get_sse_semaphore),
):
    # 1. SSE 连接限制
    if not await semaphore.try_acquire():
        raise HTTPException(429, detail=_err("SSE_CONNECTION_LIMIT", ...))

    # 2. 构造响应（兜底 release）
    try:
        response = EventSourceResponse(
            generate(request, repo_id, revision, phase_b, components, detail_level, semaphore),
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request.state.request_id,
            }
        )
    except Exception:
        semaphore.release()   # C5: 构造失败兜底释放
        raise

    return response

async def generate(request, repo_id, revision, phase_b, components, detail_level, semaphore):
    seq = 0
    queue: asyncio.Queue[SSEEvent] = asyncio.Queue(maxsize=64)   # M6: 有界队列
    hb_task = None
    pipeline_task = None
    try:
        # Phase A
        hb_task = asyncio.create_task(heartbeat_producer(queue, interval=10))
        pipeline_task = asyncio.create_task(
            run_pipeline(repo_id, revision, phase_b, components, detail_level, queue)
        )
        async for event in consume_queue(queue):
            # C6: 客户端断开检测（每个事件检查）
            if await request.is_disconnected():
                break
            seq += 1
            yield format_sse(event, seq)
            if event.type == "done":
                break
    except asyncio.TimeoutError:
        seq += 1
        yield format_sse(SSEEvent(type="error", data={"code": "ANALYSIS_TIMEOUT", "fatal": True}), seq)
        seq += 1
        yield format_sse(SSEEvent(type="done", data={"status": "failed"}), seq)
    finally:
        if hb_task:
            hb_task.cancel()
        if pipeline_task:
            pipeline_task.cancel()
        # M17: 等待 task 真正结束
        await asyncio.gather(
            *(t for t in [hb_task, pipeline_task] if t),
            return_exceptions=True
        )
        semaphore.release()   # C5: 保证释放
```

> **变更 [M6]**：Queue(maxsize=64)，满时 producer 对 progress 用 `put_nowait` + 丢弃旧条，对 result/error/done 阻塞等待。

#### 7.3.16 HTTP 状态码

| 状态码 | 场景 |
|--------|------|
| 200 | 正常 SSE 流 |
| 400 | repo_id 格式错误 / 参数校验失败 |
| 404 | 模型不存在（也通过 SSE error event 推送） |
| 429 | SSE 连接数超限 / HF Hub rate limit |
| 500 | 内部错误 |

> 注：连接建立后的错误（Phase B 超时等）通过 SSE error event 传递，不再用 HTTP 状态码。

---

### 7.4 多模型对比接口

**`POST /api/v1/compare`**

```python
from pydantic import BaseModel, Field, model_validator, ConfigDict

class CompareRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    repo_ids: list[str] = Field(..., min_length=2, max_length=4)
    mode: Literal["quick", "deep"] = "quick"
    revision: str = "main"

    # C7: 迁移到 model_validator
    @model_validator(mode="after")
    def validate_deep_limit(self) -> "CompareRequest":
        if self.mode == "deep" and len(self.repo_ids) > 2:
            raise ValueError("deep compare supports max 2 models")
        # 校验每个 repo_id 格式
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
    revision_used: int                    # 1 (quick) or 2 (deep)

class ComparisonDiff(BaseModel):
    parameter_comparison: ParameterComparison
    feature_diff: dict[str, list[bool]]    # feature → [has_a, has_b, ...]
    structure_diff: StructureDiff

# M13: 结构化参数对比
class ParameterComparison(BaseModel):
    model_level: dict[str, list[int | None]]    # field → [model_a, model_b, ...]
    per_layer: list[LayerComparison]

class LayerComparison(BaseModel):
    layer_pattern: str
    values: list[int | None]                     # 与 model 顺序一致
    ratio: float | None = None

# M13: 增加对齐映射
class StructureDiff(BaseModel):
    shared_patterns: list[str]
    unique_to: dict[str, list[str]]               # repo_id → unique features/modules
    alignment: list[AlignmentEntry]               # 节点级对齐映射
    layout_pair: LayoutPair | None = None

class AlignmentEntry(BaseModel):
    model_a_node: str | None
    model_b_node: str | None
    match_type: Literal["exact", "similar", "unique_a", "unique_b"]

class LayoutPair(BaseModel):
    model_a_offset: list[float]                   # 并排偏移 [x, y, z]
    model_b_offset: list[float]
```

```
请求:
  POST /api/v1/compare
  Content-Type: application/json

  {
    "repo_ids": ["meta-llama/Llama-3.1-70B", "Qwen/Qwen2.5-72B"],
    "mode": "quick"
  }

响应:
  200 OK
  Cache-Control: max-age=300
  Content-Location: /api/v1/compare?repos=meta-llama/Llama-3.1-70B,Qwen/Qwen2.5-72B&mode=quick

  {
    "models": [ ... ],
    "diff": {
      "parameter_comparison": {
        "model_level": {
          "num_parameters": [70553706496, 72709066752],
          "num_hidden_layers": [80, 80],
          "hidden_size": [8192, 8192]
        },
        "per_layer": [
          { "layer_pattern": "self_attn", "values": [109051904, 113246208], "ratio": 1.04 }
        ]
      },
      "feature_diff": {
        "GQA": [true, true],
        "SlidingWindow": [false, true]
      },
      "structure_diff": {
        "shared_patterns": ["GQA attention", "SwiGLU FFN", "RMSNorm"],
        "unique_to": { "Qwen/Qwen2.5-72B": ["SlidingWindowAttention"] },
        "alignment": [
          { "model_a_node": "model.layers.0.self_attn", "model_b_node": "model.layers.0.self_attn", "match_type": "exact" },
          { "model_a_node": null, "model_b_node": "model.layers.0.self_attn.sliding_window", "match_type": "unique_b" }
        ],
        "layout_pair": { "model_a_offset": [-6, 0, 0], "model_b_offset": [6, 0, 0] }
      }
    }
  }
```

> 400 错误 details 中包含限制参数：`{"max_repos_for_deep": 2, "max_repos_for_quick": 4}`

---

### 7.5 并行策略接口

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
    schedule: list[ScheduleStep] | None = None   # PP > 1 时返回
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
    supported_dimensions: list[str]     # 动态返回支持的策略维度

class PartitionedGraph(BaseModel):
    nodes: list[PartitionedNode]

class PartitionedNode(BaseModel):
    node_id: str
    tp_shard: Literal["col", "row"] | None = None
    pp_stage: int
    dp_rank: int
    ep_rank: int | None = None

# M12: 增加时间信息
class CommOp(BaseModel):
    op_type: Literal["AllReduce", "AllGather", "ReduceScatter", "P2P", "AllToAll"]
    source_ranks: list[int]
    dest_ranks: list[int]
    tensor_shape: list[int]
    estimated_bytes: int
    estimated_time_ms: float
    start_ms: float                     # 甘特图定位
    end_ms: float

class ScheduleStep(BaseModel):
    gpu_id: int
    time_slot: int                      # 逻辑序号（保留）
    start_ms: float                     # 相对起始时间
    end_ms: float                       # 相对结束时间
    action: Literal["F", "B", "W", "IDLE", "COMM"]
    stage_id: int
    micro_batch_id: int
    memory_usage_bytes: int | None = None
    comm_op_ids: list[str] | None = None   # 关联通信操作
```

---

### 7.6 模型搜索接口

**`GET /api/v1/models`**

> **变更 [m1]**：从 `/explore` 改为 `GET /api/v1/models`。

```
请求:
  GET /api/v1/models?q=llama+70b&sort=trending&limit=20

查询参数:
  q: str = ""
  sort: Literal["trending", "downloads", "recent"] = "trending"
  limit: int = Query(20, ge=1, le=50)
  offset: int = Query(0, ge=0, le=10000)
  cursor: str | None = None             # m2: cursor 分页
  filter_type: list[str] | None = Query(None)   # 重复参数支持多值
```

```python
class ModelsResponse(BaseModel):
    total: int
    total_is_estimate: bool = False     # m2: 标注是否估算
    next_cursor: str | None = None      # m2: cursor 分页
    has_more: bool
    models: list[ExploreItem]

class ExploreItem(BaseModel):
    repo_id: str
    model_type: str
    architecture: str                    # 前端#14: 增加
    num_parameters: int | None = None
    num_layers: int | None = None        # 前端#14: 增加
    features: list[str] | None = None    # 前端#14: 增加摘要
    tags: list[str]
    downloads: int
    trending_score: float | None = None
    cached: bool
```

---

### 7.7 缓存管理接口

**`DELETE /api/v1/models/{repo_id:path}/cache`**

```
请求:
  DELETE /api/v1/models/meta-llama/Llama-3.1-70B/cache
  Authorization: Bearer <api_key>

响应:
  200 OK
  {
    "deleted": true,
    "repo_id": "meta-llama/Llama-3.1-70B",
    "l0_entries": 12,
    "l1_files": 12,
    "total_bytes_freed": 524288
  }

  缓存不存在时: 204 No Content (幂等性)
  未认证: 401 Unauthorized
```

---

### 7.8 健康检查接口（m3）

```python
# GET /health/live — 进程存活 (始终 200)
@router.get("/health/live")
async def liveness():
    return {"status": "ok"}

# GET /health/ready — 就绪（200 | 503）
@router.get("/health/ready")
async def readiness():
    checks = {
        "cache_warmed": cache.is_warmed(),
        "pool_available": pool.is_available(),
        "phase_b_available": pool.phase_b_ready(),
    }
    all_ready = all(checks.values())
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "status": "ready" if all_ready else "not_ready",
            "checks": checks,
            "version": APP_VERSION,
            "commit": GIT_SHA,
        }
    )

# GET /health/startup — 基础初始化（200 | 503）
@router.get("/health/startup")
async def startup_check():
    initialized = app_state.initialized
    return JSONResponse(
        status_code=200 if initialized else 503,
        content={
            "status": "started" if initialized else "initializing",
            "version": APP_VERSION,
            "schema_version": SCHEMA_VERSION,
            "uptime_seconds": app_state.uptime_seconds,
        }
    )
```

---

### 7.9 通用响应规范

#### 7.9.1 错误响应格式

```python
class ErrorResponse(BaseModel):
    error: ErrorDetail

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, str] | None = None   # 限定 key-value，不暴露内部信息
    request_id: str = Field(default_factory=lambda: request_id_ctx.get())
```

> **变更**：`details` 类型从 `Dict[str, Any]` 收紧为 `dict[str, str]`（预定义 key：`field`, `constraint`, `suggestion`）。5xx 错误 details 仅含 request_id，完整堆栈写入服务端日志。

#### 7.9.2 错误码 → HTTP 状态码映射（M2）

| code | HTTP | 说明 | 可重试 | Retry-After |
|------|------|------|--------|-------------|
| `MODEL_NOT_FOUND` | 404 | HF Hub 返回 404 | No | — |
| `MODEL_ACCESS_DENIED` | 403 | 私有/gated 模型 | No | — |
| `INVALID_REPO_ID` | 400 | repo_id 格式校验失败 | No | — |
| `INVALID_PARAMETERS` | 422 | 查询参数校验失败 | No | — |
| `SSE_CONNECTION_LIMIT` | 429 | SSE 并发连接数超限 | Yes | 估算最早完成任务的剩余时间 |
| `RATE_LIMITED` | 429 | REST 速率限制 | Yes | slowapi 计算 |
| `HUB_RATE_LIMITED` | 429 | HF Hub 返回 429 | Yes | 上游 Retry-After 透传 |
| `HUB_UNREACHABLE` | 502 | HF Hub 不可达 | Yes | — |
| `ANALYSIS_TIMEOUT` | 504 | Phase A/B 超时 | Yes | — |
| `MEMORY_PRESSURE` | 503 | Admission 内存不足 | Yes | 30 |
| `INTERNAL_ERROR` | 500 | 未预期的内部错误 | No | — |

#### 7.9.3 通用响应头

```
X-Request-ID: req-uuid-here
X-Cache-Status: HIT | MISS | STALE | PARTIAL
X-Cache-Age: 120                       # 缓存年龄（秒）
X-Revision: 1 | 2
X-Phase-B-Status: completed | in_progress | skipped | unavailable | timeout
X-Update-Mode: full | patch            # rev2 时标记更新模式
```

> **变更**：`X-Cache-Status` 增加 `PARTIAL`（部分 segment 命中）；新增 `X-Cache-Age`、`X-Update-Mode`。

---

### 7.10 Pydantic Schema 定义（完整版）

```python
# ===== 基础类型 =====

class ByteSize(BaseModel):                  # M11
    bytes: int                              # 精确字节数，用于计算
    display: str                            # 人类可读，如 "131.0 GB"
    is_estimated: bool = False

class ProvenanceInfo(BaseModel):
    source: str                             # "L0" | "L1" | "L0+L1" | "L0+L1+L2"
    confidence: ConfidenceLevel             # m4: 复用全局类型别名

class DegradedLayer(BaseModel):             # 前端#12: 结构化降级信息
    layer: str
    reason: str
    fallback: str

# ===== Graph 数据 =====

class ModuleNode(BaseModel):
    id: str
    name: str
    module_type: ModuleType                 # M16: Enum 替代 str
    parent_id: str | None = None            # m5: 显式 = None
    children_ids: list[str] = Field(default_factory=list)
    incoming_edge_ids: list[str] = Field(default_factory=list)    # s1: 邻接索引
    outgoing_edge_ids: list[str] = Field(default_factory=list)
    template_id: str | None = None          # M8: 模板分组
    instance_index: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    provenance: ProvenanceInfo

class DataEdge(BaseModel):
    id: str
    source_id: str
    target_id: str
    source_port: str | None = None          # M9: 连接点位置
    target_port: str | None = None
    tensor_shape: list[int | None] = Field(..., min_length=1, max_length=8)
    tensor_dim_names: list[str] | None = None   # m6: 维度语义标注
    dtype: DType | None = None              # 枚举约束
    edge_type: EdgeType
    label: str | None = None                # M9: 语义标签
    estimated_bytes: int | None = None      # M9: 数据量（边粗细）
    flow_stage_ids: list[str] | None = None # M9: 关联 flow stage
    control_points: list[list[float]] | None = None   # M9: 贝塞尔控制点

class BlockTemplateInfo(BaseModel):         # M8: 模板分组
    template_id: str
    template: str
    params: dict[str, Any] = Field(default_factory=dict)
    instance_count: int
    representative_node_id: str

class ModuleGraph(BaseModel):
    nodes: list[ModuleNode]
    edges: list[DataEdge]
    block_templates: list[BlockTemplateInfo] = Field(default_factory=list)

# ===== Profile =====

class ArchitectureProfile(BaseModel):
    model_type: str
    architecture: str
    num_parameters: int = Field(..., gt=0)
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    intermediate_size: int | None = None     # m5
    num_attention_heads: int = Field(..., gt=0)
    num_key_value_heads: int | None = None
    vocab_size: int = Field(..., gt=0)
    max_position_embeddings: int | None = None
    is_encoder_decoder: bool = False
    encoder_config: dict[str, Any] | None = None
    decoder_config: dict[str, Any] | None = None

class ConfidenceSummary(BaseModel):
    overall: ConfidenceLevel                 # m4
    degraded_layers: list[DegradedLayer] = Field(default_factory=list)
    remote_code_skipped: bool = False

# ===== Flows =====

class FlowStage(BaseModel):                 # M10: 结构化 flow stage
    stage_id: str
    label: str
    description: str | None = None
    node_ids: list[str]
    edge_ids: list[str]
    input_shape: dict[str, list[int | None]] | None = None
    output_shape: dict[str, list[int | None]] | None = None
    order: int
    is_repeating: bool = False
    repeat_count: int = 1
    sub_stages: list["FlowStage"] = Field(default_factory=list)
    template_id: str | None = None
    duration_hint_ms: int | None = None

class BlockTemplate(BaseModel):
    template: str = Field(..., max_length=100)
    params: dict[str, Any] = Field(default_factory=dict)

class FlowData(BaseModel):
    macro: list[FlowStage]
    blocks: dict[str, BlockTemplate]

# ===== Resources =====

class MemoryEstimate(BaseModel):             # M11: ByteSize
    params_bf16: ByteSize
    params_fp32: ByteSize
    kv_cache_per_token: dict[str, ByteSize | str]
    estimated_inference_bf16: ByteSize
    assumptions: list[str] = Field(default_factory=list)

class QuantizationInfo(BaseModel):
    method: Literal["GPTQ", "AWQ", "GGUF", "bitsandbytes"]
    bits: int
    group_size: int | None = None
    estimated_size: ByteSize

class ResourceEstimate(BaseModel):
    memory: MemoryEstimate
    quantization: QuantizationInfo | None = None

# ===== Layout =====

class Position3D(BaseModel):                 # C4: 完备化
    x: float
    y: float
    z: float
    level: int
    scale: list[float] = Field(default=[1.0, 1.0, 1.0], min_length=3, max_length=3)
    rotation: list[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)
    geometry: GeometryHint = GeometryHint.BOX
    color: str = "#6B7280"
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)

class BoundingBox(BaseModel):
    min: list[float] = Field(..., min_length=3, max_length=3)
    max: list[float] = Field(..., min_length=3, max_length=3)

class CameraControls(BaseModel):             # m8
    min_distance: float = 5.0
    max_distance: float = 200.0
    enable_damping: bool = True

class CameraConfig(BaseModel):               # m8: 完备化
    position: list[float] = Field(..., min_length=3, max_length=3)
    target: list[float] = Field(..., min_length=3, max_length=3)
    fov: float = 50.0
    near: float = 0.1
    far: float = 500.0
    up: list[float] = Field(default=[0.0, 1.0, 0.0], min_length=3, max_length=3)
    controls: CameraControls = Field(default_factory=CameraControls)

class LODLevel(BaseModel):                   # M8
    level: int
    label: str
    visible_node_ids: list[str] | None = None   # null = 全部
    camera_distance_threshold: float

class LayoutData(BaseModel):                 # C4: 坐标系声明
    coordinate_system: Literal["right_hand_y_up"] = "right_hand_y_up"
    unit: Literal["logical"] = "logical"
    positions: dict[str, Position3D]
    bounds: BoundingBox
    suggested_camera: CameraConfig
    lod_levels: list[LODLevel] = Field(default_factory=list)

# ===== Trace =====

class Diagnostic(BaseModel):
    stage: str
    component: str | None = None
    error: str
    severity: Literal["info", "warning", "error"]

class StageTrace(BaseModel):
    name: str
    duration_ms: float
    node_count: int | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)

class PipelineTrace(BaseModel):
    stages: list[StageTrace]
    total_ms: float
    phase: Literal["A", "B"]

# ===== SSE 事件 Discriminated Union (s3) =====

class ProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    stage: str
    stage_index: int
    total_stages: int
    progress: float
    message: str
    eta_ms: int | None = None
    nodes_discovered: int | None = None
    edges_discovered: int | None = None

class ResultMetaEvent(BaseModel):
    type: Literal["result:meta"] = "result:meta"
    revision: int
    is_final: bool
    update_mode: Literal["full", "patch"]
    commit_sha: str
    pipeline_run_id: str
    confidence_summary: ConfidenceSummary
    profile: ArchitectureProfile
    features: list[str]
    resources: ResourceEstimate
    trace: PipelineTrace

class ResultGraphEvent(BaseModel):
    type: Literal["result:graph"] = "result:graph"
    revision: int
    chunk_index: int
    total_chunks: int
    update_mode: Literal["full", "patch"]
    changed_node_ids: list[str] | None = None    # patch 模式
    changed_edge_ids: list[str] | None = None
    nodes: list[ModuleNode]
    edges: list[DataEdge]
    block_templates: list[BlockTemplateInfo] = Field(default_factory=list)

class ResultFlowsEvent(BaseModel):
    type: Literal["result:flows"] = "result:flows"
    revision: int
    update_mode: Literal["full", "patch"]
    macro: list[FlowStage]
    blocks: dict[str, BlockTemplate]

class ResultLayoutEvent(BaseModel):
    type: Literal["result:layout"] = "result:layout"
    revision: int
    update_mode: Literal["full", "patch"]
    changed_position_ids: list[str] | None = None   # patch 模式
    layout: LayoutData

class ResultCompleteEvent(BaseModel):
    type: Literal["result:complete"] = "result:complete"
    revision: int
    is_final: bool
    total_events_in_revision: int
    pipeline_run_id: str

class HeartbeatEvent(BaseModel):
    type: Literal["heartbeat"] = "heartbeat"
    timestamp: float

class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    fatal: bool
    phase: Literal["A", "B"] | None = None
    details: dict[str, str] | None = None
    request_id: str

class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    status: Literal["complete", "partial", "failed"]
    phases: dict[str, Literal["complete", "skipped", "timeout", "error"]]
    total_events: int
    elapsed_ms: float

SSEEvent = Annotated[
    Union[
        Annotated[ProgressEvent, Tag("progress")],
        Annotated[ResultMetaEvent, Tag("result:meta")],
        Annotated[ResultGraphEvent, Tag("result:graph")],
        Annotated[ResultFlowsEvent, Tag("result:flows")],
        Annotated[ResultLayoutEvent, Tag("result:layout")],
        Annotated[ResultCompleteEvent, Tag("result:complete")],
        Annotated[HeartbeatEvent, Tag("heartbeat")],
        Annotated[ErrorEvent, Tag("error")],
        Annotated[DoneEvent, Tag("done")],
    ],
    Discriminator("type"),
]
```

---

### 7.11 Rate Limiting（v1.1+，v1.0 不启用）

> **原则 1（非商业化、内部工具）对齐**：v1.0 **不启用** rate limiting（不做匿名限流、不做 per-IP 计数、不引入 slowapi）。对应错误码 `RATE_LIMITED` 在 v1.0 **不产出**。以下内容保留为 v1.1+ 参考设计，实现时机视是否开放公网而定。

<!-- v1.1+ 参考设计（v1.0 未启用）

```
策略:
  全局 SSE 连接:         50 并发（BoundedSSESemaphore）
  单 IP SSE 连接:        5 并发（per-IP counter）
  Compare 接口:          10 req/min per IP
  Explore 接口:          30 req/min per IP
  Cache DELETE:          5 req/min per API key

实现:
  SSE:     BoundedSSESemaphore (in-process, 预留 Redis-backed 接口)
  REST:    slowapi + 统一错误处理器

标准响应头（所有受限接口每个响应携带）:
  X-RateLimit-Limit: 10
  X-RateLimit-Remaining: 7
  X-RateLimit-Reset: 1714000060       # Unix timestamp

429 响应:
  Retry-After: <seconds>
  {
    "error": {
      "code": "RATE_LIMITED",
      "message": "Rate limit exceeded",
      "details": { "limit": "10/min", "retry_after": "30" },
      "request_id": "req-uuid"
    }
  }

统一 slowapi 错误格式:
  @app.exception_handler(RateLimitExceeded)
  async def rate_limit_handler(request, exc):
      return JSONResponse(
          status_code=429,
          content=ErrorResponse(error=ErrorDetail(
              code="RATE_LIMITED",
              message=str(exc.detail),
          )).model_dump(),
          headers={"Retry-After": str(exc.retry_after)},
      )
```
-->

---

### 7.12 CORS 与安全（C8）

```python
# 按路径前缀分配 CORS
from fastapi.middleware.cors import CORSMiddleware

# 公开端点: allow_origins=["*"]
# 认证端点: 明确白名单

if settings.ENV == "production":
    public_origins = [settings.FRONTEND_ORIGIN]   # e.g. "https://hf-viz.example.com"
else:
    public_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=public_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Last-Event-ID", "Content-Type"],
    expose_headers=[
        "X-Request-ID", "X-Cache-Status", "X-Cache-Age",
        "X-Revision", "X-Phase-B-Status", "X-Update-Mode",
        "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset",
        "Retry-After",
    ],
    max_age=86400,                     # 24h 预检缓存
)

# 安全头
@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response
```

---

### 7.13 JSON 命名约定与 TypeScript 契约（m10）

```
约定: 所有 JSON 字段一律使用 snake_case（与 Python 生态一致, Google JSON Style Guide）。

前端对接:
  1. 后端通过 FastAPI 自动生成 OpenAPI spec (GET /api/v1/openapi.json)
  2. 前端使用 openapi-typescript 自动生成 TypeScript 类型定义
  3. SSE 事件通过 Discriminated Union 生成精确的 oneOf 类型
  4. 前端在 SSE 接收层做统一 snakeToCamel 深转换（或直接使用 snake_case）

提供官方 TypeScript 类型定义文件 (types.d.ts) 随 API 版本发布。
```

---

### 7.14 API 版本策略（m9）

```
当前版本: /api/v1/

版本演进规则:
  - 新增可选字段: 非 Breaking（客户端应忽略未知字段）
  - SSE 新增 event type: 非 Breaking（客户端应忽略未知 event）
  - 删除/重命名字段, 变更类型: Breaking → v2

废弃通知:
  - 响应头: Deprecation: true, Sunset: Sat, 01 Jan 2028 00:00:00 GMT (RFC 8594)
  - OpenAPI spec: deprecated: true 标记废弃字段
  - 版本支持窗口: N-1 版本在 N 发布后维护 6 个月
```

---

### 7.15 接口与内部模块映射

| 接口 | Router | Pipeline/Service | 缓存 segment |
|------|--------|-----------------|-------------|
| GET models/{id}/stream | `routers/model.py` | `pipeline.py` → 5 stages | 全部 6 segment |
| POST compare | `routers/compare.py` | `comparison.py` → parallel pipelines | structure + features |
| GET models/{id}/parallel | `routers/parallel.py` | `parallel/strategy_composer.py` | structure + layout |
| GET models | `routers/models.py` | `infra/hf_client.py` → search API | 无缓存 |
| DELETE models/{id}/cache | `routers/cache.py` | `cache_strategy.py` → invalidate | 全部 segment |
| GET health/* | `routers/health.py` | admission + pool status | 无 |

---

### 7.16 评审采纳追踪

| 编号 | 采纳 | 体现位置 |
|------|------|---------|
| C1 Event ID 重连 | ✅ | 7.3.2, 7.3.3 |
| C2 result 拆分 | ✅ | 7.3.1, 7.3.5 |
| C3 rev2 增量更新 | ✅ | 7.3.9 |
| C4 Layout 完备 | ✅ | 7.3.7, 7.10 Position3D |
| C5 Semaphore 生命周期 | ✅ | 7.2.4, 7.3.15 |
| C6 客户端断开检测 | ✅ | 7.3.15 |
| C7 Pydantic v2 迁移 | ✅ | 7.4 CompareRequest |
| C8 CORS 分级 | ✅ | 7.12 |
| M1 SSE 端点分离 | ✅ | 7.1, 7.3 |
| M2 错误码映射 | ✅ | 7.9.2 |
| M3 error fatal 区分 | ✅ | 7.3.11 |
| M4 Rate Limit 标准头 | ✅ | 7.11 |
| M5 done 语义 | ✅ | 7.3.12 |
| M6 backpressure | ✅ | 7.3.15 Queue(maxsize=64) |
| M7 分阶段超时 | ✅ | 7.3.14 |
| M8 template + LOD | ✅ | 7.3.5, 7.3.7, 7.10 |
| M9 DataEdge 增强 | ✅ | 7.3.5, 7.10 DataEdge |
| M10 flows.macro 结构化 | ✅ | 7.3.6, 7.10 FlowStage |
| M11 ByteSize 结构化 | ✅ | 7.3.4, 7.10 ByteSize |
| M12 ScheduleStep 时间 | ✅ | 7.5 |
| M13 Compare 对齐 | ✅ | 7.4 |
| M14 repo_id 校验 | ✅ | 7.2.1 |
| M15 类型约束 | ✅ | 7.2.2, 7.3 |
| M16 ModuleType 枚举 | ✅ | 7.2.2, 7.10 |
| M17 task cancellation | ✅ | 7.3.15 |
| m1 /explore → /models | ✅ | 7.1, 7.6 |
| m2 cursor 分页 | ✅ | 7.6 |
| m3 Health 503 | ✅ | 7.8 |
| m4 ConfidenceLevel 复用 | ✅ | 7.2.2, 7.10 |
| m5 Optional = None | ✅ | 7.10 全部 |
| m6 tensor_dim_names | ✅ | 7.3.5, 7.10 |
| m7 SSE 响应头 | ✅ | 7.3.13 |
| m8 CameraConfig 完备 | ✅ | 7.3.7, 7.10 |
| m9 版本淘汰策略 | ✅ | 7.14 |
| m10 命名约定 + TS | ✅ | 7.13 |
| s1 邻接索引 | ✅ | 7.10 ModuleNode |
| s2 progress 增强 | ✅ | 7.3.10 |
| s3 Discriminated Union | ✅ | 7.10 SSEEvent |

---

Section 7 修订完成。共定义 **8 个端点**，**9 种 SSE 事件类型**（Discriminated Union），**30+ 个 Pydantic 模型**，完整的 Event ID 重连协议、分阶段超时、错误码映射表、Rate Limit 标准头、CORS 分级策略和 API 版本演进规则。所有 38 条评审意见已逐条落实。

请确认后进入 Section 8 风险与待办。

---

## 八、风险与待办

> 汇总技术、外部依赖、运营、安全、部署风险，定义 SLO 体系、监控告警、容量模型、路线图。

---

### 8.1 SLO / SLI / Error Budget（C1 新增）

> **v1.0 修订说明（原则 1 / 原则 5 / 原则 10）**：
> - 本节 SLO 指标（99.5% 可用性、TTFB P95 < 5s、SSE 完成率 > 95%、Phase A P95 < 10s 等）**在 v1.0 仅作观测指标，不作准入门槛**（原则 5）。Phase 0/1/2 达不到先记录不阻塞发布，Phase N 独立优化阶段再拉门槛。
> - **交互响应延迟例外**（原则 5 例外条款）：`PATCH /config` 后端 < 200ms、端到端 < 300ms、模块点选 < 50ms、动画 scrub < 16ms/frame、WS 本机 < 50ms **属于硬约束**，Phase 1 起必达，不因"前期不做性能优化"豁免。
> - 本节涉及的 **Admission / Circuit Breaker / 组合熔断 / setrlimit / SSE 绝对超时 / 全局 SSE 硬上限 / HF_TOKEN 策略 / 异常 repo_id 监控** 等"公网级硬化"内容，按**原则 1（非商业化、内部工具）+ 原则 10（拒绝过度工程）** 处理：
>   - **v1.0 不引入** admission controller、circuit breaker、rate limiting、SSE 绝对超时熔断（保留最小自卫：`trust_remote_code=False`）。
>   - 对应风险项（T1/T4/T14、O1/O2/O6/O7）在 v1.0 降级为"观察 + 手动处置"，不做自动化硬化。v1.1+ 若开放公网访问再按 ADR 流程补齐。

#### 8.1.1 服务级目标

| SLI | SLO 目标 | 测量方式 | 告警触发 |
|-----|---------|---------|---------|
| **可用性** | 99.5%（月度） | 成功请求 / 总请求（排除 4xx） | Error Budget 消耗 >50% |
| **SSE 首事件延迟 (TTFB)** | P95 < 5s, P99 < 15s | HTTP 连接建立 → 第一个 `result:meta` 事件 | P95 > 5s 持续 5min |
| **SSE 完成率** | > 95% | 收到 `done(status=complete)` / 总 SSE 连接 | < 90% 持续 10min |
| **Phase A 延迟** | P95 < 3s（缓存命中）, P95 < 10s（缓存 miss） | pipeline Phase A 总耗时 | P95 > 10s |
| **Compare 延迟** | P95 < 10s（quick）, P95 < 30s（deep） | POST /compare 响应时间 | P95 > 15s(quick) |

#### 8.1.2 Error Budget

```
月度 Error Budget = (1 - 0.995) × 30d × 24h × 60min = 216 min ≈ 3.6h

消耗追踪:
  Budget 消耗 > 30% → Warning（P2）
  Budget 消耗 > 50% → 冻结非关键变更（P1）
  Budget 消耗 > 80% → 全力修复可靠性（P0）
```

---

### 8.2 技术风险（原 10 项 + 新增 5 项 = 15 项）

| # | 风险 | 影响 | 概率 | 等级 | 缓解策略 | 阶段 |
|---|------|------|------|------|---------|------|
| **T1** | meta-device OOM（70B+ 模型 cgroup 限制） | Phase B 不可用 | 中 | 高 | Admission cgroup 检查（安全边际 60%）+ SIGKILL + circuit breaker 3 次暂停 5min；cgroup `memory.oom.group=1` 确保 OOM 仅杀 cgroup 内进程（C8 联动） | P2 |
| **T2** | transformers 版本 Breaking Change | 解析中断 | 高 | 高 | 版本锁定 pin + CI 周跑 top-50 兼容性 + 二阶段升级 | P3 |
| **T3** | SSE 大 payload 代理截断 | 客户端解析失败 | 中 | 高 | result:graph 语义级分 chunk ≤200KB + `Content-Encoding: gzip` + `proxy_buffering off` + HTTP/2；监控 `sse_chunk_size_p99` | P2 |
| **T4** | HF Hub 不可用/限流 **+ SSRF 风险**（M16） | 新模型失败 + 信息探测 | 中 | 中 | tenacity 退避 + 3 级降级 + 缓存兜底；HF_TOKEN 限只读公开仓库；统一错误消息防枚举；监控异常 repo_id 模式 | P2 |
| **T5** | ProcessPool 进程泄漏 **+ 全部不可恢复**（M11） | 进程池耗尽 | 低 | 高 | SIGKILL + warmup 分级 + 监控；连续 3 次 spawn 失败 → DEGRADED 状态快速拒绝 503；pool 操作 `wait_for(30s)` 超时 | P2 |
| **T6** | asyncio 事件循环阻塞 | SSE 卡顿 | 中 | 中 | **时间预算制**（M13）：预估执行 >50ms 才 offload ThreadPoolExecutor（替代固定 500 阈值）；基于 `pipeline_stage_duration` 校准 | P2 |
| **T7** | 缓存一致性窗口 | 用户看旧数据 | 高 | 低 | volatile repo TTL 1h + STALE 标记 + 手动刷新；**RevisionFence 始终解析到 commit SHA**（架构师-R21） | P2 |
| **T8** | L1 磁盘写满 | Pod eviction | 低 | 高 | AtomicCounter quota + emergency_gc(触发点 75%) + 80% 告警 | P2 |
| **T9** | 前端 3D 帧率崩溃 | FPS < 30 | 中 | 中 | LOD + 分 chunk 渲染 + block_templates 折叠 + instanced mesh | P2-3 |
| **T10** | SSE Semaphore 泄漏 | 服务不可用 | 低 | 高 | 双层 try-except + 监控；**自愈**：周期比对活跃连接数 vs Semaphore 已获取数，差异超阈值强制重置 + 告警；带 deadline 超时自动释放（90s） | P2 |
| **T11** | **Pipeline 纯函数契约违反**（C5 新增） | 数据污染 | 低 | 高 | 入口 frozen dataclass / deepcopy(debug 模式)；阶段间 schema 断言；CI 纯函数属性测试；`pipeline_schema_violation_total` 监控 | P2 |
| **T12** | **L2 meta-device 架构兼容性失败**（C6 新增） | L2 完全不可用（非 OOM） | 中 | 高 | try-except 降级到 L1 + SSE 标注 `degraded_level: "L1"`；维护不兼容架构 blocklist；`l2_instantiation_failure_rate` 按架构分维度监控 | P2 |
| **T13** | **前后端 SSE Schema 版本漂移**（M10 新增） | 前端静默丢弃关键事件 | 中 | 中 | SSE 事件嵌入 `schema_version` 字段；前端 ignore+log 未知 type；`/health/startup` 暴露 schema 版本；前端启动校验 | P2 |
| **T14** | **组合攻击 DoS**（C8 新增） | 多向量联动服务不可用 | 低 | 高 | SSE 绝对超时 300s；per-task 内存 setrlimit；全局并发 SSE 硬上限 50；各资源限制联动熔断（任一达 90% → 全局 admission 收紧） | P2 |
| **T15** | **滚动更新断裂 SSE**（C2 新增） | 每次发版用户中断 | 高 | 高 | `terminationGracePeriodSeconds=90`；preStop hook 停止新连接 + 等待存量完成；`maxSurge=1, maxUnavailable=0` | P1 |

---

### 8.3 外部依赖风险

| # | 依赖 | 风险 | 缓解 |
|---|------|------|------|
| E1 | HuggingFace Hub API | API 变更/限流/不可用/**政策变更** | SDK 抽象层 + 镜像 fallback + 缓存兜底；**v1.2 备用数据源(GitHub 直下)** |
| E2 | transformers | 大版本 Breaking Change | 版本锁定 + CI 兼容性矩阵 + adapter 隔离层 |
| E3 | PyTorch (CPU-only) | meta-device 跨版本不一致 | 版本锁定 + ProcessPool 隔离 + Phase B 可禁用 |
| E4 | safetensors | 格式演进 | 防御性解析：未知字段忽略，必需字段缺失降级 |
| E5 | Diffusers | pipeline 发现机制变更 | find_denoiser 白名单 + 通用 fallback |
| E6 | Docker 基础镜像 | 安全补丁 | 月度更新 + CVE 扫描（M19） |
| **E7** | **供应链安全**（M19 新增） | 传递依赖投毒 / pickle RCE | pip-audit CI 扫描 + 版本锁定 + hash 校验(--require-hashes) + 全局禁止 torch.load + Dependabot |

---

### 8.4 运营风险

| # | 风险 | 影响 | 缓解 |
|---|------|------|------|
| O1 | repo_id 注入 | 路径穿越/SSRF | 三层防御（具体化）：① regex `^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$` + max_length=200；② `..`/`\x00`/Unicode 非 ASCII 黑名单；③ HF SDK 调用（非 URL 拼接）+ 缓存路径 resolve 验证 |
| O2 | DDoS SSE | Semaphore 耗尽 | Rate limiting + 429 + 组合熔断（T14） |
| O3 | 信息泄漏 | 内部架构暴露 | details 类型 `dict[str,str]`；5xx 仅 request_id；异常消息白名单/模板；DEBUG_MODE 环境变量 |
| O4 | 镜像过大 | 部署慢 | 多阶段构建预计 1.3-2.0GB（待实测，README §技术栈）；未来 distroless |
| O5 | Semaphore 无法水平扩展 | 多 Pod 限流失效 | v1.0 本地 Semaphore + **Nginx limit_conn**（M14 简化方案）；v1.1 评估 Redis consumer group |
| **O6** | **容器安全加固缺失**（M17 新增） | 攻击面大 | `USER nonroot` + `--read-only` rootfs（/data/cache + /tmp 可写）+ `cap-drop=ALL` + NetworkPolicy 仅 `*.huggingface.co:443` + `tini` init 进程 |
| **O7** | **CORS 生产泄露**（M20 新增） | 任意域跨域请求 | 默认空白名单；`ENV=production + allow_origins=["*"]` → 拒绝启动；CI 配置检查 |
| **O8** | **日志敏感信息**（m2 新增） | 信息泄露 + 日志注入 | 路径脱敏仅保留最后两级；禁止 token/credential 入日志；结构化 JSON 日志(sanitize 换行符/控制字符) |
| **O9** | **平台依赖风险**（PM-R9 新增） | HF 政策变更/推出竞品 | v1.2 备用数据源(GitHub)；主动沟通 HF 合作可能性 |

---

### 8.5 安全监控（M18 新增）

| 指标 | 含义 | 告警阈值 |
|------|------|---------|
| `repo_id_validation_rejected_total` | 三层防御拒绝数（按层分类） | >100/min(可能扫描) |
| `rate_limit_exceeded_total{ip}` | 限流触发（按 IP 分类） | 单 IP >50/min |
| `cache_delete_total` | 缓存清除操作数 | >20/min(可能滥用) |
| `worker_oom_killed_total` | Worker OOM 被杀数 | >0 |
| `sse_connection_timeout_total` | SSE 超时断开数 | >10% |

---

### 8.6 已知限制

| # | 限制 | 用户感知影响 | v1.0 缓解 | 计划版本 |
|---|------|-----------|----------|---------|
| L1 | 不支持 trust_remote_code | **高** — DeepSeek-V3/Kimi-K2.6 等头部模型仅 config-only（C7） | **内置金牌缓存**（Top-N 模型预生成完整数据）+ UI 标注"部分解析" | v1.0.1 沙箱评估 |
| L2 | 不支持 GGUF/ONNX/TensorRT | **中** — 基础设施工程师常用格式不支持 | UI 明确标注"支持 HF transformers 格式" | **v1.1**（GGUF 基础支持提前） |
| L3 | Parallel 静态估算 | 低 — 不考虑通信拓扑 | 文档标注"估算值" | v2.0 |
| L4 | Compare deep 限 2 模型 | 低 — 资源约束 | 400 错误含 `max_repos_for_deep: 2` | v1.1 排队 |
| L5 | 无用户认证体系 | **中** — 无 per-user 配额 | **v1.0 实现 API Key 认证**（简单方案：环境变量配置 + 中间件校验），降低安全暴露面 | v1.1 OAuth2 |
| L6 | 单实例部署 | **高** — 无水平扩展 | Nginx limit_conn 辅助；HF Spaces 标注并发限制 | v1.0.1 |
| L7 | SSE 单向通信 | **中** — 无法取消分析 | **v1.0 监听连接断开主动中止**（request.is_disconnected → cancel pipeline_task，零额外端点） | v1.1 cancel 端点 |
| L8 | JSON 非二进制 | 低 — 传输效率 | **启用 gzip 压缩**（Nginx 或 middleware）；JSON 排除冗余字段 | v1.2 |

> **变更 [C7]**：L1 影响上调为"高"，增加金牌缓存缓解。  
> **变更 [PM-R3]**：L7 在 v1.0 实现轻量取消（监听断开）。  
> **变更 [架构师-R13]**：L5 在 v1.0 实现简单 API Key 认证。

---

### 8.7 部署与运维规范（C2/C3/C4 + SRE 评审新增）

#### 8.7.1 K8s 资源配置（C3）

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hf-model-visualizer
spec:
  replicas: 1
  revisionHistoryLimit: 5              # M1: 回滚支持
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1                      # C2: 先拉新再终旧
      maxUnavailable: 0
  template:
    spec:
      terminationGracePeriodSeconds: 90  # C2: 覆盖最长 75s SSE
      containers:
      - name: api
        image: hf-viz:${GIT_SHA}        # M1: 不可变 tag
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
            ephemeral-storage: "7Gi"     # m4: 节点调度保障
          limits:
            cpu: "4000m"
            memory: "3Gi"
        securityContext:                 # M17: 容器安全
          runAsNonRoot: true
          runAsUser: 65534
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: cache
          mountPath: /data/cache
        - name: tmp
          mountPath: /tmp
        livenessProbe:                   # M4
          httpGet: { path: /health/live, port: 8000 }
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet: { path: /health/ready, port: 8000 }
          periodSeconds: 5
          failureThreshold: 2
        startupProbe:
          httpGet: { path: /health/startup, port: 8000 }
          periodSeconds: 5
          failureThreshold: 12          # 最多等 60s 完成 warmup
        lifecycle:
          preStop:                       # C2: 优雅终止
            exec:
              command: ["/bin/sh", "-c", "touch /tmp/shutdown && sleep 75"]
      volumes:
      - name: cache
        emptyDir: { sizeLimit: "6.5Gi" }
      - name: tmp
        emptyDir: { sizeLimit: "500Mi" }
---
apiVersion: policy/v1
kind: PodDisruptionBudget               # M3
metadata:
  name: hf-viz-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels: { app: hf-model-visualizer }
```

#### 8.7.2 冷启动预热（C4）

```
Pod 启动流程:
  1. initContainer: 从 ConfigMap 读取 Top-N 热门模型列表
  2. startup hook: 逐个预热到 L0 缓存（从 L1 文件或 Hub 拉取）
  3. 预热期间 admission semaphore 收紧为 1（减少 Hub 并发）
  4. readiness probe 检查 warmup 完成 + worker 就绪后开始接流量
  5. warmup 超时上限 45s（不阻塞 startup probe 的 60s 上限）

v1.1 迁移到 PVC 后: Pod 重启不丢失 L1 缓存
```

#### 8.7.3 回滚策略（M1）

```
回滚 SOP:
  1. 监控检测: error_rate_5xx > 5% 持续 5min → 自动告警
  2. 确认回滚: kubectl rollout undo deployment/hf-model-visualizer
  3. 验证恢复: 确认 error_rate 恢复正常 + SLO 恢复
  4. 通知团队: 自动发送回滚报告

镜像 tag 规范:
  - 生产: git SHA 或 semver (v1.0.0)
  - 禁止: latest tag
  - revisionHistoryLimit >= 5
```

#### 8.7.4 配置热更新（M2）

```
短期 (v1.0): ConfigMap → 环境变量 → 滚动重启
  - SSE_MAX_CONNECTIONS=50
  - ADMISSION_SEMAPHORE=2
  - PHASE_A_TIMEOUT=15
  - PHASE_B_TIMEOUT=60
  - GC_THRESHOLD_PERCENT=75
  - LOG_LEVEL=INFO

中期 (v1.1): /admin/config endpoint
  - 支持运行时变更: LOG_LEVEL, SSE_MAX_CONNECTIONS, ADMISSION_SEMAPHORE
  - 变更立即生效，无需重启
  - 变更记录审计日志
```

#### 8.7.5 容量模型（s2）

```
单 Pod 容量上界:
  瓶颈: Admission Semaphore = 2
  最长加载: 75s (Phase A 15s + Phase B 60s)

  冷态（缓存 miss 100%）: 2 / 75s × 60s ≈ 1.6 req/min
  热态（缓存命中 80%）: ~8-10 req/min
  SSE 连接上限: 50 / 75s ≈ 40 new req/min（非瓶颈）

  真实瓶颈: Admission Semaphore（非 SSE Semaphore）

扩容触发条件:
  - admission 排队等待 P95 > 5s
  - Semaphore(2) 利用率持续 > 80% (即 2/2 持续 > 60s)
  - SSE Semaphore 利用率 > 70%

容量公式:
  所需 Pod 数 = 峰值 req/min / (单 Pod 吞吐 × 0.8)

v1.1 HPA 策略:
  主指标: sse_semaphore_current / 50 (利用率)
  辅指标: CPU 利用率 > 70%
  scaleDown.stabilizationWindowSeconds: 300
  minReplicas: 2, maxReplicas: 5
```

---

### 8.8 监控与告警（原 13 项 + 新增 12 项 = 25 项）

#### 8.8.1 完整指标清单

| 分类 | 指标 | Warning(P2) | Critical(P1) | Fatal(P0) |
|------|------|------------|-------------|-----------|
| **SLI 用户体验** | `sse_time_to_first_event` (M8) | P95 > 3s | P95 > 5s | P95 > 15s |
| | `sse_total_duration` (M8) | P95 > 45s | P95 > 60s | P95 > 75s |
| | `sse_client_disconnect_rate` (M5) | > 10% | > 20% | > 40% |
| | `sse_completion_rate` | < 95% | < 90% | < 80% |
| **SSE 连接** | `sse_semaphore_current` | > 35 (70%) | > 45 (90%) | = 50 持续 >60s |
| | `semaphore_wait_duration` (M8) | P95 > 3s | P95 > 10s | P95 > 30s |
| **Pipeline** | `pipeline_stage_duration{phase=A}` | P95 > 5s | P95 > 10s | P95 > 15s |
| | `pipeline_schema_violation_total` (C5) | > 0 | — | — |
| | `l2_instantiation_failure_rate` (C6) | > 10% | > 30% | > 50% |
| **缓存** | `cache_l0_hit_rate` (M7) | < 70% | < 50% | < 30% |
| | `cache_l1_hit_rate` (M7) | < 60% | < 40% | — |
| | `cache_miss_to_hub_rate` (M7) | > 30% | > 50% | — |
| | `cache_l0_memory_bytes` | > 300MB | > 400MB | — |
| | `cache_l1_disk_bytes` | > 4.9GB(75%) | > 5.2GB(80%) | > 5.9GB(90%) |
| | `cache_l1_rejected_total` | > 0/min | > 5/min | — |
| | `gc_duration_seconds` | > 3s | > 5s | > 10s |
| **ProcessPool** | `pool_workers_active` | = max-1 | = max 持续 >60s | — |
| | `process_pool_restart_total` (M5) | > 1/hour | > 3/hour | > 5/hour |
| | `pool_spawn_failure_consecutive` (M11) | ≥ 2 | ≥ 3 | — |
| **Hub** | `hub_request_duration` | P95 > 3s | P95 > 5s | P95 > 10s |
| | `hub_error_rate` | > 5%(5min) | > 10%(5min) | > 30%(5min) |
| **基础设施** | `container_memory_working_set_bytes` (M5) | > 2.0GB | > 2.5GB | > 2.8GB |
| | `error_rate_5xx` | > 1%(5min) | > 5%(5min) | > 20%(5min) |
| | `http_request_duration{endpoint}` | 按端点分别设置 | — | — |
| **安全** | `repo_id_validation_rejected_total` | > 50/min | > 100/min | — |

> **变更 [M9]**：每项指标三级告警(Warning/Critical/Fatal)。  
> **变更 [M5/M7/M8]**：新增用户体验、缓存命中率、容器内存等关键 SLI。  
> 所有 HTTP 指标包含 `endpoint` label 维度。

#### 8.8.2 可观测性体系（M6 + m3）

```
Tracing (M6 — OpenTelemetry):
  每 SSE 请求生成 trace_id
  关键 span: http_request → semaphore_acquire → hub_download → model_load → sse_stream
  跨进程: ProcessPool worker 通过 span context 传播
  trace_id 注入 SSE result:meta 事件 + 结构化日志

Logging (m3):
  格式: 结构化 JSON (stdout)
  收集: Fluentd/Vector DaemonSet → Loki/ES
  脱敏: 路径仅保留最后两级；禁止 token；sanitize 控制字符

Dashboards (m3 — Grafana 预置 3 个):
  1. Overview: 请求量、错误率、SLO 达标率、Error Budget 消耗
  2. Resource: CPU/Memory/Disk/Semaphore 利用率、ProcessPool 状态
  3. Cache: L0/L1 命中率、Hub 回源率、GC 频率、RevisionFence 淘汰数
```

---

### 8.9 待办路线图（M15/PM-R8 重构）

#### v1.0 MVP

- [x] 4 层解析 + 5 阶段 pipeline + SSE 双阶段推送
- [x] 二级缓存 + RevisionFence + Admission + ResilientProcessPool
- [x] 8 端点 + 30+ Pydantic + Discriminated Union + Event ID 重连
- [x] **简单 API Key 认证**（L5 前移）
- [x] **金牌缓存：Top-N 热门模型预生成数据**（C7）
- [x] **监听 SSE 断开主动中止 pipeline**（L7 轻量取消）
- [x] **gzip 压缩**（L8 缓解）
- [x] **容器安全加固**（O6: nonroot + readonly + cap-drop）
- [x] **CORS 生产兜底**（O7: 默认空白名单）
- [x] **K8s 部署规范**（C2/C3: 资源限制 + 优雅终止 + 探针）
- [x] **localStorage 最近查看 + URL 分享按钮**（m10 零成本留存）
- [x] SLO 定义 + 25 项监控指标 + 三级告警

#### v1.0.1 Hotfix（GA + 2 周内）

| 待办 | 来源 | 优先级 |
|------|------|--------|
| HF Spaces 部署 + 并发限制标注 | PM-R4 | P0 |
| trust_remote_code 沙箱方案评估 | C7/L1 | P0 |
| GGUF 基础支持（文件头解析） | L2 提前 | P1 |
| camelCase 决策（做则此时做，不做则从路线图删除） | PM-R8 | P1 |

#### v1.1 增强（v1.0 GA + 6-8 周）

| 待办 | 来源 | 优先级 |
|------|------|--------|
| L3 AST 解析 — trust_remote_code 沙箱执行 | L1 | P0 |
| Cancel 端点 `DELETE /api/v1/models/{id}/stream/{run_id}` | L7 | P0 |
| OAuth2 + per-user Rate Limiting | L5/架构师-R13 | P0 |
| Redis 分布式缓存（L0 迁移 + PVC L1） | L6/O5 | P1 |
| Compare deep 排队策略（3-4 模型） | L4 | P1 |
| JSON Patch (RFC 6902) rev2 增量更新 | Section 7 C3 | P2 |
| `GET /api/v1/models/{id}` 同步 JSON 端点 | Section 7 M1 | P2 |
| HTML 导出 + CLI 工具 | PM 路线图 | P2 |

#### v1.2 优化

| 待办 | 来源 | 优先级 |
|------|------|--------|
| GGUF/ONNX/TensorRT 完整支持 | L2 | P1 |
| Binary SSE (MessagePack/Protobuf) | L8 | P1 |
| 备用数据源（GitHub 直下 config.json） | O9/E1 | P1 |
| AsyncAPI spec | Section 7 | P2 |
| graph presigned URL（超大模型） | Section 7 | P2 |

#### v2.0 并行策略增强

| 待办 | 来源 | 优先级 |
|------|------|--------|
| 通信拓扑感知（NVLink/IB） | L3 | P0 |
| 新策略维度（Ring Attention/Ulysses） | Section 7 | P1 |
| 策略参数结构化对象 | Section 7 | P1 |
| `GET /api/v1/parallel/options` 元信息 | Section 7 | P2 |

#### PMF 检查点（PM-R2）

```
v1.0 GA + 30 天检查:
  北极星指标（月度唯一 model_id）>= 200
  7 日再访率 >= 10%（v1.0 无账号系统下合理预期）
  3D 加载成功率 >= 90%

未达标 → 优先级从"功能增强"转向"用户调研 + 易用性修复"
v1.1 有 OAuth2 后再访率目标上调至 >= 20%
```

---

### 8.10 合规（m7/m8 新增）

```
GDPR:
  - Rate Limiting IP 仅存内存，窗口过期自动清除
  - 访问日志 IP 截断最后一段 octet（如 192.168.1.x）
  - 日志保留策略: 30 天自动清理
  - v1.1 增加隐私政策页面

许可证审计:
  - CI 集成 pip-licenses 扫描
  - 白名单: MIT / BSD / Apache 2.0 / ISC
  - 拒绝 GPL/AGPL 依赖进入生产镜像
```

---

### 8.11 评审遗留追踪（m9 增加 owner/deadline）

| 编号 | 内容 | 延后原因 | 目标版本 | Owner | Deadline |
|------|------|---------|---------|-------|---------|
| R-01 | Parallel 接口结构化参数 | v1.0 query params 足够 | v2.0 | 后端 Lead | v2.0 GA |
| R-02 | AsyncAPI spec | 文档工作 | v1.2 | 前端 Lead | v1.2 GA |
| R-03 | Compare 改 GET | **v1.0.1 决策**（公开前完成） | v1.0.1 | API Owner | GA + 2 周 |
| R-04 | 动画过渡参数 | 纯 UI 关注点，前端自行决定 | 不纳入 | — | — |
| R-05 | DataEdge 路由 waypoints | compute_layout 待实现 | v1.1 | 后端 Layout | v1.1 GA |

> **变更 [m9]**：每项增加 owner + deadline。R-03 提前到 v1.0.1（避免 Breaking Change）。

---

### 8.12 评审采纳追踪

| 编号 | 采纳 | 体现位置 |
|------|------|---------|
| C1 SLO/SLI/Error Budget | ✅ | 8.1 |
| C2 滚动更新 SSE 断裂 | ✅ | 8.2 T15 + 8.7.1 |
| C3 K8s 资源配置 | ✅ | 8.7.1 |
| C4 冷启动预热 | ✅ | 8.7.2 |
| C5 Pipeline 纯函数契约 | ✅ | 8.2 T11 |
| C6 L2 架构兼容性 | ✅ | 8.2 T12 |
| C7 trust_remote_code 金牌缓存 | ✅ | 8.6 L1 + 8.9 v1.0 |
| C8 组合攻击 DoS | ✅ | 8.2 T14 |
| M1 回滚策略 | ✅ | 8.7.3 |
| M2 配置热更新 | ✅ | 8.7.4 |
| M3 HPA + PDB | ✅ | 8.7.1 + 8.7.5 |
| M4 探针配置 | ✅ | 8.7.1 |
| M5 缺失 SLI | ✅ | 8.8.1 |
| M6 OpenTelemetry | ✅ | 8.8.2 |
| M7 缓存命中率 | ✅ | 8.8.1 |
| M8 端到端延迟 | ✅ | 8.8.1 |
| M9 告警分级 | ✅ | 8.8.1 三级阈值 |
| M10 Schema 版本漂移 | ✅ | 8.2 T13 |
| M11 Pool 不可恢复 | ✅ | 8.2 T5 |
| M12 zstd 文件损坏 | ✅ | 8.2 T3 注释 |
| M13 T6 阈值量化 | ✅ | 8.2 T6 |
| M14 Semaphore 简化方案 | ✅ | 8.4 O5 |
| M15 v1.1 拆分 | ✅ | 8.9 v1.0.1 + v1.1 |
| M16 SSRF | ✅ | 8.2 T4 |
| M17 容器安全 | ✅ | 8.4 O6 + 8.7.1 |
| M18 安全监控 | ✅ | 8.5 |
| M19 供应链安全 | ✅ | 8.3 E7 |
| M20 CORS 兜底 | ✅ | 8.4 O7 |
| M21 PMF 时间线 | ✅ | 8.9 PMF 检查点 |
| m1 告警阈值校准 | ✅ | 8.8.1 三级阈值 |
| m2 日志脱敏 | ✅ | 8.4 O8 + 8.8.2 |
| m3 Dashboard 预置 | ✅ | 8.8.2 |
| m4 emptyDir 节点调度 | ✅ | 8.7.1 ephemeral-storage |
| m5 spawn 开销量化 | ✅ | 8.7.5 备用 worker 建议 |
| m6 缓存路径安全编码 | ✅ | 8.4 O1 resolve 验证 |
| m7 GDPR 合规 | ✅ | 8.10 |
| m8 许可证审计 | ✅ | 8.10 |
| m9 遗留追踪 owner/deadline | ✅ | 8.11 |
| m10 留存机制 | ✅ | 8.9 v1.0 |
| s1 v0.9 alpha 检查点 | ✅ | 8.9 PMF 检查点 |
| s2 容量模型 | ✅ | 8.7.5 |
| s3 竞品壁垒策略 | ✅ | 8.9 路线图注释 |

---

---

## 变更日志

| 日期 | 版本 | 变更内容 |
|---|---|---|
| 2026-04-25 | v1.0 | 初始详细设计文档：Section 1-8 全部完成，含多轮 Agent Team 评审 |

---

> 本文档由 HF Model Visualizer 后端详细设计评审流程产出。
> 每个 Section 经过 4 角色 Agent Team 评审 → 合成去重 → 逐条修订。
