# HF Model Visualizer — 技术设计文档

> HuggingFace 模型结构 3D 交互式可视化 Web 服务 | v1.0

## 产品原则（北极星，不可妥协）

以下原则是本产品的存在理由，任何设计决策与之冲突时，原则优先、方案让步。

1. **非商业化定位**：本产品不考虑商业化，仅供作者本人及团队内部成员使用。因此不做多租户、配额、计费、公开注册、SEO、营销漏斗；不为"潜在付费用户"做妥协，只为"我们自己用得爽"做优化。
2. **精美 3D 风格 + 现代化前端**：必须是 3D 优先（2D 不在 v1.0 范围），视觉质感必须精美（材质、光照、后处理、微交互、排版、动效都要经得起截图分享）；前端栈必须现代化（React 19 / Next 15 / R3F ^9 / Tailwind / 类型安全），拒绝"能跑就行"的审美与实现。
3. **结构与数据流 100% 正确**：模型结构展示、推理数据流动画、未来的各种并行策略（TP/PP/DP/EP/CP/SP）数据流动画，必须与真实模型行为 100% 一致。**优先级顺序：正确性 > 教学性 > 美观 > 性能 > 进度**。宁可不展示，不可错展示；任何无法确证的内容必须通过 Provenance 明确标注为 `INFERRED`/`ESTIMATED`，不得伪装成 `EXACT`。
4. **教学深度与动画精细度大幅超越竞品**：在教学叙事与动画精细度两个维度，必须明显超越 LLM Visualization (bbycroft.net/llm)、Transformer Explainer、Netron、Model Explorer。做不到"看一眼就懂 Attention/MoE/MLA 在算什么"不算合格。
5. **前期不做性能优化（但交互响应延迟不算性能优化）**：项目前期（Phase 0 / Phase 1 / Phase 2）以**正确性 + 教学性**为唯一交付标准，不为性能做任何妥协式取舍——不砍动画细节、不降模型规模、不做 LOD 降级、不做 frameloop 限制。性能预算（30fps / L0<10ms / 冷启动<3s）在前期仅作**观测指标**，不作**准入门槛**；达不到就先记录，不阻塞发布。性能优化作为独立阶段（Phase N）在正确性与教学性验收通过后启动。
   
   **例外**：**交互响应延迟**（config 编辑后重渲染、视角切换、模块点选、动画时间轴拖动等）不属于"性能优化"范畴，而是**功能正确性的一部分**——延迟过高会直接破坏教学节奏与"结构与数据流 100% 正确"的直观验证。交互响应预算（如 config 重算 < 300ms、点选 < 50ms）是硬约束，必须从 Phase 1 起即满足。
6. **v1.0 架构必须最健壮且可扩展**：v1.0 虽然功能收敛，但架构不允许"一次性"。必须为后续扩展预留清晰扩展点，包括但不限于：
   - **模型类型扩展**：新增架构族（Mamba/RWKV/Diffusion/ViT/多模态/Encoder-Decoder/MoE 变体 等）不应修改核心管线，只新增 `ArchitectureAdapter` 即可。禁止在 `synthesize_flows` / `compute_layout` 中写 `if model_type == "llama"` 式分支。
   - **结构粒度扩展**：`ModuleGraph` 必须能表达从 Layer → Block → Op → Tensor 四级粒度，v1.0 只渲染 Block 级但底层数据结构已预留 Op/Tensor 级槽位。
   - **动画精细度扩展**：动画系统采用 `AnimationLayer` 插件式叠加（Stage-1 结构动画 / Stage-2 数据流动画 / Stage-3 数值热力 / Stage-4 并行策略）；各层可独立开关、独立演进，层间通过声明式时间轴协调，不得硬编码相互依赖。
   - **并行策略扩展（带数据流动画）**：TP / PP / DP / EP / CP / SP 以及其组合（3D/4D/5D parallelism）必须作为 `ParallelismStrategy` 插件注册，每种策略声明其对 ModuleGraph 的切分/复制规则 + 通信原语（AllReduce/AllGather/ReduceScatter/All2All/P2P）+ 时序动画脚本。新增策略 = 新增 1 个策略文件，不动核心渲染与 ModuleGraph。
   - **前向/反向传播扩展（带数据流动画）**：`DataFlowDirection` 作为一等公民（`forward` / `backward` / `forward_backward`），ModuleGraph 的 DataEdge 必须能表达反向梯度流、激活值缓存（activation checkpointing）、梯度累积等；动画层需支持前向正放 / 反向逆放 / 前反向对照同屏三种模式。
   - **显存估计扩展（Megatron / FSDP + GPU 选型）**：`MemoryEstimator` 作为插件接口，v1.0 后续迭代至少支持 **Megatron-LM（TP+PP+SP）**、**FSDP（ZeRO-1/2/3）** 两种估计器；必须提供 **GPU 型号选择器**（A100-40G / A100-80G / H100-80G / H200-141G / B200 / 4090-24G / 3090-24G / L40S / 国产卡如昇腾 910B / 寒武纪 / 昆仑芯 等），GPU 清单与规格（显存容量 / 带宽 / FP16/BF16/FP8 算力）作为可扩展的数据表，不硬编码。
   - **配置参数动态编辑**：前端必须支持用户**就地修改模型 config 参数**（如 num_layers / hidden_size / num_experts / num_heads / moe_top_k / tp_size / pp_size / micro_batch / seq_len / dtype 等），修改后 ModuleGraph、数据流动画、显存估计全部**实时重算并重渲染**（< 300ms 响应），无需重新请求 HF Hub。这要求 pipeline 五阶段必须是**可增量重跑**的纯函数，且前端状态层（Zustand）与渲染层（R3F）解耦到位。
     
     **架构约束（单一计算源）**：pipeline 只在**后端 Python** 实现一次，**禁止**在前端用 TypeScript/WASM 重写一份 —— 维护两份算法会导致前后端不一致且维护负担翻倍。动态重算链路 = 前端发 `PATCH /stream/{model}/config` → 后端走常驻内存快路径（跳过 L0 之前的 HF Hub 加载）→ 通过 WebSocket 或 SSE 回推新 ModuleGraph snapshot。后端对"config-only 热更新"必须支持 < 200ms 端到端延迟（本机开发环境）。
   - **功能扩展**：前端模板（Template A/B/C/G）遵循 `TemplateContract` 接口；新增模板 = 实现接口 + 注册，不修改主渲染循环。后端 pipeline 五阶段（parse/detect/synthesize/estimate/layout）必须是纯函数且可替换，任一阶段可被独立重写而不影响其他阶段。
   - **数据契约稳定性**：ModuleGraph / DataEdge / Provenance / SSE segment schema 一旦发布即视为**公共契约**，后续变更走 `revision` 字段向前兼容，禁止破坏性修改。
   
   衡量标准：**任何一类扩展（新架构 / 新动画层 / 新并行策略 / 新显存估计器 / 新 GPU 型号）的接入成本，应 ≤ 新增 1 个文件 + 1 处注册**。若做不到，说明当前架构不合格，推倒重来。
7. **真实模型优先，不造玩具**：所有可视化必须由真实 HF 模型（meta-device + safetensors + config）驱动，禁止硬编码示意结构 / 伪造权重形状 / 用"代表性小模型"糊弄。这是与 LLM vis 的根本分水岭。
8. **架构广度有底线**：v1.0 至少正确覆盖 LLaMA 族、LLaMA-MoE、DeepSeek-MoE 三类；对不认识的架构走 Template G 通用回退并明确打 `INFERRED` 徽标，绝不默认回退成 LLaMA 骗用户。
9. **可信度可追溯**：任何呈现给用户的数字/结构/箭头，都必须能回答"这个信息来自哪儿"。Provenance（source + confidence + caveats）是强制字段，不是可选装饰。

## 项目简介

输入 HuggingFace 模型路径 → 自动生成 3D 交互式架构可视化 + 端到端推理数据流。
覆盖 MoE/MLA/量化/多模态等现代 LLM 架构。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-project-background](01-project-background.md) | 市场空白、竞品分析、产品目标、用户画像、获客渠道 |
| [02-tech-stack](02-tech-stack.md) | 技术选型（R3F+FastAPI）、依赖版本、3D 组件映射 |
| [03-system-architecture](03-system-architecture.md) | 整体架构图、数据流、项目目录结构、部署策略 |
| [04-api-design](04-api-design.md) | REST API 端点、Pydantic Schema、缓存策略、安全规则、进度推送 |
| [05-visualization-design](05-visualization-design.md) | 2D/3D 双模式、交互设计、MoE/MLA 可视化、数据流动画、视觉规范 |
| [06-implementation-phases](06-implementation-phases.md) | Phase 1-6 分步实施计划 |
| [07-testing-and-metrics](07-testing-and-metrics.md) | 验证矩阵、成功指标、产品路线图 |
| [08-architecture-decisions](08-architecture-decisions.md) | 架构决策记录（ADR）、MVP 路径、性能预算、GPU 检测、FPS 自适应 |
| **[09-backend-detailed-design](09-backend-detailed-design.md)** | **后端 API 模块详细设计（8 章：背景→需求→设计目标→术语→方案设计→存储设计→接口设计→风险与待办）** |
| **[11-extension-points](11-extension-points.md)** | **扩展点契约（Adapter/Template/AnimationLayer/ParallelismStrategy/MemoryEstimator/GPU Catalog/DataFlowDirection/ConfigEdit/Pipeline）—— 原则 6 的落地单一事实源** |
| [附录 A: 标杆产品分析](appendix-a-benchmark-analysis.md) | Netron/Model Explorer/Transformer Explainer/LLM Viz 源码级分析 |
| [附录 B: 并行策略设计](appendix-b-parallel-visualization.md) | v2.0 TP/PP/DP/EP/CP/SP 可视化设计 |

## 技术栈一览

| 层 | 技术 |
|---|---|
| 前端框架 | Next.js 15 (React 19) |
| 3D 渲染 | React Three Fiber + Drei + Three.js |
| 动画 | GSAP (HTML overlay) + react-spring (3D) |
| 状态管理 | Zustand |
| UI | Tailwind CSS |
| 后端 | FastAPI + transformers + torch (meta device) |
| 缓存 | cachetools (L0 内存 + L1 文件) |

## 关键架构决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 渲染框架 | R3F (非 Threlte) | 3D 生态最成熟，社区解决方案最多 |
| 后端依赖 | 保留 torch+transformers | meta-device 真实模型树 + forward() 分析 + 零重写 |
| 3D 优先级 | 3D 优先，2D v1.2 | 核心差异化功能 |
| 默认材质 | MeshStandardMaterial | Mac 集成 GPU 优化 |
| 渲染循环 | frameloop="demand" | 按需渲染，idle 零 GPU |
| 粒子数量 | 默认 3K | Mac 集成 GPU 上限 |
| 部署 | CPU-only torch + Docker 多阶段构建 | 镜像预计 1.3-2.0GB（待构建实测；组成：Python ~150MB + torch CPU ~200MB + transformers ~90MB + Node.js ~80MB + Next 产物 + 其他） |

## 目标用户

- **ML 研究员/工程师** (主要) — 快速理解架构、对比差异、论文插图
- **基础设施工程师** (主要) — 部署规划、量化评估、MoE 路由分析
- **学生/博主** (次要) — 理解 Transformer 机制、教学素材

## 性能预算（观测指标，非准入门槛）

> 按原则 5：以下为**观测指标**，Phase 0/1/2 不作为验收门槛，仅用于记录与 Phase N 优化决策参考。

| 指标 | 观测目标 |
|---|---|
| 3D FPS (Mac 集成 GPU) | ≥ 30fps |
| API P95 (L0 缓存) | < 10ms |
| API P95 (L1 缓存) | < 50ms |
| API P95 (HF Hub 冷启动) | < 3s |
| 3D 场景内存 | < 200MB |
| 首屏加载 | < 3s |

## 交互响应预算（硬约束，Phase 1 起必达）

> 按原则 5 例外条款：交互延迟属功能正确性范畴，不因"前期不做性能优化"而豁免。

| 交互 | 硬约束 |
|---|---|
| config 字段编辑 → 3D 重渲染（端到端） | < 300ms |
| 模块点选 / 悬停高亮 | < 50ms |
| 动画时间轴拖动（scrub） | < 16ms/frame（60fps） |
| 视角切换 / 相机动画 | < 16ms/frame |
| 后端 config-only 热更新（PATCH /config） | < 200ms |

---
