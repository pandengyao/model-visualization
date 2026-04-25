# 架构决策记录 (ADR) 与性能预算

> [HF Model Visualizer](README.md) 技术设计文档 — 架构决策 + 性能指标
>
> **产品原则对齐声明**：本文档严格对齐 [README 产品原则 1–9](README.md#产品原则北极星不可妥协)。
> 与 README / 11-extension-points / 09-backend-detailed-design 冲突时，优先级为：
> **README 产品原则 > 11 扩展契约 > 09 后端详细设计 > 本文档**。

---

## MVP 增量路径（与 06 保持一致）

```
Phase 0 — Tracer Bullet
  最小端到端骨架：HF 路径 → 后端 pipeline 五阶段 → ModuleGraph → 前端 Template A 静态 3D
  （正确性优先，不做任何性能优化）
  ↓
Phase 1 — Formal Engineering（v1.0 主体）
  • 后端：ArchitectureAdapter 注册表（LLaMA / LLaMA-MoE / DeepSeek-MoE + Template G 回退）
  • 后端：pipeline 五阶段纯函数化 + Provenance 强制字段
  • 后端：InferenceMemoryEstimator + GPU Catalog（8 款）
  • 后端：PATCH /config 动态编辑 + WS/SSE 增量推送
  • 前端：Template A/B/C/G 实现 + ConfigEditor + GPU 选择器
  • 前端：Stage-1 结构动画 + Stage-2 最小子集（Attention Q/K/V、MoE 路由、Residual flow）
  • 交互响应硬约束全部满足
  ↓
Phase 2 — 扩展演练与契约固化
  • 用 Mamba 或 ViT 做 cold-start 接入验证（≤ 1 文件 + 1 注册）
  • 反向接口（DataFlowDirection.backward）schema 预留但不启用
  ↓
Phase N — 性能优化集中阶段（正确性/教学性验收通过后启动）
  LOD 降级、frameloop 精细调度、粒子动态伸缩、FPS 自适应、GPU 能力检测
  （Phase 0/1/2 不做；详见下文「观测指标」与「Phase N 专项」）
  ↓
v1.1+ 规划（超出本文档范围，见 README）
  Stage-2 其余动画 / Megatron & FSDP 估计 / 并行策略 / 反向传播动画 / 2D SVG / 模型对比
```

---

## 关键架构决策

### ADR-001: 后端依赖 — 保留 torch + transformers

- **决策**: 保留 torch + transformers 作为后端依赖
- **理由**: meta-device 加载真实模型树 + forward() 分析 + 零重写（对齐原则 7 真实模型优先）
- **替代方案**: 仅使用 huggingface_hub 轻量下载 config，放弃真实模型树（拒绝：违背原则 7）
- **状态**: 已采纳

### ADR-002: 3D vs 2D 优先级 — 3D 优先

- **决策**: 3D 优先开发，2D 移至 v1.1+
- **理由**: 核心差异化功能（对齐原则 2 精美 3D），对标 LLM Viz / Transformer Explainer
- **替代方案**: 先实现 2D SVG 可视化，后补 3D
- **状态**: 已采纳

### ADR-003: 默认 3D 材质 — MeshStandardMaterial

- **决策**: 默认使用 MeshStandardMaterial
- **理由**: 在保证精美质感的前提下与主流设备兼容；MeshPhysicalMaterial 可在 Phase N 作为高端档位升级项
- **替代方案**: MeshPhysicalMaterial（更真实但性能开销大）
- **状态**: 已采纳
- **备注（原则 5）**：Phase 0/1/2 不因性能理由主动降级材质；若观测到交互响应受影响再评估

### ADR-004: 后处理 — Bloom + SSAO + ACES（v1.0 必需）

- **决策**: v1.0 EffectComposer 固定三件套 `Bloom + SSAO + ACESFilmicToneMapping`；`ChromaticAberration` 为可选（默认极轻微强度，可关闭）
  - Bloom: threshold 0.9 / intensity 0.4 / radius 0.8
  - SSAO: intensity 1.2
  - ToneMapping: ACESFilmic
  - ChromaticAberration: optional（默认启用但极轻微）
- **理由**: 对齐原则 2「精美 3D 视觉质感」硬约束；与 05-visualization-design.md §6.4 保持单一事实源
- **替代方案**: 仅 Bloom（视觉质感不足，已否决）；全效果链 + Vignette/Noise（Phase N 候选，不在 v1.0）
- **状态**: 已采纳
- **备注**：
  - 禁止因性能理由在 v1.0 主动关闭 Bloom/SSAO/ACES（对齐原则 5「前期不做性能优化」）
  - Phase N 若交互响应受影响，优先降低 Bloom 分辨率或 SSAO 采样数，而非整体关闭
  - `ChromaticAberration` 可通过 config 开关关闭（非强制，不违反硬约束）

### ADR-005: 粒子数量 — 默认 3K

- **决策**: 默认粒子上限 3K
- **理由**: 在主流设备上平衡视觉丰富度与动画精细度；高端设备 Phase N 可提升至 10K
- **替代方案**: 统一 10K（低端设备体验受影响）
- **状态**: 已采纳
- **备注（原则 5）**：Phase 0/1/2 不因性能理由主动砍粒子数

### ADR-006: 渲染循环 — 渲染模式状态机

- **决策**: 三模式状态机 `renderMode: 'static' | 'interactive' | 'animated'`
  - `static`（默认）: `frameloop="demand"`，idle 零 GPU 负载
  - `interactive`: 用户拖拽/hover 时切为 `always`，操作结束 2s 回退 `demand`
  - `animated`: Tour/数据流播放时持续 `always`
- **理由**: 解决 `frameloop="demand"` 与持续动画的矛盾；本决策主要服务于"交互响应延迟硬约束"，而非节流式性能优化
- **替代方案**: 全局 `always` / 全局 `demand` + 手动 invalidate
- **状态**: 已采纳

### ADR-007: API 响应树结构 — ~~嵌套递归 TreeNode~~ → ModuleGraph（已废弃/取代）

- **决策（2026-04-25 更新）**: **废弃 TreeNode**，采用 `ModuleGraph` = `{nodes: dict[id, ModuleNode], edges: list[DataEdge], hierarchy: HierarchyTree}`（DAG + 平面 nodes/edges + 独立层级树），详见 [09 §5.1.2](09-backend-detailed-design.md)
- **理由**: TreeNode 仅能表达纯树；实际模型存在 residual / cross-attn / branch-merge（VLM、多模态）等 DAG 关系，违背原则 3「结构 100% 正确」。扁平 nodes+edges 便于 O(1) 查询、2D/3D 双模式渲染与 Provenance 字段化携带
- **取代关系**: ModuleGraph + Provenance 完整取代 TreeNode
- **状态**: 已采纳（TreeNode 从此禁止重新引入）

### ADR-008: 参数统计 — ~~ParamStats 双形态~~ → ModuleNode.params + EstimateResult（已废弃/取代）

- **决策（2026-04-25 更新）**: **废弃 ParamStats 双形态**。`ModuleNode.params` 存精确值（meta-device 得到）；估算值统一走 `EstimateResult`，带 `Provenance.confidence=ESTIMATED`。前端通过 `provenance.confidence` 判定展示徽标，不再依赖 `is_estimated` 标志位
- **理由**: Provenance 机制（source + confidence + caveats）信息量严格覆盖并大于 ParamStats 双形态（对齐原则 9）
- **取代关系**: ModuleGraph + Provenance 完整取代 ParamStats
- **状态**: 已采纳（ParamStats 双形态从此禁止重新引入）

### ADR-009: 并行策略 — v1.1+

- **决策**: 并行策略可视化（TP/PP/DP/EP/CP/SP + 组合）移至 v1.1+；v1.0 仅保留 `ParallelismStrategy` Protocol 接口与空注册表（见 ADR-020）
- **理由**: 实现复杂度极高，v1.0 聚焦架构可视化与 Stage-2 最小子集；但接口必须在 v1.0 就位以满足原则 6
- **状态**: 已采纳

### ADR-010: Zustand 状态范围 — 仅低频交互状态

- **决策**: Zustand 仅管理低频交互状态（选中项、UI 面板开关、config overrides 等）
- **理由**: 帧级数据走 useFrame + ref，避免 React 重渲染；服务于交互响应硬约束（scrub < 16ms/frame）
- **替代方案**: 所有状态统一 Zustand 管理
- **状态**: 已采纳

---

### ADR-011: 非商业化定位 → 架构简化（对齐原则 1）

- **上下文**: 内部工具，团队用户 < 10 人，不公开注册、无 SLA 对外承诺
- **决策**: 不引入鉴权 SSO、多租户、配额系统、公开 API 限流、K8s 强制化、Kafka / Redis / ClickHouse / 消息队列 / 多区域部署 / A/B 实验平台 / 灰度系统等"企业级"组件
- **替代方案**: 预置企业级基础设施以"未来可能用得上"（拒绝：违背原则 1 与"拒绝过度工程"边界声明）
- **后果**:
  - 单 Docker 镜像即可部署，架构保持小巧
  - 缓存采用 L0 内存 + L1 文件（见 ADR-021），不引入 Redis
  - 若未来商业化需重新评估本决策，届时视为破坏性变更，整体架构重审
- **状态**: 已采纳

### ADR-012: 前期不做性能优化（对齐原则 5）

- **上下文**: 功能正确性与教学深度优先；性能优化在 Phase 0/1/2 会与正确性/教学性争夺工时
- **决策**: Phase 0 / Phase 1 / Phase 2 **不做** LOD 降级、不做 frameloop 限制、不砍动画细节、不降模型规模、不做 FPS 自适应降级；性能指标仅作**观测**记录
- **例外（硬约束，不豁免）**: **交互响应延迟**属于功能正确性范畴，必须从 Phase 1 起即满足：
  - PATCH /config 后端热更新 **< 200ms**
  - PATCH /config 端到端（含 WS 往返 + 前端重渲染） **< 300ms**
  - 模块点选 / 悬停高亮 **< 50ms**
  - 动画时间轴拖动 scrub **< 16ms/frame（60fps）**
  - 视角切换 / 相机动画 **< 16ms/frame**
- **替代方案**: 从 Phase 1 起把性能指标作为发布门禁（拒绝：违背原则 5）
- **后果**: Phase N 会有显著的性能优化集中阶段；观测指标用于 Phase N 决策输入
- **状态**: 已采纳

### ADR-013: 单一计算源（禁止前后端双 pipeline）（对齐原则 6）

- **上下文**: PATCH /config 动态重算要求 < 300ms 端到端响应，理论上可在前端（TypeScript / WASM）跑一份 pipeline 以省去网络往返
- **决策**: pipeline 只在**后端 Python** 实现一次，前端通过 WebSocket/SSE 获取增量 ModuleGraph snapshot；禁止前端重写 estimate / layout / synthesize 等任一阶段
- **替代方案**:
  - A) 前端 TS 重写 pipeline → **拒绝**，双份算法维护负担翻倍，且易出现前后端不一致（违背原则 3）
  - C) 将 Python pipeline 通过 Pyodide / WASM 打包进前端 → **拒绝**，bundle 体积膨胀 + 首屏加载时间不可接受（与原则 2 现代化前端要求冲突）
- **后果**:
  - 后端 config-only 热更新必须 < 200ms（硬约束）
  - pipeline 五阶段必须是纯函数且无隐式 I/O（见 11 §9）
  - WS/SSE 增量协议（segmented-data, revision 字段）是 v1.0 公共契约
- **状态**: 已采纳

### ADR-014: 插件化扩展点（对齐原则 6）

- **上下文**: 原则 6 要求 v1.0 架构最健壮可扩展，而扩展点的**实现机制**本身也是一个决策
- **决策**: 5 类扩展点（**ArchitectureAdapter / TemplateContract / AnimationLayer / ParallelismStrategy / MemoryEstimator**）均采用「**Protocol 接口 + 显式注册表**」模式；**禁止**使用 `entry_points` / `pluggy` / 文件系统扫描等隐式自动发现机制
- **衡量标准**: 任一扩展接入成本 **≤ 1 个新文件 + 1 处注册**，违反即架构违规（见 11 §10 Checklist）
- **替代方案**: pluggy / setuptools entry_points 自动发现（拒绝：增加运行时不可预测性，且与"显式优于隐式"冲突）
- **后果**:
  - v1.0 必须完成**扩展演练**：在 Phase 2 用 Mamba 或 ViT 做 cold-start 接入验证
  - 扩展契约单一事实源见 [11-extension-points.md](11-extension-points.md)
- **状态**: 已采纳

### ADR-015: Template G 通用回退 — 拒绝默认回退 LLaMA（对齐原则 3 & 8）

- **上下文**: 未知架构若默认回退 Template A（LLaMA），会将 GPT-2 / GPT-J / Falcon / Phi / Mamba / RWKV / StarCoder 等错误地展示为 LLaMA 结构，严重违背原则 3「结构 100% 正确」
- **决策**:
  - 未识别架构 → **Template G（通用回退）**：仅展示 config 已知字段 + 通用 Decoder 骨架 + 在节点与面板显著位置打 `INFERRED` 醒目徽标
  - 只有通过强特征检测 `_matches_llama_family()`（**RoPE + RMSNorm + SwiGLU/GatedMLP** 三特征必须齐备）才会走 Template A
- **替代方案**: 基于 `*ForCausalLM` 后缀默认回退 LLaMA（已废弃：误分类率高）
- **后果**:
  - v1.0 必须交付 Template G 实现（前端 + 后端 adapter）
  - Provenance.confidence=INFERRED 路径必须在 UI 层明确可见
- **状态**: 已采纳

### ADR-016: Provenance 强制字段（对齐原则 9）

- **上下文**: 原则 9 要求任何呈现给用户的数字/结构/箭头都必须可追溯
- **决策**:
  - `ModuleNode` / `DataEdge` / `ArchitectureProfile` / `MemoryBreakdown` / `EstimateResult` 等**所有**对外 schema 必须携带 `Provenance = {source, confidence, caveats}`
  - `confidence` 枚举统一为 **3 档**：`EXACT` / `INFERRED` / `ESTIMATED`（对齐 04/09/10；按原则 1 简洁性删除 `UNKNOWN`，"不确定/缺失" 语义统一通过 `caveats: list[str]` 表达，避免枚举与文本双轨）
  - HTTP 响应头必须携带 `X-Provenance-Summary: exact=<n>,inferred=<n>,estimated=<n>`
- **替代方案**: Provenance 作为可选装饰（拒绝：违背原则 9，且与 ADR-008 取代关系冲突）
- **后果**:
  - 后端 Pydantic 序列化负担增加（轻微）
  - 前端必须实现徽标/caveats 展示组件
  - 单元测试：每个 adapter 必须断言其产出节点 100% 携带 Provenance
- **状态**: 已采纳

### ADR-017: v1.0 Stage-2 动画最小子集（对齐原则 4）

- **上下文**: 原则 4 要求教学深度与动画精细度大幅超越竞品；但 v1.0 不能把所有 Stage-2 动画都做完，否则过载
- **决策**: v1.0 拉回 **3 项 Stage-2 动画**：
  1. **Attention Q/K/V 分解**（展示线性投影 + 分头 + dot-product + softmax + value 加权）
  2. **MoE 路由**（router 打分 → top-k 选择 → experts 并发 → 结果加权合并）
  3. **Residual flow**（主干 + 残差旁路 + 加法合并）
- **v1.1+ 交付**: 脉动 / 膨胀 / 螺旋 / 热力图 / token residual / 反向传播动画
- **替代方案**: v1.0 只做 Stage-1 结构动画（拒绝：无法满足原则 4 的"超越竞品"要求）
- **后果**: v1.0 动画精细度达到或超越 LLM Viz (bbycroft.net) 的基础水平
- **状态**: 已采纳

### ADR-018: DataFlowDirection 一等公民（反向接口预留）（对齐原则 6）

- **上下文**: 原则 6 要求 v1.0 预留反向传播扩展点
- **决策**:
  - `DataEdge.edge_type` 枚举**预留** `gradient_flow` / `activation_checkpoint` / `gradient_accumulation` 值
  - `DataEdge.direction: Literal["forward", "backward", "bidirectional"]` 作为一等公民
  - v1.0 只产出 `direction="forward"` 的边；v1.1+ 启用 backward 时直接填充现有字段，**不破坏契约**
- **替代方案**: v1.0 不预留字段，v1.1 时再扩 schema（拒绝：破坏 ADR-013 单一计算源 + 11 §0 契约向前兼容）
- **后果**: schema 提前稳定；前端渲染层提前按 `direction` 字段分支（即便 v1.0 永远命中 forward 分支）
- **状态**: 已采纳

### ADR-019: MemoryEstimator 插件化 — v1.0 推理版 + GPU Catalog（对齐原则 6）

- **上下文**: 原则 6 要求显存估计器作为扩展点
- **决策**:
  - v1.0 实现 **1 个** `InferenceMemoryEstimator`：覆盖 weights + KV cache + activations
  - v1.0 交付 `backend/data/gpu-catalog.yaml`，包含 **8 款 GPU**：A100-40G、A100-80G、H100-80G、H200-141G、4090-24G、3090-24G、L40S-48G、昇腾 910B、昆仑芯 P800（注：9 款名称列举以覆盖国产卡底线；最终以 yaml 为准）
  - v1.1+ 扩展：`MegatronEstimator`（TP+PP+SP）+ `FSDPEstimator`（ZeRO-1/2/3）
- **替代方案**: v1.0 直接做 Megatron 估计器（拒绝：实现复杂度与 v1.0 目标不匹配）
- **后果**:
  - GPU 选择器 UI 必须在 v1.0 就位，即使估计结果精度较粗糙（Provenance.confidence=ESTIMATED + caveats 说明局限）
  - GPU Catalog 作为唯一数据源，严禁代码硬编码 GPU 规格（见 11 §6）
- **状态**: 已采纳

### ADR-020: ParallelismStrategy 仅接口 — v1.0 无实现（对齐原则 6 + 拒绝过度工程）

- **上下文**: 原则 6 要求并行策略扩展点就位，但实际策略实现复杂度极高
- **决策**:
  - v1.0 定义 `ParallelismStrategy` Protocol + **空的** Registry，**不实现任何策略**
  - TP / PP / DP / EP / CP / SP 及其组合（3D/4D/5D parallelism）全部留 v1.1+
  - 前端并行策略 UI 在 v1.0 **不做**
- **替代方案**: v1.0 实现 TP 基础版（拒绝：与 ADR-009 冲突，且 v1.0 交付目标过载）
- **后果**:
  - 扩展点预留但**无用例验证**；首次扩展演练在 v1.1 首个并行策略接入时进行
  - v1.0 收尾时必须有针对该 Protocol 的空注册表单测，防止接口漂移
- **状态**: 已采纳

### ADR-021: 缓存 — L0 内存 + L1 文件（两层缓存 + 数据源）

- **决策**: 两层缓存架构（L0 进程内 LRU + L1 文件系统 JSON），HF Hub API 为数据源（非缓存层）
- **理由**: 跨重启持久化，单镜像部署即可（对齐 ADR-011）
- **替代方案**: 纯内存 TTLCache（重启丢失）或直接引入 Redis（拒绝：违背 ADR-011 非商业化定位 → 架构简化）
- **后果**: v1.1+ 如需 Redis 须与 ADR-011 一并重评
- **状态**: 已采纳（由原 ADR-011 "缓存" 重新编号，以让位给新增 ADR 序列）

---

## 观测指标（非准入门槛，对齐原则 5 与 ADR-012）

> 下列指标在 Phase 0 / Phase 1 / Phase 2 仅作**观测记录**，不作为发布门禁。达不到先记录，不阻塞发布。
> 这些数据是 Phase N 性能优化集中阶段的决策输入。

### 按渲染模式分级（观测）

| 渲染模式 | frameloop | FPS 观测目标 | 适用场景 |
|---|---|---|---|
| **static** (默认) | `"demand"` | 按需刷新 | 静态浏览 |
| **interactive** | `"always"` (自动切换) | ≥ 30fps | 拖拽/hover/缩放交互 |
| **animated** | `"always"` | ≥ 30fps | 数据流动画/Tour/粒子 |

### 具体观测指标

| 指标 | 观测目标 | 说明 |
|---|---|---|
| 3D 场景 FPS (animated 模式) | ≥ 30fps | 仅记录，不强制 |
| 3D 场景 FPS (Bloom ON vs OFF 差异) | < 5fps | Phase N 评估是否需降分辨率 |
| API 响应 (L0 缓存) | < 10ms | 观测 |
| API 响应 (L1 缓存) | < 50ms | 观测 |
| API 响应 (HF Hub 冷启动) | < 3s | 观测 |
| 3D 场景内存 | < 200MB | 观测 |
| 首屏加载 | < 3s | 观测 |

---

## 交互响应预算（硬约束，Phase 1 起必达，对齐原则 5 例外条款 & ADR-012）

> 下列指标属于**功能正确性**范畴，不受"前期不做性能优化"豁免。
> 任一未达 = 阻塞发布。

| 交互 | 硬约束 |
|---|---|
| PATCH /config 后端热更新 | **< 200ms** |
| PATCH /config 端到端（含 WS 往返 + 前端重渲染） | **< 300ms** |
| 模块点选 / 悬停高亮 | **< 50ms** |
| 动画时间轴拖动 scrub | **< 16ms/frame（60fps）** |
| 视角切换 / 相机动画 | **< 16ms/frame** |

---

## Phase N 专项（不在 v1.0 范围）

> 以下内容**不在 Phase 0/1/2 做**（对齐 ADR-012），在 Phase N 性能优化集中阶段启动。

### Bloom 后处理性能优化（Phase N）

Bloom 是 multi-pass 全屏效果，对集成 GPU 填充率消耗大。Phase N 候选优化：

| 优化项 | 候选方案 |
|---|---|
| Canvas DPR 上限 | `dpr={[1, 1.5]}`（Retina 2x → 1.5x） |
| Bloom 分辨率 | `resolutionScale={0.5}`（半分辨率 Bloom pass） |
| 合格参考 | Bloom ON/OFF 的 FPS 差异 < 5fps |

### GPU 能力检测与自动材质选择（Phase N）

通过 WebGL 扩展查询自动检测 GPU 能力，选择最优材质级别：

| 检测项 | 高端 GPU | 中端 | 低端/移动端 |
|---|---|---|---|
| `MAX_TEXTURE_SIZE` | ≥8192 | ≥4096 | <4096 |
| `WEBGL_compressed_texture_s3tc` | 有 | 可能有 | 无 |
| 材质 | MeshPhysicalMaterial | MeshStandardMaterial | MeshBasicMaterial |
| 后处理 | MeshPhysicalMaterial + 全效果链 | Bloom + SSAO + ACES（v1.0 基线） | 降分辨率 Bloom only |
| 粒子上限 | 10K | 3K | 1K |

### FPS 自适应策略（Phase N）

运行时 FPS 监控 + 自动降级候选规则：
- 连续 5 帧 < 25fps → 粒子数量减半
- 连续 10 帧 < 20fps → Bloom 半分辨率；再 5 帧未恢复 → SSAO 降采样数
- 连续 20 帧 < 15fps → 切换到 MeshBasicMaterial（保留 Bloom + ACES）
- 用户可手动覆盖自动降级（设置面板）

> 注：以上 Phase N 规则在 Phase 0/1/2 **不启用**。Phase 1 若观测到交互响应硬约束被破坏，先评估是否是交互响应相关再针对性解决，不得为规避观测指标而提前引入 LOD/降级。

---

[← 返回目录](README.md)
