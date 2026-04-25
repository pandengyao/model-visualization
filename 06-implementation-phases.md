# 六、实现阶段

> [HF Model Visualizer](README.md) 技术设计文档 — 章六
>
> **产品原则对齐声明**：本章严格对齐 [README 产品原则 1–9](README.md#产品原则北极星不可妥协)。
> 与 README / 11-extension-points / 08-architecture-decisions 冲突时，优先级为：
> **README 产品原则 > 11 扩展契约 > 09 后端详细设计 > 08 ADR > 本文档**。
>
> **Phase 结构（对齐 08 §MVP 增量路径）**：
> `Phase 0 Tracer Bullet` → `Phase 1 v1.0 正式开发（1a/1b/1c/1d）` → `Phase 2 扩展演练` → `Phase N 性能优化`。
>
> **Phase 0/1/2 不设性能门槛**（原则 5 / ADR-012）；**交互硬约束**（PATCH <300ms 端到端、点选 <50ms、scrub <16ms/frame）属功能正确性范畴，自 Phase 1 起必达。
>
> **v1.0 范围冻结**（2026-04-25）：以下能力**不在** v1.0 交付范围，不在本章任一 Phase 的 DoD 出现：
> - 并行策略可视化（TP/PP/DP/EP/CP/SP）→ v1.1+（ADR-009/ADR-020）
> - 训练数据流与反向图 → v1.1+（ADR-018 仅预留 schema）
> - Megatron / FSDP 显存估算 → v1.1+（ADR-019）
> - 源码 AST 解析 → v1.1+
> - Stage-2 细节动画（脉动/膨胀/螺旋/RoPE 旋转/注意力热力图/token residual）→ v1.1+（ADR-017）
> - 2D SVG 模式（Netron / Model Explorer 风格）→ v1.1+
> - 模型对比分屏 → v1.1+

---

## Phase 0 — Tracer Bullet（端到端最小验证）

> **目的**（原则 5 / 原则 6）：用 **Qwen2-0.5B** 做最小端到端链路打通，验证**技术假设**与**扩展点契约是否自洽**——而非验证性能。
>
> **核心技术假设**：
> 1. meta-device 加载真实模型（Qwen2-0.5B）耗时与内存在可接受范围（仅记录，不设阈值）
> 2. safetensors header 解析可提供权重 shape 与分片信息（11 §9 S1 阶段输入）
> 3. Pydantic 能完整表达 ModuleGraph / DataEdge / Provenance 契约（11 §1/§7/§9）
> 4. SSE（冷启动 revision=1）与 WebSocket（热更新 revision=N+1）协议可打通
> 5. R3F + Drei 在 Mac 集成 GPU 上能完成基础渲染
> 6. `PATCH /config` 热更新链路在**功能上**跑通（延迟仅记录）

### Phase 0 任务 checklist

- [ ] **P0-01** Qwen2-0.5B meta-device 加载 + ModuleGraph 构建（v1.0 Block 粒度；Op/Tensor 槽位保留但不填充）
- [ ] **P0-02** Config + safetensors header 解析（对齐 11 §9 S1 parse_structure）
- [ ] **P0-03** 最简 Adapter 注册表：仅注册 `LlamaAdapter`（1 个）以驱动 Qwen2-0.5B
- [ ] **P0-04** 最简 Template 注册表：仅注册 `Template A` 一个，用于 Qwen2-0.5B 渲染
- [ ] **P0-05** SSE 冷启动链路：`GET /api/v1/stream/{org}/{repo}` 推送 `revision=1` 首帧 snapshot
- [ ] **P0-06** WebSocket 热更新链路：`PATCH /api/v1/stream/{org}/{repo}/config` 触发 → WS 推送 `revision=N+1` snapshot（仅验证链路通达）
- [ ] **P0-07** 前端 R3F 最简渲染：30 个 Block 方块（Qwen2-0.5B 有 24 层）+ Drei `<OrbitControls>` + `<Environment preset="studio">`（材质保持精美，不降级）
- [ ] **P0-08** 前端 Zustand 订阅 SSE（冷启动）与 WS（热更新），状态只管低频交互字段（对齐 ADR-010）
- [ ] **P0-09** Provenance 在 schema（后端 Pydantic 必填）与前端节点/面板（EXACT/INFERRED/ESTIMATED 徽标）双端均显示
- [ ] **P0-10** 交互预算观测 + 告警：
  - 记录 `PATCH /config` 后端处理延迟、端到端延迟、点选响应、scrub 帧时（仅记录，不作门槛）
  - **告警阈值**：若 Qwen2-0.5B 的 `PATCH /config` 后端延迟 ≥ 150ms 或端到端 ≥ 250ms（即接近 Phase 1 硬约束的 75%），必须在 Phase 0 验收报告中单列**风险项**，并附带初步瓶颈定位（detect/synthesize/estimate/layout 哪一阶段占大头）与优化建议
  - **DeepSeek-V3 级延迟估算**：Phase 0 收官前，基于 Qwen2-0.5B 实测 + 四阶段复杂度外推，给出 DeepSeek-V3（671B, 256 experts）的 `PATCH` 预期延迟；若外推值 > 200ms，Phase 1 启动前需讨论：调整预算值 / 增加异步预计算 / 降级为「仅后端 <200ms」放宽端到端
- [ ] **P0-11** Docker 镜像 build（多阶段：Python backend + Next.js frontend），本地 `docker run` 通过，浏览器访问可看到 Qwen2-0.5B 3D 渲染
  - **必须记录实际镜像大小**并回填 README / 03-system-architecture.md 的「预计 1.3–2.0GB」占位
  - **告警阈值**：若实际镜像 > 2.5GB，Phase 0 验收报告必须讨论镜像拆分策略（web/api 分离 / slim torch wheel / 移除 tokenizers 多语言资源）
- [ ] **P0-12** CI 管线最小集：前端 `pnpm lint && pnpm typecheck && pnpm test:unit`；后端 `ruff check . && mypy src/ && pytest`（对齐 02 §2.8；Phase 0 不强制覆盖率阈值，Phase 1 起强制 ≥80%）

### Phase 0 验收（DoD）

- 上述 **12 条 checklist 全部勾选** 即通过；
- **不设性能门槛**（FPS / 冷启动时延 / 内存均仅记录）；
- **扩展契约自洽性**：Adapter 注册表从 1 扩到 N 无需改核心 pipeline（为 Phase 1 与 Phase 2 做铺垫）；
- Provenance 全字段覆盖已产出的全部节点与边。

---

## Phase 1 — v1.0 正式开发（按 11-extension-points 契约全量落地）

> **目的**：按 [11 扩展契约](11-extension-points.md) 落地全部扩展点 + v1.0 交付内容（README §v1.0 必交付）。
>
> 拆分为 **1a 后端 / 1b 前端 / 1c 动画 / 1d 交互** 四个子阶段，可部分并行。
>
> 示例模型基线：**Qwen2-0.5B**（P0 遗产） / **Llama-3-8B** / **Mixtral-8x7B** / **DeepSeek-V3** / **GPT-2**（Template G 回退验证）。

### Phase 1a — 后端（扩展契约 + Pipeline 五阶段 + 热更新）

- [ ] **P1a-01** `ArchitectureAdapter` Protocol 接口（11 §1.1）+ 显式 Registry（ADR-014；禁止 entry_points / pluggy）
- [ ] **P1a-02** 实现 4 个 adapter：`LlamaAdapter` / `LlamaMoEAdapter` / `DeepseekMoEAdapter` / `GenericAdapter (Template G)`；覆盖 LLaMA 族（Llama-3-8B / Qwen2）/ LLaMA-MoE（Mixtral-8x7B / Qwen2-MoE）/ DeepSeek-MoE（DeepSeek-V3）/ 未知架构（GPT-2 走 G）
- [ ] **P1a-03** Pipeline 五阶段纯函数化（11 §9）：S1 parse_structure / S2 detect_features / S3 synthesize_flows / S4 estimate_resources / S5 compute_layout；每阶段无全局状态、无隐式 I/O
- [ ] **P1a-04** `MemoryEstimator` Protocol + Registry + 唯一实现 `InferenceMemoryEstimator`（weights + KV cache + activations；ADR-019）
- [ ] **P1a-05** GPU Catalog YAML（`backend/data/gpu-catalog.yaml`）：A100-40G / A100-80G / H100-80G / H200-141G / 4090-24G / 3090-24G / L40S-48G / 昇腾 910B / 昆仑芯 P800（唯一数据源，严禁代码硬编码；11 §6）
- [ ] **P1a-06** `ParallelismStrategy` Protocol + **空 Registry**（ADR-020；v1.0 无任何策略实现；仅加空注册表单测防接口漂移）
- [ ] **P1a-07** `PATCH /api/v1/stream/{org}/{repo}/config` 热更新路由（11 §8.2；后端 <200ms 预算）
- [ ] **P1a-08** WebSocket endpoint（增量推送 ModuleGraph snapshot，`revision` 字段单调递增；对齐 ADR-018 schema 向前兼容）
- [ ] **P1a-09** Provenance 全字段强制（ADR-016）：`ModuleNode` / `DataEdge` / `ArchitectureProfile` / `MemoryBreakdown` / `EstimateResult` 均携带 `{source, confidence ∈ {EXACT,INFERRED,ESTIMATED,UNKNOWN}, caveats}`；HTTP 响应头 `X-Provenance-Summary`
- [ ] **P1a-10** 错误契约统一为 RFC 7807 `application/problem+json`
- [ ] **P1a-11** 模板选择算法（整合自原 v1.0 §7b）：含 `_matches_llama_family()`（**RoPE + RMSNorm + SwiGLU/GatedMLP** 三特征齐备才走 Template A）；未识别架构 → **Template G**，**禁止默认回退 A**（ADR-015）

  ```python
  def select_template(config: dict) -> str:
      model_type = config.get("model_type", "")
      architectures = config.get("architectures", [])

      # 0. VLM / Encoder-Decoder 优先检测（v1.0 返回 "Unsupported"）
      if hasattr_or_key(config, "vision_config"):
          return "Unsupported"   # v1.1+ 启用 VLM 模板
      for arch in architectures:
          if "ForSeq2SeqLM" in arch or "ForConditionalGeneration" in arch:
              return "Unsupported"

      # 1. 直接匹配 model_type
      if model_type in TEMPLATE_MAP:
          return TEMPLATE_MAP[model_type]

      # 2. architectures 回退
      for arch in architectures:
          if "DeepseekV3" in arch or "DeepseekV2" in arch:
              return "C"
          if "ForCausalLM" in arch:
              if config.get("n_routed_experts") or config.get("num_local_experts"):
                  if config.get("q_lora_rank"):
                      return "C"   # MLA + MoE
                  return "B"       # LLaMA-MoE
              if _matches_llama_family(config):
                  return "A"

      # 3. 特征指纹
      if config.get("q_lora_rank") and config.get("kv_lora_rank"):
          return "C"

      # 4. 未识别 → Template G（通用回退；禁止默认回退 A）
      return "G"


  def _matches_llama_family(config: dict) -> bool:
      """LLaMA 族强特征：RoPE + RMSNorm + SwiGLU/GatedMLP 三者齐备"""
      pos = (config.get("rope_type")
             or ("rope" if config.get("rope_theta") else None)
             or config.get("position_embedding_type"))
      if pos in ("absolute", "alibi", "learned", None):
          return False
      norm_ok = (config.get("rms_norm_eps") is not None
                 or config.get("layer_norm_type") == "rms")
      if not norm_ok:
          return False
      act = (config.get("hidden_act") or config.get("activation_function") or "").lower()
      # 注意：仅接受 SwiGLU/GatedMLP 家族激活；
      # gelu_pytorch_tanh 属于 GELU 变体，不在此列（避免 LLaMA 族误判）。
      if act not in ("silu", "swiglu", "geglu"):
          return False
      return True
  ```

### Phase 1b — 前端（TemplateContract + ConfigEditor + GPUSelector + Provenance）

- [ ] **P1b-01** `TemplateContract` 接口（11 §2）+ Registry；实现 Template **A / B / C / G**（覆盖 Llama-3-8B / Mixtral-8x7B / DeepSeek-V3 / GPT-2 未知回退）
- [ ] **P1b-02** R3F `<Scene>` 主组件：`<Canvas>` + `<OrbitControls damping>` + `<Environment>` + `<EffectComposer><Bloom/></EffectComposer>`；材质默认 `MeshStandardMaterial`（ADR-003），不因性能主动降级
- [ ] **P1b-03** `<ConfigEditor>` 浮动面板（受控组件，白名单字段见 11 §8.1）：300ms debounce + `PATCH /config` 发送 + 等待 WS `revision+=1` snapshot
- [ ] **P1b-04** `<GPUSelector>` 下拉菜单：从 `/api/v1/gpus` 拉取 yaml 内容（11 §6.4），前端无硬编码 GPU 列表
- [ ] **P1b-05** Provenance 徽标组件：EXACT（绿）/ INFERRED（蓝 `#3b82f6`，空心圆 + ⓘ tooltip）/ ESTIMATED（黄）在节点与侧边面板均可见（ADR-015/ADR-016；配色以 05 §视觉规范为准）
- [ ] **P1b-06** 交互硬约束达标（ADR-012；非性能优化，属功能正确性）：
  - PATCH 端到端 **< 300ms**
  - 模块点选 / 悬停 **< 50ms**
  - 时间轴 scrub **< 16ms/frame**
  - 视角切换 / 相机 **< 16ms/frame**

### Phase 1c — 动画（AnimationLayer 插件化 + L1/L2 最小子集）

- [ ] **P1c-01** `AnimationLayer` 接口（11 §3.2）+ Registry（L1 / L2 / L3 / L4 均注册；L3/L4 为空实现占位，对应 v1.1+/v1.2）
- [ ] **P1c-02** **L1 StructureAnimation**：模块展开 / 收起 / 层级过渡（v1.0 必交）
- [ ] **P1c-03** **L2 DataFlowAnimation 最小子集**（ADR-017）：① Attention Q/K/V 分解 ② MoE 路由（router → top-k → experts → 加权合并） ③ Residual flow
- [ ] **P1c-04** 四层独立开关 UI + 统一时间轴（层间不得硬编码相互依赖，通过 `AnimationContext` 共享时间轴；11 §3.3）

### Phase 1d — 交互（硬约束达标的 UI 细节）

- [ ] **P1d-01** 模块点选 + 悬停高亮（<50ms）
- [ ] **P1d-02** 相机 OrbitControls + damping（<16ms/frame）
- [ ] **P1d-03** 时间轴拖动 scrub（<16ms/frame）
- [ ] **P1d-04** 信息面板滑入（节点详情 / 参数 / Provenance caveats）
- [ ] **P1d-05** GSAP 微交互：hover scale / glow、click ripple（DOM overlay，不影响 R3F 帧率）

### Phase 1 验收（DoD）

- 上述 1a/1b/1c/1d **全部 checklist** 勾选；
- **扩展点全部就位**（Adapter / Template / AnimationLayer / ParallelismStrategy / MemoryEstimator / GPU Catalog / DataFlowDirection schema / ConfigEdit 路由）；
- **Provenance 全字段覆盖**（ADR-016）；
- **交互硬约束达标**（本节 P1b-06 四项）；
- 单元测试覆盖率 **> 80%**（后端 pipeline 五阶段 + 前端核心组件）；
- **无性能门槛**（原则 5 / ADR-012）；FPS、冷启动时延、内存仅作观测记录。

---

## Phase 2 — 扩展演练（≤ 1 文件 + 1 注册）

> **目的**（原则 6 衡量标准）：选一个 **Phase 1 未覆盖且与 LLaMA 族特征不重合** 的架构，做 cold-start 接入，验证扩展成本。若做不到 **≤ 1 adapter 文件 + 1 注册**（+ 至多 ≤ 1 template 文件 + 1 注册），视为架构违规，**推倒重来**。
>
> **候选模型**（择一）：
> - **Mamba-1.4B**（`state-spaces/mamba-1.4b`）—— 纯 SSM，无 attention，无 causal mask
> - **ViT-B-16**（`google/vit-base-patch16-224`）—— 无因果掩码，有 patch embedding
>
> 任选其一即可（建议 Mamba，架构落差更大，验证更彻底）。

### Phase 2 任务 checklist

- [ ] **P2-01** 选定 Mamba-1.4B 或 ViT-B-16（选一个 Phase 1 注册表里不存在、开发者不熟的架构）
- [ ] **P2-02** 新增 **1 个** adapter 文件（如 `backend/adapters/mamba.py` 或 `backend/adapters/vit.py`）
- [ ] **P2-03** `backend/adapters/__init__.py` 追加 **1 行** `from . import <name>`（触发模块级 `@register` 装饰器）
- [ ] **P2-04** 若前端需新 Template（如 Mamba 的 SSM 布局），新增 **1 个** template 文件（如 `frontend/src/templates/mamba.tsx`）
- [ ] **P2-05** `frontend/src/templates/registry.ts` 追加 **1 行**
- [ ] **P2-06** 严格验证：除 P2-02/P2-03/P2-04/P2-05 涉及文件外，**不得修改任何核心文件**（pipeline / 渲染循环 / 路由层 / 既有 adapter / 既有 template）
- [ ] **P2-07** 视觉 QA：渲染正确，Provenance 徽标正确（Mamba 走新 adapter 时 confidence=EXACT；若退到 Template G 则 INFERRED）
- [ ] **P2-08** 扩展演练报告（记录新增文件数、新增注册行数、修改既有文件数 = 0）

### Phase 2 验收（DoD）

- **硬门槛**：新增 ≤ 1 adapter 文件 + 1 注册行 （+ 可选 ≤ 1 template 文件 + 1 注册行）；
- **零核心修改**：pipeline 五阶段、渲染主循环、路由层、既有 adapter / template 均未被触碰；
- **零 `if model_type == "..."` 分支**（11 §1.3 硬性）；
- 单元测试覆盖率 **> 80%**；Provenance 全字段覆盖；
- **无性能门槛**（原则 5）；交互硬约束继续达标；
- **若任一未达 → 视为架构违规 → 回退并重构（推倒重来）**。

---

## Phase N — 性能优化集中阶段（延后执行，Phase 0/1/2 完成后启动）

> **目的**（原则 5 / ADR-012）：Phase 0/1/2 的正确性 + 教学性验收通过后，统一启动性能优化。**Phase 0/1/2 观测到的指标作为 Phase N 的决策输入**。
>
> 目标：追上 [README § 性能预算](README.md#性能预算观测指标非准入门槛) 的观测目标（3D FPS ≥ 30fps / API P95 / 首屏 < 3s / 3D 场景内存 < 200MB）。

### Phase N 任务清单（简述）

- [ ] **PN-01** 3D LOD 系统：相机距离降级（远距离合并层块、隐藏文字标签）
- [ ] **PN-02** `frameloop="demand"` 精细调度：渲染模式状态机 `static / interactive / animated`（ADR-006）
- [ ] **PN-03** Bundle splitting / lazy loading：Template B/C/G 按需加载，Three.js / R3F 分块
- [ ] **PN-04** L0 序列化缓存（orjson bytes 缓存） + L1 文件缓存调优（与 ADR-021 一致）
- [ ] **PN-05** GPU 能力检测 / FPS 自适应策略（ADR / 08 §Phase N 专项：高端 → MeshPhysicalMaterial + 10K 粒子；低端 → MeshBasicMaterial + 1K 粒子）

### Phase N 验收（DoD）

- 观测指标达到或接近 README §性能预算目标（非硬门槛，但为 Phase N 的显式交付目标）；
- Phase 0/1/2 的交互硬约束**继续保持**（不得因 LOD / 降级引入点选或 scrub 延迟回退）；
- 单元测试覆盖率继续 > 80%；Provenance 全字段覆盖不受影响。

---

## 统一 DoD（Definition of Done，适用于所有 Phase）

任一 Phase 的"完成"必须同时满足下列条款：

1. **单元测试覆盖率 > 80%**（后端 pipeline 五阶段 + 前端核心组件）
2. **扩展点接入成本验证**：Phase 2 为显式验证（≤ 1 文件 + 1 注册）；Phase 1 为隐式（4 个 adapter 都满足此成本）
3. **Provenance 全字段覆盖**（ADR-016）：所有对外 schema 携带 `{source, confidence, caveats}`；HTTP 响应头携带 `X-Provenance-Summary`
4. **无性能门槛**（原则 5 / ADR-012）：FPS、冷启动时延、内存仅作观测记录，不作发布门禁
5. **交互硬约束达标**（原则 5 例外条款 / ADR-012；从 Phase 1 起）：
   - PATCH 端到端 **< 300ms**
   - 模块点选 **< 50ms**
   - 动画 scrub **< 16ms/frame**
   - 视角切换 **< 16ms/frame**
   - 后端 PATCH /config 热更新 **< 200ms**

---

[← 返回目录](README.md)
