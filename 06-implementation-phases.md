# 六、实现阶段

> [HF Model Visualizer](README.md) 技术设计文档 — 章六
>
> **产品原则对齐声明**：本章严格对齐 [README 产品原则 1–11](README.md#产品原则北极星不可妥协)。
> 与 README / 11-extension-points / 08-architecture-decisions 冲突时，优先级为：
> **README 产品原则 > 11 扩展契约 > 09 后端详细设计 > 08 ADR > 本文档**。
>
> **Phase 结构（对齐 08 §MVP 增量路径）**：
> `Phase 0 Tracer Bullet` → `Phase 1 v1.0 正式开发（1a/1b/1c/1d）` → `Phase 2 扩展演练` → `Phase N 性能优化`。
>
> **Phase 0/1/2 不设性能门槛**（原则 7 / ADR-012）；**交互硬约束**（PATCH 端到端 <300ms、点选 <50ms、scrub <16ms/frame、视角切换 <16ms/frame、后端 PATCH /config <200ms）属功能正确性范畴，自 Phase 1 起必达。
>
> **v1.0 范围冻结**（2026-04-25）：以下能力**不在** v1.0 交付范围，不在本章任一 Phase 的 DoD 出现：
> - 并行策略可视化（TP/PP/DP/EP/CP/SP + 2D/3D/4D/5D 组合）→ v1.2（ADR-009/ADR-020）
> - 训练数据流与反向图 → v1.1+（ADR-018 仅预留 schema）
> - Megatron / FSDP 显存估算 → v1.1+（ADR-019）
> - 源码 AST 解析 → v1.1+
> - Stage-2 细节动画（脉动/膨胀/螺旋/RoPE 旋转/注意力热力图/token residual）→ v1.1+（ADR-017）
> - 2D SVG 模式（Netron / Model Explorer 风格）→ v1.1+
> - 模型对比分屏 → v1.1+

---

## Phase 0 — Tracer Bullet（端到端最小验证）

> **目的**（原则 7 / 原则 8）：用 **Qwen2-0.5B** 做最小端到端链路打通，验证**技术假设**与**扩展点契约是否自洽**——而非验证性能。
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
- [ ] **P0-07** 前端 R3F 最简渲染：24 个 Block 方块（对齐 Qwen2-0.5B num_hidden_layers=24）+ Drei `<OrbitControls>` + `<Environment preset="studio">`（材质保持精美，不降级）
- [ ] **P0-08** 前端 Zustand 订阅 SSE（冷启动）与 WS（热更新），状态只管低频交互字段（对齐 ADR-010）
- [ ] **P0-09** Provenance 在 schema（后端 Pydantic 必填）与前端节点/面板（EXACT/INFERRED/ESTIMATED 徽标）双端均显示
- [ ] **P0-10** 交互预算观测 + 告警：
  - 记录 `PATCH /config` 后端处理延迟、端到端延迟、点选响应、scrub 帧时（仅记录，不作门槛）
  - **告警阈值**：若 Qwen2-0.5B 的 `PATCH /config` 后端延迟 ≥ 150ms 或端到端 ≥ 250ms（即接近 Phase 1 硬约束的 75%），必须在 Phase 0 验收报告中单列**风险项**，并附带初步瓶颈定位（detect/synthesize/estimate/layout 哪一阶段占大头）
  - **决议产物要求**：Phase 0 仅产出观测数据；DeepSeek-V3 级外推、架构方案择优属于 Phase 1 启动前的独立 ADR 评审议题，Phase 0 不预先合法化任何"放宽硬约束"的备选方案（原则 7 例外条款：PATCH 端到端 < 300ms 属功能正确性，不可豁免）
- [ ] **P0-11** Docker 镜像 build（多阶段：Python backend + Next.js frontend），本地 `docker run` 通过，浏览器访问可看到 Qwen2-0.5B 3D 渲染
  - **必须记录实际镜像大小**并回填 README / 03-system-architecture.md 的「预计 1.3–2.0GB」占位
  - Phase 0 DoD 验收项：回填 README + 03 对应镜像大小占位行
  - **告警阈值**：若实际镜像 > 2.5GB，Phase 0 验收报告必须讨论镜像拆分策略（web/api 分离 / slim torch wheel / 移除 tokenizers 多语言资源）
- [ ] **P0-12** CI 管线最小集：前端 `pnpm lint && pnpm typecheck && pnpm test:unit`；后端 `ruff check . && mypy src/ && pytest`（对齐 02 §2.8；Phase 0 不强制覆盖率阈值，Phase 1 起强制 ≥80%）
- [ ] **P0-13** `trust_remote_code` 双路径验证：选取 **Phi-3-mini**（`microsoft/Phi-3-mini-4k-instruct`）跑通两条路径，仅验证双路径存在性；全量 top-download 架构矩阵（Qwen2-VL / InternLM2 / ChatGLM3 / Yi-VL 等 ≥ 5 个）挪至 Phase 1 补齐。
  - **Path A**（默认，`TRUST_REMOTE_CODE=true`）：完整 meta-device 加载，含远程自定义代码；产出精确 ModuleGraph（provenance=EXACT），正常渲染对应模板
  - **Path B**（`TRUST_REMOTE_CODE=false`）：降级为合成图（config-only）；`AutoConfig.from_pretrained(..., trust_remote_code=False)` 成功取 config.json；meta-device 加载跳过 → 仅基于 config.json 产出 ModuleGraph（provenance=INFERRED，caveats=`["auto_map 指向远程代码，已降级为合成图"]`）
  - 用户可见提示（Path B）：UI 顶部黄色 Banner "此模型含自定义代码，已降级为配置推断模式" + 节点徽标全部显示 INFERRED
  - DoD：Phi-3-mini 在 Path A 下可正常渲染对应模板；在 Path B 下可渲染（至少落 Template G）+ Banner 与 INFERRED 徽标均可见
- [ ] **P0-14** 仓库根目录补 `CONTRIBUTING.md`（开发环境搭建最小集）：Python 3.12 + uv / Node 20 + pnpm 9；`make dev` 或等价脚本启动后端 :8000 + 前端 :3000；HF_TOKEN 配置指引（含"无 token 也可用离线 fixture"说明，对齐 07 §七(a)）；常见错误三条（meta-device 不可用 / safetensors 404 / pnpm 版本不匹配）

### Phase 0 验收（DoD）

- 上述 **14 条 checklist 全部勾选** 即通过；
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

### Phase 1 任务依赖图 + 工作量估算

> T-shirt size：**S** = 0.5–1 天 / **M** = 1–2 天 / **L** = 3–5 天 / **XL** = > 5 天。工程师可按团队情况折算人天，文档只做量级约束。
>
> **注**：下方依赖图为概要示意（展示关键依赖路径），任务编号和描述**以后续 Checklist 为权威**。依赖图中的编号可能与 Checklist 不完全对应。

```
Phase 1a (后端)                      Phase 1b (前端)             Phase 1c (动画)        Phase 1d (交互)
─────────────────────                ─────────────────           ───────────────        ───────────────
P1a-01 Adapter Protocol (S) ───┐
P1a-02 4 Adapters (L) ─────────┼──→ P1b-01 TemplateContract+ABCG (L) ──┐
P1a-03 Pipeline 5阶段 (L) ─────┤                                       │
P1a-04 MemoryEstimator (M) ────┤                                       ├──→ P1c-01 AnimationLayer 基座 (M) ──┐
P1a-05 GPU Catalog YAML (S) ───┼──→ P1b-04 GPUSelector (S)              │    P1c-02 Attention 动画 (L) ──────┤
P1a-06 SSE 两段推送 (M) ───────┼──→ P1b-08 SSE/WS 订阅 (M)              │    P1c-03 MoE 路由动画 (M) ────────┤
P1a-07 PATCH /config (M) ──────┼──→ P1b-03 ConfigEditor (M) ───────────┤    P1c-04 Residual 动画 (S) ───────┤
P1a-08 WS /updates (S) ────────┘                                       │                                    │
P1a-09 Provenance 全路径 (M) ──────→ P1b-05 Provenance 徽标 (S)         │                                    │
                                                                        │                                    │
                                     P1b-02 R3F Scene 主组件 (M) ───────┴───────────────────────────────────┼──→ P1d-01 点选/tooltip (S)
                                     P1b-06 布局引擎对接 (S)                                                 ├──→ P1d-02 scrub 时间轴 (M)
                                                                                                              └──→ P1d-03 相机切换 (S)

检查点 ☑ 1a-check (P1a-01..12 完成)：后端能端到端跑通 Qwen2-0.5B 的 SSE 两段推送 + PATCH，WS 可收到 revision=3
检查点 ☑ 1b-check (P1b-01..06 + 1a-check)：前端能渲染 Qwen2-0.5B + Llama-3-8B + Mixtral + DeepSeek-V3 + GPT-2 五个模型的 ABCG 模板
检查点 ☑ 1c-check (P1c-01..04 + 1b-check)：三项 Stage-2 动画在 Mixtral 上可见、可暂停、可 scrub
检查点 ☑ 1d-check (P1d-01..06 + 1c-check)：交互预算（§4.8）实测达标 + 教学深度对照表通过评审
```

**关键约束**（注：以下编号引用依赖图示意中的任务编号，非 Checklist 任务 ID）
- 1a-01/03/06 是 1b/1c/1d 的前置，必须优先完成（关键路径）
- 1a-04/05 与 1a-06 可**并行**（无数据依赖）
- 1a-check 与 1b-check 之间留 **1 个增量验收检查点**（非全 26 项做完才验收）
- 同一人不得同时承接 1a 与 1b 的关键路径任务（避免串行化）

---

### Phase 1a — 后端（扩展契约 + Pipeline 五阶段 + 热更新）

- [ ] **P1a-01** `ArchitectureAdapter` Protocol 接口（11 §1.1）+ 显式 Registry（ADR-014；禁止 entry_points / pluggy）
- [ ] **P1a-02** 实现 4 个 adapter：`LlamaAdapter` / `LlamaMoEAdapter` / `DeepseekMoEAdapter` / `GenericAdapter (Template G)`；覆盖 LLaMA 族（Llama-3-8B / Qwen2）/ LLaMA-MoE（Mixtral-8x7B / Qwen2-MoE）/ DeepSeek-MoE（DeepSeek-V3）/ 未知架构（GPT-2 走 G）
- [ ] **P1a-03** Pipeline 五阶段纯函数化（11 §9）：S1 parse_structure / S2 detect_features / S3 synthesize_flows / S4 estimate_resources / S5 compute_layout；每阶段无全局状态、无隐式 I/O
- [ ] **P1a-04** `MemoryEstimator` Protocol + Registry + 唯一实现 `InferenceMemoryEstimator`（weights + KV cache + activations；ADR-019）
- [ ] **P1a-05** GPU Catalog YAML（`backend/data/gpu-catalog.yaml`）：最小集以 [11-extension-points §6.3](11-extension-points.md) 为唯一事实源（NVIDIA：A100-40G/80G、H100-80G、H200-141G、B200、RTX 4090-24G、RTX 3090-24G、L40S-48G；国产：昇腾 910B、寒武纪 MLU370、昆仑芯 P800/R200）。该 YAML 为唯一数据源，严禁代码硬编码
  - **国产卡最小字段要求**：昇腾 910B / 寒武纪 MLU370 / 昆仑芯 P800/R200 至少填 `memory_gb` + `bf16_tflops` 两个关键字段（对齐 11 §6.3.1）；`memory_bandwidth_gbps` / `fp8_tflops` / `nvlink_gbps` 允许 null，按 11 §6.3.1 语义走 caveats
- [ ] **P1a-06** ParallelismStrategy 接入备忘（v1.2+ 占位）：不写代码，不建目录，不建 Registry；在 Phase 1 末尾补 1 页 markdown 记录预期接入方式（对齐 11 §4 + ADR-020）
- [ ] **P1a-07** `PATCH /api/v1/stream/{org}/{repo}/config` 热更新路由（11 §8.2；后端 <200ms 预算）
- [ ] **P1a-08** WebSocket endpoint（增量推送 ModuleGraph snapshot，`revision` 字段单调递增；对齐 ADR-018 schema 向前兼容）
- [ ] **P1a-09** Provenance 全字段强制（ADR-016）：`ModuleNode` / `DataEdge` / `ArchitectureProfile` / `MemoryBreakdown` / `EstimateResult` 均携带 `{source, confidence ∈ {EXACT,INFERRED,ESTIMATED}, caveats}`；HTTP 响应头 `X-Provenance-Summary`
- [ ] **P1a-10** 错误契约统一为 RFC 7807 `application/problem+json`
- [ ] **P1a-11** 模板选择算法（整合自 09 §5.1.20）：遍历已注册 Adapter，依次调用 `adapter.detect(config)` → 首个命中者胜出 → 全部未命中 → **Template G**（通用回退），**禁止默认回退 A**（ADR-015）。`LlamaAdapter.detect()` 内含 `_matches_llama_family()`（**RoPE + RMSNorm + SwiGLU/GatedMLP** 三特征齐备才命中）
  - **通用 config 特征提取**（feature detection，非 model_type 分支；对齐 11 §1.3「不得出现 `if model_type == '...'`」）：
    - `sliding_window`：任何模型的 `config.sliding_window != null` 且 `config.use_sliding_window == true` → `ArchitectureProfile.features[]` 附加 `"sliding_window"`（适用于 Qwen2、Mistral 等任何启用滑动窗口的架构）
    - `tie_word_embeddings`：任何模型的 `config.tie_word_embeddings == true` → `features[]` 附加 `"tie_word_embeddings"`（影响参数量估算：embed 与 lm_head 共享不重复计数；适用于 Qwen2、Gemma、T5 等任何共享 embedding 的架构）
    - Template 渲染层根据 `features[]` 中的标记显示对应 tooltip 标签，参数量估算根据 `"tie_word_embeddings"` 特征正确去重 tied weights

  > 唯一事实源：09 §5.1.20。本处代码为概述副本，实现以 09 为准。

  ```python
  # ── 通用 config 特征提取（pipeline S2 detect_features 阶段执行） ──
  # 不含任何 model_type 分支；纯粹基于 config 字段是否存在（feature detection）

  def extract_features(config: dict) -> list[str]:
      """从 config 中提取通用特征标记，写入 ArchitectureProfile.features[]。"""
      features: list[str] = []
      # sliding_window：适用于任何启用滑动窗口的架构
      if config.get("sliding_window") is not None and config.get("use_sliding_window", False):
          features.append("sliding_window")
      # tie_word_embeddings：影响参数量估算（embed 与 lm_head 共享不重复计数）
      if config.get("tie_word_embeddings", False):
          features.append("tie_word_embeddings")
      return features


  # ── Adapter dispatch（对齐 11 §1.1 + 09 §5.1.20） ──
  # 遍历已注册 adapter → 首个 detect() 命中者胜出 → 全部未命中 → GenericAdapter (Template G)
  # 严禁 if model_type == "..." 分支（对齐 11 §1.3 硬性验收）

  def select_adapter(config: dict, registry: list[ArchitectureAdapter]) -> ArchitectureAdapter:
      """
      registry 由 @register 装饰器按注册顺序填充：
      [LlamaAdapter, LlamaMoEAdapter, DeepseekMoEAdapter, ...]
      """
      # 0. 遍历注册表，首个 detect() 命中者胜出
      #    未命中 Adapter 的架构（含 VLM / Encoder-Decoder）通过 GenericAdapter 自然回退 Template G
      #    （INFERRED 徽标 + 顶部免责浮层），不阻塞渲染。
      #    若需标识 VLM / Enc-Dec，放入 ArchitectureProfile.features 作为元数据。
      for adapter in registry:
          if adapter.detect(config):
              return adapter

      # 1. 全部未命中 → GenericAdapter（Template G；禁止默认回退 A）
      return GenericAdapter()


  # 唯一事实源：09 §5.1.20。本处为概述副本，实现以 09 为准。
  def _matches_llama_family(config: dict) -> bool:
      """Template A 三特征检测：RoPE + RMSNorm + SwiGLU/GatedMLP 必须同时命中。
         供 LlamaAdapter.detect() 内部调用，不含 model_type 判断。"""
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
          # gelu_pytorch_tanh 属 GELU 变体、非 Gated，剔除避免 LLaMA 族误判
          config.get("hidden_act", "").lower() in {"silu", "swiglu"}
          and config.get("intermediate_size") is not None
      )
      return has_rope and has_rmsnorm and has_gated_mlp
  ```

- [ ] **P1a-12** Adapter 冷启动 Smoke Test：挑选 ViT-B-16（非 LLaMA 族）跑一次 Adapter-only 接入，验证"≤ 1 文件 + 1 注册"的原则 8 架构承诺。**Phase 1a 验收项**——不通过则说明 Adapter 契约有设计缺陷，必须 Phase 1a 内修正。

### Phase 1b — 前端（TemplateContract + ConfigEditor + GPUSelector + Provenance）

- [ ] **P1b-01** `TemplateContract` 接口（11 §2）+ Registry；实现 Template **A / B / C / G**（覆盖 Llama-3-8B / Mixtral-8x7B / DeepSeek-V3 / GPT-2 未知回退）
- [ ] **P1b-02** R3F `<Scene>` 主组件：`<Canvas>` + `<OrbitControls damping>` + `<Environment>` + `<EffectComposer><Bloom/><SSAO/><ToneMapping mode="ACESFilmic"/></EffectComposer>`（ADR-004 固定三件套，缺一即违反原则 2）；材质默认 `MeshStandardMaterial`（ADR-003），不因性能主动降级
- [ ] **P1b-03** `<ConfigEditor>` 浮动面板（受控组件，白名单字段见 11 §8.1）：300ms debounce + `PATCH /config` 发送 + 等待 WS `revision+=1` snapshot
- [ ] **P1b-04** `<GPUSelector>` 下拉菜单：从 `/api/v1/gpus` 拉取 yaml 内容（11 §6），前端无硬编码 GPU 列表
- [ ] **P1b-05** Provenance 徽标组件：EXACT（绿）/ INFERRED（蓝 `#3b82f6`，空心圆 + ⓘ tooltip）/ ESTIMATED（黄）在节点与侧边面板均可见（ADR-015/ADR-016；配色以 05 §视觉规范为准）
- [ ] **P1b-06** 交互硬约束达标（ADR-012；非性能优化，属功能正确性）：
  - PATCH 端到端 **< 300ms**
  - 模块点选 / 悬停 **< 50ms**
  - 时间轴 scrub **< 16ms/frame**
  - 视角切换 / 相机 **< 16ms/frame**
  - 后端 PATCH /config 热更新 **< 200ms**

### Phase 1c — 动画（AnimationLayer 插件化 + L1/L2 最小子集）

- [ ] **P1c-01** v1.0 仅实现 L1 结构动画 + L2 数据流动画具体 React 组件；不抽象 AnimationLayer Protocol，不建 Registry（对齐 11 §3）
- [ ] **P1c-02** **L1 StructureAnimation**：模块展开 / 收起 / 层级过渡（v1.0 必交）
- [ ] **P1c-03** **L2 DataFlowAnimation 最小子集**（ADR-017）：① Attention Q/K/V 分解 ② MoE 路由（router → top-k → experts → 加权合并） ③ Residual flow
- [ ] **P1c-04** L1/L2 双层独立开关 UI + 统一时间轴（层间不得硬编码相互依赖，通过 `AnimationContext` 共享时间轴；对齐 05 §5.2 + 10 §10.10）

### Phase 1d — 交互（硬约束达标的 UI 细节）

- [ ] **P1d-01** 模块点选 + 悬停高亮（<50ms）
- [ ] **P1d-02** 相机 OrbitControls + damping（<16ms/frame）
- [ ] **P1d-03** 时间轴拖动 scrub（<16ms/frame）
- [ ] **P1d-04** 信息面板滑入（节点详情 / 参数 / Provenance caveats）
- [ ] **P1d-05** GSAP 微交互：hover scale / glow、click ripple（DOM overlay，不影响 R3F 帧率）
- [ ] **P1d-06** 原则 6 教学深度自评（Phase 1 验收参考项，非阻塞）：
  - 对标 [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) 现有动画（GPT-2 单模型），记录 v1.0 三项 Stage-2 动画（Attention Q/K/V + MoE 路由 + Residual flow）在以下维度的差异：
    - **模型真实度**：真实 Llama-3-8B / Mixtral-8x7B / DeepSeek-V3 vs GPT-2 玩具
    - **数据流完整性**：Q/K/V 三路分叉 + Softmax + 加权合并 vs 仅单层示意
    - **MoE 路由**：top-k 激活 + expert 扇形 + 加权回流（竞品完全缺失）
    - **Residual stream**：跨 block 残差主干 + Pre-LN 位置标注（竞品无此维度）
  - 产出物：一页对照表（Markdown）+ v1.0 截图 vs 竞品截图，存入 `docs/principle-6-design-notes.md`
  - **阻塞门槛**：Phase 1 验收仅要求三项动画**正确、流畅**；教学深度是持续打磨方向，不作一次性验收门禁

### Phase 1 验收（DoD）

- 上述 1a/1b/1c/1d **全部 checklist** 勾选；
- **扩展点全部就位**（Adapter / Template / AnimationLayer / MemoryEstimator / GPU Catalog / DataFlowDirection schema / ConfigEdit 路由；ParallelismStrategy 为 v1.2+ 占位，仅文档备忘）；
- **Provenance 全字段覆盖**（ADR-016）；
- **交互硬约束达标**（本节 P1b-06 四项）；
- **教学深度超越竞品**（P1d-06 对照表通过评审）；
- 单元测试覆盖率 **≥ 80%**（后端 pipeline 五阶段 + 前端核心组件）；
- **无性能门槛**（原则 7 / ADR-012）；FPS、冷启动时延、内存仅作观测记录。

---

## Phase 2 — 扩展演练（≤ 1 文件 + 1 注册）

> **目的**（原则 8 衡量标准）：选一个 **Phase 1 未覆盖且与 LLaMA 族特征不重合** 的架构，做 cold-start 接入，验证扩展成本。若做不到 **≤ 1 adapter 文件 + 1 注册**（+ 至多 ≤ 1 template 文件 + 1 注册），视为架构违规，**推倒重来**。
>
> **候选模型**（择一）：
> - **Mamba-1.4B**（`state-spaces/mamba-1.4b-hf`）—— 纯 SSM，无 attention，无 causal mask
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
- 单元测试覆盖率 **≥ 80%**；Provenance 全字段覆盖；
- **无性能门槛**（原则 7）；交互硬约束继续达标；
- **若任一未达 → 视为架构违规 → 回退并重构（推倒重来）**。

---

## Phase N — 性能优化集中阶段（延后执行，Phase 0/1/2 完成后启动）

> **目的**（原则 7 / ADR-012）：Phase 0/1/2 的正确性 + 教学性验收通过后，统一启动性能优化。**Phase 0/1/2 观测到的指标作为 Phase N 的决策输入**。
>
> 目标：追上 [README § 性能预算](README.md#性能预算观测指标非准入门槛) 的观测目标（3D FPS ≥ 30fps / API P95 / 首屏 < 3s / 3D 场景内存 < 200MB）。

### Phase N 任务清单（简述）

- [ ] **PN-01** 3D LOD 系统：相机距离降级（远距离合并层块、隐藏文字标签）
- [ ] **PN-02** `frameloop="demand"` 精细调度：对 Phase 1 已交付的渲染模式状态机 `static / interactive / animated`（ADR-006）进行**阈值调参**（hover timeout、idle 回退时间、animated → static 切回延迟等），状态机本体已在 Phase 1 交付，PN 阶段不新建
- [ ] **PN-03** Bundle splitting / lazy loading：Template B/C/G 按需加载，Three.js / R3F 分块
- [ ] **PN-04** L0 序列化缓存（orjson bytes 缓存） + L1 文件缓存调优（与 ADR-021 一致）
- [ ] **PN-05** GPU 能力检测 / FPS 自适应策略（ADR / 08 §Phase N 专项：高端 → MeshPhysicalMaterial + 10K 粒子；低端仍保持 PBR 材质基线（MeshStandardMaterial）并降低粒子与后处理采样）

### Phase N 验收（DoD）

- 观测指标达到或接近 README §性能预算目标（非硬门槛，但为 Phase N 的显式交付目标）；
- Phase 0/1/2 的交互硬约束**继续保持**（不得因 LOD / 降级引入点选或 scrub 延迟回退）；
- 单元测试覆盖率继续 ≥ 80%；Provenance 全字段覆盖不受影响。

---

## 统一 DoD（Definition of Done，适用于所有 Phase）

任一 Phase 的"完成"必须同时满足下列条款：

1. **单元测试覆盖率 ≥ 80%**（后端 pipeline 五阶段 + 前端核心组件）
2. **扩展点接入成本验证**：Phase 2 为显式验证（≤ 1 文件 + 1 注册）；Phase 1 为隐式（4 个 adapter 都满足此成本）
3. **Provenance 全字段覆盖**（ADR-016）：所有对外 schema 携带 `{source, confidence, caveats}`；HTTP 响应头携带 `X-Provenance-Summary`
4. **无性能门槛**（原则 7 / ADR-012）：FPS、冷启动时延、内存仅作观测记录，不作发布门禁
5. **交互硬约束达标**（原则 7 例外条款 / ADR-012；从 Phase 1 起）：
   - PATCH 端到端 **< 300ms**
   - 模块点选 **< 50ms**
   - 动画 scrub **< 16ms/frame**
   - 视角切换 **< 16ms/frame**
   - 后端 PATCH /config 热更新 **< 200ms**

---

[← 返回目录](README.md)
