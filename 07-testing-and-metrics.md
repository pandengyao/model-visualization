# 七、测试策略 · 验证矩阵 · 成功指标 · 产品路线图

> [HF Model Visualizer](README.md) 技术设计文档 — 测试、验证与路线图
>
> **测试与指标的权威性**：本文档的测试策略与成功指标严格对齐 README 产品原则——
> - 对齐 **原则 3（结构与数据流 100% 正确）**：以结构正确性、数据流正确性、Provenance 全字段覆盖作为硬约束，宁可不测，不可误测；
> - 对齐 **原则 5（前期不做性能优化；但交互响应是硬约束）**：3D FPS / API P95 / 内存 / 首屏等全部降为观测指标，仅 PATCH 端到端 / 点选 / scrub / 后端 PATCH 等交互响应作为硬约束；
> - 对齐 **原则 6（可扩展）**：「扩展接入成本 ≤ 1 文件 + 1 注册」作为必测项，通过 Mamba / ViT 扩展演练自动化脚本验证；
> - 对齐 **原则 9（Provenance 可追溯）**：Provenance 每字段覆盖率必须 100%，confidence 三级（EXACT/INFERRED/ESTIMATED）各有测试样本，`caveats` 在已知不确定场景必须填写。
>
> **已删除所有外部商业指标**（DAU / MAU / 留存率 / NPS / 付费转化 / ARR / CAC / LTV / GitHub Stars / HF Spaces 嵌入量 / SEO 关键词排名 / Twitter / Reddit 讨论热度 等），因原则 1 规定本产品非商业化、仅供团队内部使用。

---

## 七(a)、验证矩阵（真实 HF 模型）

> 对齐原则 7（真实模型优先）与原则 8（架构广度底线）。每个模型都必须以 meta-device 实际加载后跑通完整 pipeline（parse → detect → synthesize → estimate → layout），并比对 HF 官方 config 字段。

#### Gated 模型凭证与离线 fixture 策略

部分模型（`meta-llama/Llama-3-8B` / `meta-llama/Llama-3-70B` / `deepseek-ai/DeepSeek-V3` / 部分 Mistral/Gemma 版本）在 HF Hub 需要授权访问。CI 执行策略：

1. **CI 凭证**：`HF_TOKEN` 作为 GitHub Actions secret 注入 `env.HF_TOKEN`；读权限 token（只读 gated repo）由项目维护者单独申请并按季度轮换
2. **离线 fixture 回落**（对齐原则 1 非商业化 + 原则 7 真实模型）：每个 gated 模型在 `tests/fixtures/gated/<org>_<repo>/` 提交其 `config.json` + `model.safetensors.index.json`（仅 metadata 头，无权重），大小 < 10KB/模型；测试默认读 fixture，`HF_TOKEN` 存在时切换为真实 Hub 拉取并比对 fixture 是否过期（CI warn，季度同步）
3. **无 token 的 PR 环境**（外部贡献者）：自动走 fixture 路径，跳过真实网络拉取；CI 标记 `[offline]` 但仍必须通过 fixture 校验
4. **禁止**：在 repo 内提交真实权重文件；在测试代码中明文硬编码任何 token

### Decoder-only（v1.0 必测）

| 模型 | 预期 Template | 预期参数数 | 预期 Provenance `layers_used` 信号源 | 预期 `is_final=true` 延迟（观测） |
|---|---|---|---|---|
| `Qwen/Qwen2-0.5B` | A | ~0.5 B | config.num_hidden_layers + safetensors layer 扫描 | < 2s |
| `Qwen/Qwen2.5-7B` | A | ~7 B | config + safetensors meta | < 3s |
| `meta-llama/Llama-3-8B` | A | ~8 B | config + safetensors meta | < 3s |
| `meta-llama/Llama-3-70B` | A | ~70 B | config + safetensors multi-shard index | < 5s |
| `gpt2` | **G**（INFERRED，absolute pos + LayerNorm） | ~124 M | config + safetensors meta；Adapter 未命中 LLaMA 族三条件（RoPE+RMSNorm+SwiGLU） | < 2s |
| `EleutherAI/gpt-j-6b` | **G**（INFERRED，RoPE + LayerNorm + GELU） | ~6 B | config + safetensors meta；缺 RMSNorm → 回退 G | < 3s |
| `tiiuae/falcon-7b` | **G**（INFERRED，RoPE + LayerNorm） | ~7 B | config + safetensors meta；缺 RMSNorm/SwiGLU → 回退 G | < 3s |
| `microsoft/phi-2` | **G**（INFERRED，RoPE + LayerNorm + GELU） | ~2.7 B | config + safetensors meta；缺 RMSNorm/SwiGLU → 回退 G | < 2s |
| `bigcode/starcoder2-7b` | **G**（INFERRED，RoPE + LayerNorm） | ~7 B | config + safetensors meta；缺 RMSNorm/SwiGLU → 回退 G | < 3s |

### MoE（v1.0 必测）

| 模型 | 预期 Template | 预期参数数 | 预期 Provenance 关键字段 | 预期 `is_final=true` 延迟（观测） |
|---|---|---|---|---|
| `mistralai/Mixtral-8x7B-v0.1` | B | ~46.7 B | `num_local_experts`、`num_experts_per_tok`（EXACT，来自 config） | < 4s |
| `Qwen/Qwen1.5-MoE-A2.7B` | B | ~14.3 B | `num_experts` + shared_experts 标记（INFERRED，需 caveat） | < 4s |

### MoE + MLA（v1.0 必测）

| 模型 | 预期 Template | 预期参数数 | 预期 Provenance 关键字段 | 预期 `is_final=true` 延迟（观测） |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V3` | C | ~671 B | MLA(`q_lora_rank`/`kv_lora_rank`) + MoE(`n_routed_experts`/`n_shared_experts`)（EXACT） | < 6s |
| `deepseek-ai/DeepSeek-V2-Lite` | C | ~15.7 B | MLA + MoE（EXACT） | < 3s |

### Encoder（v1.0 必测 · 应回退 Template G，禁止误判为 A）

| 模型 | 预期 Template | 预期参数数 | 预期 Provenance 关键字段 | 预期 `is_final=true` 延迟（观测） |
|---|---|---|---|---|
| `bert-base-uncased` | **G**（通用回退，INFERRED 徽标） | ~110 M | `model_type=bert`，Adapter 未命中 → G | < 2s |
| `google/flan-t5-base` | **G** | ~248 M | Encoder-Decoder，Adapter 未命中 → G | < 2s |

> 硬约束：**Encoder 族绝不允许默认回退 Template A**，违反即测试失败（原则 8）。
>
> **Template A 判定规则（对齐 08 ADR-015 与 09 `_matches_llama_family()`）**：需同时满足 `RoPE + RMSNorm + SwiGLU-family 激活`。GPT-2/GPT-J/Falcon/Phi-2/StarCoder2 因缺少 RMSNorm（皆为 LayerNorm）或激活函数不在 SwiGLU 族，均应回退 Template G（INFERRED 徽标）。这一保守策略遵循原则 3「100% 正确 > 漂亮展示」与原则 9「Provenance 可追溯」——宁可标 INFERRED，不可错认为 LLaMA。

### ViT / 多模态（v1.1 扩展演练用）

| 模型 | 预期 Template | 预期参数数 | 预期 Provenance 关键字段 | 预期 `is_final=true` 延迟（观测） |
|---|---|---|---|---|
| `google/vit-base-patch16-224` | G（v1.0） → 专用 ViT Adapter（v1.1） | ~86 M | `image_size` / `patch_size`（EXACT） | < 2s |
| `openai/clip-vit-base-patch32` | G（v1.0） → CLIP Adapter（v1.1） | ~151 M | 双塔 vision/text（INFERRED） | < 2s |

### SSM（v1.1 扩展演练用）

| 模型 | 预期 Template | 预期参数数 | 预期 Provenance 关键字段 | 预期 `is_final=true` 延迟（观测） |
|---|---|---|---|---|
| `state-spaces/mamba-1.4b` | G（v1.0） → Mamba Adapter（v1.1） | ~1.4 B | SSM state_size，无 attention（INFERRED） | < 2s |
| `RWKV/rwkv-6-world-1b6` | G（v1.0） → RWKV Adapter（v1.1） | ~1.6 B | time-mix / channel-mix（INFERRED） | < 2s |

### 量化（v1.0 需正确识别 `quantization_config`）

| 模型 | 预期 Template | 预期参数数 | 预期 Provenance 关键字段 | 预期 `is_final=true` 延迟（观测） |
|---|---|---|---|---|
| `TheBloke/Llama-2-7B-GPTQ` | A + 量化徽标 | ~7 B（INT4） | `quantization_config.quant_method="gptq"` + `bits=4`（EXACT） | < 3s |
| `TheBloke/Llama-2-7B-AWQ` | A + 量化徽标 | ~7 B（INT4） | `quantization_config.quant_method="awq"` + `bits=4`（EXACT） | < 3s |

> 模型总数：Decoder 9 + MoE 2 + MoE+MLA 2 + Encoder 2 + ViT/多模态 2 + SSM 2 + 量化 2 = **21 个**。

---

## 七(b)、测试分层

### 1. 单元测试

- **后端 pipeline 五阶段**：对每个纯函数（`parse_structure` / `detect_features` / `synthesize_flows` / `estimate_resources` / `compute_layout`）做 input→output 对比测试，固定 fixture 输入，断言输出结构与数值；
- **Adapter 注册与检测逻辑**：`detect()` 方法在正/反样本上断言行为（如 LlamaAdapter 对 `model_type=llama` 返回 True、对 `bert` 返回 False）；
- **MemoryEstimator 计算**：对若干 (graph, TrainingConfig, GPU) 组合断言 weights / gradients / optimizer_states / activations / kv_cache / comm_buffer 六类显存分量；
- **Provenance 字段强制**：任一 Schema 字段（ModuleGraph.Node / DataEdge / ArchitectureProfile / MemoryBreakdown …）若缺 `provenance`，单测失败；
- **Template 匹配算法**：对每个 ArchitectureProfile 断言命中唯一 Template；对无命中情形断言回退 Template G。

### 2. 集成测试

- **冷启动 SSE 端到端**：21 个必测模型各跑一遍（含 `is_final=false` 分段推送顺序、`revision` 单调递增、最终 `is_final=true` 结构完整）；
- **PATCH `/config` 热更新**：每个架构家族（Decoder / MoE / MoE+MLA / Encoder / 量化）至少 1 个模型，测试典型 overrides（`num_hidden_layers` / `num_experts_per_tok` / `tp_size`）；
- **WebSocket / SSE 消息顺序与 revision 单调递增**：断言每个 session 内消息严格单调、无回退、无重复；
- **GPU Catalog YAML 加载**：启动时解析 `backend/data/gpu-catalog.yaml`，断言最小集（A100-40G/80G、H100-80G、H200-141G、B200、4090-24G、3090-24G、L40S-48G、昇腾 910B、寒武纪 MLU370、昆仑芯 P800/R200）全部成功加载；schema 不合法即启动失败。

### 3. 视觉回归测试

- **规模**：每个 Template（A / B / C / G）× 3 个代表模型 × 3 个相机角度 = **36 张快照**；
- **工具**：Playwright + 3D canvas 截图对比（像素差容忍阈值 0.1%）；
- **触发**：PR 上 CI 自动跑，有 diff 要求人工确认更新 baseline；
- 目的：兜住 Template 渲染结构不被误改（对齐原则 3「结构 100% 正确」在渲染层的验证）。

### 4. 扩展接入测试（原则 6 的量化验收）

- **Phase 2 DoD**：用 Mamba（SSM，无 attention）**或** ViT（无 causal mask）做一次冷启动 Adapter 接入演练；
- 自动化脚本 `scripts/test_extension_cost.py` 验证：
  1. 新增文件数 ≤ 1（仅在 `backend/adapters/<name>.py` 新建）；
  2. 注册改动 ≤ 1 行（仅在 `backend/adapters/__init__.py` 追加 `from . import <name>`，由模块内 `@register` 装饰器完成注册）；
  3. 未触动 `detect_features` / `synthesize_flows` / `compute_layout` / 路由层；
  4. 未新增 `if model_type == "..."` 分支；
  5. 该新 Adapter 对应模型能跑通冷启动 SSE 且产出合法 ModuleGraph。

### 5. 交互响应测试（原则 5 例外条款的硬约束）

- **工具**：Playwright 自动化模拟用户编辑 config 字段；
- **断言**：PATCH 端到端 < 300ms、后端 PATCH < 200ms、点选 < 50ms、scrub < 16ms/frame；
- **失败处理**：任一交互超预算即阻塞合入；性能观测类指标（3D FPS / 首屏加载）即使恶化也不阻塞。

---

## 七(c)、Provenance 测试矩阵

对齐原则 9。每个 Schema 字段都必须有 Provenance，测试组织如下：

| 维度 | 要求 | 失败判定 |
|---|---|---|
| 覆盖率 | ModuleGraph / DataEdge / ArchitectureProfile / MemoryBreakdown 的**每个字段** 100% 有 `provenance` | 任一字段缺 provenance → 测试失败 |
| Confidence = EXACT | 样本：meta-device 实际加载的层数/头数/维度（可直接读 state_dict shape） | 若标注为 EXACT 但无法复核来源 → 失败 |
| Confidence = INFERRED | 样本：从 config 推断（如 `num_key_value_heads` 缺失时回退 `num_attention_heads`）、从 safetensors shape 反推 | 必须填 `source` 与 `caveats` |
| Confidence = ESTIMATED | 样本：MemoryEstimator 的 activations / comm_buffer 估算、FLOPs 估算 | 必须填估算公式来源（论文 / 代码实现链接）|
| `caveats` 必填场景 | `num_key_value_heads` 缺失回退 / MoE shared_experts 结构需推断 / 量化 bits 未显式写出时 | 已知不确定场景 caveats 为空 → 失败 |

---

## 七(d)、成功指标（按原则重分类）

### 硬约束（Phase 1 起必达，任一不达则阻塞发布）

| 类别 | 指标 | 硬约束值 | 对齐原则 |
|---|---|---|---|
| 结构正确性 | 所有 21 个必测模型 Adapter 产出 ModuleGraph 的层数 / 头数 / 维度与 HF config 100% 吻合 | 差异为 0 | 原则 3 / 7 |
| 数据流正确性 | v1.0 三项 Stage-2 动画（Attention Q/K/V 分解、MoE 路由、Residual flow）与真实计算流程 100% 一致（人工 + fixture 双重校对） | 差异为 0 | 原则 3 / 4 |
| Provenance 覆盖 | 所有 Schema 字段 100% 有 provenance，EXACT / INFERRED / ESTIMATED 三级均有覆盖样本 | 100% | 原则 9 |
| 交互响应 | PATCH 端到端 | < 300 ms | 原则 5 例外 |
| 交互响应 | 模块点选 / 悬停高亮 | < 50 ms | 原则 5 例外 |
| 交互响应 | 动画时间轴拖动 scrub | < 16 ms/frame（60fps） | 原则 5 例外 |
| 交互响应 | 后端 config-only 热更新 (PATCH /config) | < 200 ms | 原则 5 例外 |
| 扩展接入成本 | 新 Adapter / Template / AnimationLayer / ParallelismStrategy / MemoryEstimator / GPU | ≤ 1 文件 + 1 注册 | 原则 6 |
| Template G 回退正确性 | Encoder / ViT / SSM 等未识别架构**必须**回退 G；不得默认回退 A | 无误判 | 原则 8 |

### 观测指标（Phase N 前仅记录，不作准入门槛）

| 指标 | 观测目标 | 说明 |
|---|---|---|
| 3D FPS（Mac 集成 GPU） | ≥ 30 fps | Phase 0/1/2 只记录，不阻塞 |
| API P95（L0 缓存） | < 10 ms | 仅观测 |
| API P95（L1 缓存） | < 50 ms | 仅观测 |
| API P95（HF Hub 冷启动） | < 3 s | 仅观测 |
| 3D 场景内存 | < 200 MB | 仅观测 |
| 首屏加载 | < 3 s | 仅观测 |

### 已删除的外部商业指标

按原则 1（非商业化，仅供团队内部使用），下列指标一律不采集、不作为目标：

- 用户规模类：DAU / MAU / 留存率（7 日 / 30 日） / NPS
- 商业转化类：付费转化率 / ARR / CAC / LTV
- 传播类：GitHub Stars / HuggingFace Spaces 嵌入量 / Twitter / Reddit 讨论热度
- 获客类：SEO 关键词排名 / 首页搜索提交次数 / 从分享链接进入的用户比例

---

## 七(e)、产品路线图（Roadmap）

> 已删除所有商业化节点（公测 Beta、付费层级、GTM 发布等），对齐原则 1。

### v1.0 — 核心发布

范围详见 [README「v1.0 范围」](README.md) 与 [06-implementation-phases](06-implementation-phases.md)。包含：

- HF 模型 → 3D 结构可视化（Template A/B/C/G）
- Stage-1 结构动画 + Stage-2 最小子集（Attention Q/K/V 分解、MoE 路由、Residual flow）
- PATCH `/config` 动态编辑 + 交互响应硬约束
- MemoryEstimator 推理版 + GPU 选型（≥ 8 款）
- Provenance 强制字段全覆盖
- 扩展接口就位（ArchitectureAdapter / TemplateContract / AnimationLayer / ParallelismStrategy / MemoryEstimator / DataFlowDirection.backward 占位）

### v1.1 — 深度与广度扩展

- 反向传播动画（DataFlowDirection = backward / forward_backward_split_screen / overlay）
- Megatron-LM（TP+PP+SP）/ FSDP（ZeRO-1/2/3）MemoryEstimator
- Stage-2 其余动画：脉动 / 膨胀 / 螺旋 / 热力图 / token residual
- AST 源码解析（modeling_*.py 精细化）
- ViT / Mamba 专用 Adapter（扩展演练落地实现）
- 模型对比分屏

### v1.2 — 并行策略与训练数据流

- TP / PP / DP / EP / CP / SP 并行策略可视化（含通信原语动画：AllReduce / AllGather / ReduceScatter / All2All / P2P）
- N-D 组合（TP+PP、TP+PP+DP、TP+PP+DP+EP）
- 训练数据流全链路（梯度累积、激活 checkpointing、优化器步）
- 2D SVG 模式（dagre 布局 + SVG 导出，面向论文插图与快速浏览）

### v2.0 — 多模态与硬件精细化

- 多模态统一模型（Vision-Language 统一 Template）
- 国产卡 MemoryEstimator 精细化（昇腾 / 寒武纪 / 昆仑芯）
- 自定义模型：允许上传本地 `state_dict` / `config.json` 离线可视化

---

[← 返回目录](README.md)
