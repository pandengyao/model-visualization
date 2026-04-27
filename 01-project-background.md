# 一、项目背景

> [HF Model Visualizer](README.md) 技术设计文档 — 章一

### 1.0 产品初心

这是一个**作者本人 + 团队内部 ML 研究员/工程师**自用的工具，**不是商品**。

- 不考虑商业化：不做多租户、配额、计费、公开注册、SEO、营销漏斗、GTM、付费分层、滥用限流。
- 唯一目的：让自己和团队在理解现代 LLM 架构（尤其是 LLaMA 族、LLaMA-MoE、DeepSeek-MoE 等）时**用得爽**。
- 所有设计决策以"我们自己用得爽"为准绳，不为外部用户妥协；在精美度、教学深度、结构正确性、架构健壮性上不设上限，在商业化相关维度上零投入。
- 相关联的核心原则（参见 [README](README.md) 的产品原则）：精美 3D + 现代化前端、结构与数据流 100% 正确、教学深度与动画精细度大幅超越同类工具、前期不做性能优化（交互响应延迟除外）、v1.0 架构必须健壮且可扩展、真实模型优先不造玩具、架构广度有底线、Provenance 可追溯。

### 1.1 我们为什么需要这个产品

基于对 40+ 现有工具/服务的调研，现有方案在我们的真实工作流中都不够用：

- **Netron** — 需上传模型文件，不能直接输入 HF 路径；只有静态 2D op-level 图，看不到 MoE 路由、MLA 压缩、数据流这类我们真正关心的语义。
- **Google Model Explorer** — 需先导出为 TFLite/ONNX；对我们日常 PyTorch/HF 工作流阻力太大，且对 MoE/MLA 等现代结构无专项表达。
- **Transformer Explainer** — 只支持 GPT-2 单一模型，2D 可视化；无法覆盖 LLaMA 族、MoE、多模态。
- **LLM Viz** — 只支持 minGPT，3D 效果极佳但完全无法泛化到真实模型。

由此暴露的真实痛点（这些就是我们要做这个产品的理由）：

1. **无法"输入 HF 路径 → 直接看 3D 架构 + 推理数据流"**——我们每天都要理解新模型，这条最短路径不存在。
2. **MoE 路由可视化缺失**——DeepSeek-MoE / LLaMA-MoE 的专家路由、共享专家、Top-K 激活几乎没有工具能直观呈现。
3. **MLA 注意力可视化缺失**——KV 压缩结构、Latent 维度、吸收矩阵等在现有工具里完全不可见。
4. **多模态架构统一视图缺失**——视觉编码器 + LLM + 投影层的整体结构无法一张图看完。
5. **KV Cache / 量化影响缺失**——推理部署和显存规划时没有工具能把这些"内部状态"可视化。
6. **教学深度与动画精细度不够**——现有工具要么静态，要么只在一个玩具模型上动，看不到真实架构的流动过程。

### 1.1.1 技术差异化目标

> **完整分析**: [appendix-a-benchmark-analysis.md](appendix-a-benchmark-analysis.md)
>
> 基于对 Netron、Google Model Explorer、Transformer Explainer、LLM Viz 的**源码级**分析，提炼出 26 项可借鉴特性（10 项 TIER-1 必须实现 + 10 项 TIER-2 高价值 + 6 项 TIER-3 锦上添花）。

本产品不与这些工具做"商业竞争"，而是在以下**技术维度**全面超越它们，以满足我们自己的使用需求：

| 维度 | 现状（上述工具） | 我们要做到的 |
|---|---|---|
| **精美 3D** | 仅 LLM Viz 在玩具模型上实现；其他皆 2D | 真实模型上的精美 3D，现代化前端审美 |
| **真实模型优先** | 仅支持单一玩具（GPT-2 / minGPT）或需导出格式 | 直接吃 HuggingFace 路径，LLaMA 族 / LLaMA-MoE / DeepSeek-MoE 原生支持 + Template G 通用回退 |
| **动画精细度** | 静态图或粗粒度动画 | 数据流粒子、MoE 路由高亮、MLA 压缩漏斗、逐层 tensor shape 演变 |
| **教学深度** | 停留在 op-level 或单层示意 | 从整体架构 → 层 → 子模块 → 算子级的渐进揭示，结构与数据流 100% 正确 |
| **架构广度** | 单模型 / 单格式 | LLaMA 族、LLaMA-MoE、DeepSeek-MoE 为底线，Template G 兜底未知架构 |
| **Provenance 可信度** | 基本没有 | 每个可视化元素可追溯到 config/权重/推断来源，避免"好看但不对" |

我们从借鉴的 License 兼容实现中取各家所长：

| 来源 | License | 核心借鉴 |
|---|---|---|
| **Netron** (MIT) | SVG + Dagre | Web Worker 布局 + 操作元数据系统 + 双路径边命中检测 |
| **Model Explorer** (Apache-2.0) | WebGL + Three.js | GPU 实例化渲染 + 层级展开/折叠 + 分屏同步对比 + LOD（Phase N） |
| **Transformer Explainer** (MIT) | Svelte + D3 | GSAP 分段动画 + Sankey 数据流 + 注意力矩阵可视化 |
| **LLM Viz** (无 License, 仅思路参考) | 自研 WebGL | **3D 空间隐喻**(Y=推理, X=分支, Z=并行) + 弹簧物理相机 + Guided Tour + 按需渲染 |

### 1.2 目标能力

> v1.0 冻结范围的权威定义见 [06-implementation-phases.md](06-implementation-phases.md) §v1.0 冻结块与 [README §原则 10](README.md)。本表仅为快速参考。

输入 HuggingFace 模型路径（如 `deepseek-ai/DeepSeek-V3`），产出以下 7 项能力。**v1.0 交付其中 4 项**（✅ 标记），其余 3 项推迟至 v1.1+（🕐 标记）：

1. ✅ **v1.0** — **3D 交互式架构可视化**：可旋转、缩放、点击展开的 3D 模型架构（默认且唯一模式）
2. 🕐 **v1.1+** — **2D 结构图可视化**：经典 DAG 布局 + SVG 渲染，适合论文导出（v1.0 不交付，见 [06 §v1.0 范围冻结](06-implementation-phases.md)）
3. 🕐 **v1.1+** — **2D/3D 一键切换**：工具栏切换按钮（依赖 #2，同步推迟）
4. ✅ **v1.0** — **端到端推理数据流**：发光粒子沿推理路径流动，展示 tensor shape 变化（Stage-2 范围：Attention Q/K/V + MoE 路由 + Residual flow 三项，见 05 §5.3）
5. ✅ **v1.0** — **MoE/MLA/量化专项可视化**：专家路由网格、MLA 压缩漏斗、量化徽标
6. ✅ **v1.0** — **模型信息面板**：参数统计、模型卡片、配置详情、Provenance 徽标
7. 🕐 **v1.1+** — **模型对比**：两个模型并排 3D 对比（v1.0 冻结决议推迟；API 端点 `/api/v1/compare` 仅保留占位）

> v1.0 范围等价于 README 的「v1.0 必交付 / 冻结推迟」两栏；此处使用能力编号便于和原 PRD 对齐。

### 1.3 已有资产

`hf-model-explorer` skill（`/Users/frank/work/.claude/skills/hf-model-explorer/scripts/`）提供了可复用的后端核心逻辑：

> **原则 9 / 原则 11 约束**：`build_synthetic_tree()` 和 `estimate_params_from_config()` 的输出为 config-derived 合成数据，**一律打 INFERRED**（非 EXACT）。仅 meta-device ground truth 路径（L2）的输出可标 EXACT。

| 函数 | 文件 | 功能 |
|---|---|---|
| `fetch_model_card()` | explore_model.py:42 | HF 模型卡片信息获取 |
| `detect_moe()` | explore_model.py:128 | MoE 检测（含 DeepSeek-V3 风格） |
| `detect_mla()` | explore_model.py:194 | MLA 注意力检测 |
| `detect_quantization()` | explore_model.py:220 | 量化配置检测 |
| `detect_sub_configs()` | explore_model.py:268 | 多模态子配置检测 |
| `extract_key_config()` | explore_model.py:307 | 关键配置提取 |
| `build_synthetic_tree()` | explore_model.py:578 | 从 config 构建合成架构树 |
| `estimate_params_from_config()` | explore_model.py:428 | MLA/MoE 感知的参数估算 |
| `_estimate_attention_params()` | explore_model.py:347 | 注意力参数估算（MLA 感知） |
| `_estimate_mlp_params()` | explore_model.py:383 | MLP 参数估算 |
| `_estimate_vision_params()` | explore_model.py:390 | ViT 视觉编码器参数估算 |
| `format_params()` | explore_model.py:964 | 参数量格式化 |
| `tree_to_text()` | explore_model.py:943 | 树结构文本化 |

### 1.4 目标用户与使用场景

#### 用户画像

本产品的用户范围是**封闭的、明确的**：作者本人 + 团队内部 ML 研究员/工程师。不面向外部、不面向公开用户。

| 画像 | 典型角色 | 核心诉求 | 使用频率 |
|---|---|---|---|
| **A: 作者本人 / 团队 ML 研究员** | 模型微调者、架构设计者、论文合作者 | 快速理解新模型架构、对比架构差异、在内部分享中作为插图 | 高频（每周多次） |
| **B: 团队内部基础设施工程师** | 推理部署、训练平台、显存优化 | 了解参数分布做部署规划、量化影响评估、MoE 路由分析 | 中频（每周 1–2 次） |

#### 三条核心使用旅程

| 旅程 | 占比 | 场景 | 关键页面 | 完成标准 |
|---|---|---|---|---|
| **Quick Check** | ~40% | "DeepSeek-V3 是 MoE 还是 Dense？多少参数？" | 首页 → 搜索 → 信息面板 | <5s 看到参数表 + 架构徽章 |
| **Deep Dive** | ~45% | "MLA 注意力的 KV 压缩比是多少？数据怎么流的？" | 模型页 → 展开层 → 数据流动画 → Guided Tour | 完成 Tour 或自由探索 >2min |
| **Compare** | ~15% (v1.1+) | "Qwen3-30B 和 DeepSeek-V3 的 MoE 结构有什么区别？" | 对比页 → 并排查看 → 差异高亮 | 识别出关键配置差异 |

> **旅程比例说明**：Deep Dive 是我们团队的核心使用路径，不是边缘场景。v1.0 Deep Dive 核心路径聚焦于：Attention Q/K/V + MoE 路由 + Residual flow 三项 Stage-2 动画 + MLA 压缩 + 量化徽标 + 推理版 MemoryEstimator。并行策略可视化（v1.2+ 规划，v1.0 不交付）、前向/反向传播动画（v1.1+ 规划，v1.0 不交付）、显存估算等高级功能是本产品相对其他工具的核心差异化。

#### 设计优先级原则

1. **信息密度优先于动画炫酷** — 信息密度优先于纯装饰性动效，动画必须服务于教学理解
2. **首屏即答** — Quick Check 需求必须在首屏（无需滚动/展开）满足
3. **渐进式复杂度** — 默认显示简洁概要，点击/展开才揭示深层细节
4. **数据流动画是核心功能** — 不可裁剪，是我们日常使用中最有价值的差异化能力
5. **专业术语不翻译** — MoE、MLA、GQA、RoPE 等术语保持英文原文

---

[← 返回目录](README.md)
