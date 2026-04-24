# 一、项目背景

> [HF Model Visualizer](README.md) 技术设计文档 — 章一

### 1.1 市场空白

基于对 40+ 工具/服务的全面调研，确认一个明确的市场空白：

**不存在一个服务能做到：输入 HuggingFace 模型路径 → 自动生成 3D 交互式架构可视化 + 推理流程图。**

最接近的竞品：
- **Netron** (32.8K stars) — 需上传模型文件，不能输入 HF 路径；只有静态 2D op-level 图
- **Google Model Explorer** (1.4K stars) — 需导出为 TFLite/ONNX 格式
- **Transformer Explainer** (7.2K stars) — 只支持 GPT-2 一个模型，2D 可视化
- **LLM Viz** (5.4K stars) — 只支持 minGPT，3D 效果极佳但无法泛化

细分空白领域：MoE 路由可视化、MLA 注意力可视化、多模态架构统一可视化、KV Cache 可视化、量化影响可视化。

### 1.1.1 标杆产品技术深度分析与借鉴

> **完整分析**: [appendix-a-benchmark-analysis.md](docs/model-visualization/appendix-a-benchmark-analysis.md)
>
> 基于对 Netron、Google Model Explorer、Transformer Explainer、LLM Viz 四个标杆产品的**源码级**深入分析，提炼出 36 项可借鉴特性（10 项 TIER-1 必须实现 + 10 项 TIER-2 高价值 + 6 项 TIER-3 锦上添花）。

| 来源 | License | 核心借鉴 |
|---|---|---|
| **Netron** (MIT) | SVG + Dagre | Web Worker 布局 + 操作元数据系统 + 双路径边命中检测 |
| **Model Explorer** (Apache-2.0) | WebGL + Three.js | GPU 实例化渲染 + 层级展开/折叠 + 分屏同步对比 + LOD |
| **Transformer Explainer** (MIT) | Svelte + D3 | GSAP 分段动画 + Sankey 数据流 + 注意力矩阵可视化 |
| **LLM Viz** (无 License, 仅思路参考) | 自研 WebGL | **3D 空间隐喻**(Y=推理, X=分支, Z=并行) + 弹簧物理相机 + Guided Tour + 按需渲染 |

### 1.2 目标

构建一个 Web 服务，用户输入 HuggingFace 模型路径（如 `moonshotai/Kimi-K2.6`），即可获得：
1. **3D 交互式架构可视化** — 可旋转、缩放、点击展开的 3D 模型架构（默认模式）
2. **2D 结构图可视化** — 经典的 DAG 布局 + SVG 渲染，适合快速浏览和论文导出
3. **2D/3D 一键切换** — 工具栏切换按钮，两种模式共享数据但各自最优渲染
4. **端到端推理数据流** — 发光粒子沿推理路径流动，展示 tensor shape 变化
5. **MoE/MLA/量化专项可视化** — 专家路由网格、MLA 压缩漏斗、量化热力图
6. **模型信息面板** — 参数统计、模型卡片、配置详情
7. **模型对比** — 两个模型并排 2D/3D 对比

### 1.3 已有资产

`hf-model-explorer` skill（`/Users/frank/work/.claude/skills/hf-model-explorer/scripts/`）提供了可复用的后端核心逻辑：

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

| 画像 | 典型角色 | 核心诉求 | 使用频率 |
|---|---|---|---|
| **A: ML 研究员/工程师** (主要) | 模型微调者、架构设计者、论文作者 | 快速理解新模型架构、对比架构差异、论文插图 | 高频 (每周多次) |
| **B: 基础设施工程师** (主要) | 推理部署、训练平台、显存优化 | 了解参数分布做部署规划、量化影响评估、MoE 路由分析 | 中频 (每周 1-2 次) |
| **C: 学生/博主** (次要) | AI 学习者、技术博主、课程讲师 | 理解 Transformer 内部机制、生成教学素材、博客配图 | 低频 (按需) |

#### 三条用户旅程

| 旅程 | 占比 | 场景 | 关键页面 | 完成标准 |
|---|---|---|---|---|
| **Quick Check** | ~80% | "Kimi-K2.6 是 MoE 还是 Dense？多少参数？" | 首页 → 搜索 → 信息面板 | <5s 看到参数表 + 架构徽章 |
| **Deep Dive** | ~15% | "MLA 注意力的 KV 压缩比是多少？数据怎么流的？" | 模型页 → 展开层 → Guided Tour | 完成 Tour 或自由探索 >2min |
| **Compare** | ~5% | "Qwen3-30B 和 Kimi-K2.6 的 MoE 结构有什么区别？" | 对比页 → 并排查看 → 差异高亮 | 识别出关键配置差异 |

#### 设计优先级原则

1. **信息密度优先于动画炫酷** — 专业用户要的是快速获取准确信息，动画是锦上添花
2. **首屏即答** — 80% 用户的 Quick Check 需求必须在首屏（无需滚动/展开）满足
3. **渐进式复杂度** — 默认显示简洁概要，点击/展开才揭示深层细节
4. **专业术语不翻译** — MoE、MLA、GQA、RoPE 等术语保持英文原文

### 1.5 用户获取渠道 (GTM)

| 渠道 | 策略 | 预期效果 |
|---|---|---|
| **HuggingFace Spaces** | Docker 镜像部署到 HF Spaces，零成本公开服务 | 直接触达 HF 用户群 |
| **社区推广** | Reddit (r/MachineLearning)、Twitter/X、知乎、Hacker News | 开发者社区曝光 |
| **论文引用** | 提供 BibTeX 引用格式，鼓励论文中使用可视化图片 | 学术圈传播 |
| **技术博客** | 发布架构可视化对比文章（如 "Kimi-K2.6 vs DeepSeek-V3 架构深度对比"） | SEO + 长尾流量 |
| **HF 模型卡片集成** | 提供嵌入代码片段，模型作者可在 README 中嵌入可视化 | 生态融合 |
| **开发者工具链** | CLI 工具 `npx hf-visualizer <model>` + VS Code 插件 | 开发者工作流集成 |

---

[← 返回目录](README.md)