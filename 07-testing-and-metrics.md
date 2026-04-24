# 八、验证矩阵 · 成功指标 · 路线图

> [HF Model Visualizer](README.md) 技术设计文档 — 章八+九

---

## 八、验证矩阵

| 模型 | 验证点 |
|---|---|
| `moonshotai/Kimi-K2.6` | 多模态(MoonViT) + MoE(384) + MLA + INT4 量化 → 全功能验证 |
| `deepseek-ai/DeepSeek-V3-0324` | MLA + MoE + shared experts + dense-layer 替换 |
| `Qwen/Qwen3-30B-A3B` | 标准 MoE（无 MLA、无量化） |
| `meta-llama/Llama-3.1-8B` | 标准 Dense + GQA（无 MoE/MLA/量化） |
| `google/gemma-3-27b-it` | Dense 大模型 |

**端到端测试流程**:
1. 启动后端：`cd backend && uvicorn app.main:app --reload --port 8000`
2. 启动前端：`cd frontend && npm run dev` (port 3000)
3. 访问 `http://localhost:3000`
4. 输入模型名 → 验证 3D 架构树、数据流粒子、MoE 网格、MLA 漏斗
5. 测试模型对比

---

## 八(b)、成功指标

### 北极星指标

**月度独立可视化模型数** — 反映产品核心价值的被采纳程度。

### 分层指标体系

| 层 | 指标 | v1.0 目标 | 衡量方式 |
|---|---|---|---|
| **获客** | 首页搜索提交次数 | 500/月 | 前端埋点 |
| **激活** | 3D 场景加载成功率 | >95% | API 成功率 + WebGL 检测 |
| **留存** | 7 日内再访率 | >20% | Cookie/localStorage |
| **深度** | 展开层内结构的用户比例 | >40% | 前端事件 |
| **传播** | 从分享链接进入的用户比例 | >10% | URL referrer |
| **性能** | API P95 响应时间 (L0 缓存) | <50ms | 后端监控 |
| **性能** | 3D 场景 FPS (Mac 集成 GPU) | >30fps | 前端 `<Perf>` |
| **性能** | 首屏加载时间 (信息面板可见) | <3s | Lighthouse |
| **覆盖** | 支持的 model_type 数量 | >60 | 自动化测试 |

### 用户旅程成功率

| 旅程 | 定义 | 目标 |
|---|---|---|
| Quick Check (80%) | 搜索→看到参数表+架构树 | <5s，>90% 成功 |
| Deep Dive (15%) | 展开层内结构+播放 Guided Tour | >60% 完成 Tour |
| Compare (5%) | 完成两模型对比 | >80% 成功 |

---

## 九、产品路线图 (Roadmap)

### v1.0 — 核心发布（本设计文档范围）

完整的 3D/2D 双模式可视化 + 端到端数据流 + Guided Tour + MoE/MLA/量化专项 + 模型对比。

### v1.1 — 部署与分发

| 里程碑 | 内容 | 预期影响 |
|---|---|---|
| HF Spaces 部署 | Docker 镜像 + Spaces 免费托管 | 零成本公开服务 |
| 独立 HTML 导出 | 离线自包含 HTML（含 3D 场景序列化） | 论文/博客嵌入 |
| 嵌入式组件 | `<iframe>` / Web Component 嵌入代码片段 | HF 模型卡片集成 |
| CLI 工具 | `npx hf-visualizer moonshotai/Kimi-K2.6` 本地启动 | 开发者本地使用 |

### v1.2 — 规模与性能
- CDN 预生成 Top-5000 模型 + WebGPU 渲染（10x 粒子性能）+ 渐进式加载（首屏 <1s）+ 国际化（中/英/日）
- 2D SVG 可视化模式（dagre 布局 + SVG 导出，面向论文和快速浏览）

### v1.3 — 社区与生态
- 自定义主题 + 可视化分享链接 + VS Code 插件 + 公开 REST API + 排行榜集成

### v2.x — 深度可视化
- **v2.0 并行策略**: TP/PP/DP/EP/CP/SP 可视化 + 训练数据流（详见 [parallel-visualization-design.md](parallel-visualization-design.md)）
- **v2.0 浏览器内推理**: Transformers.js 小模型推理 + 实时注意力热力图 + 逐层激活值 3D 热力体
- **v2.1 深度分析**: MoE 真实路由追踪 + 注意力模式分类 + 量化误差可视化
- **v2.2 训练可视化**: 训练过程回放 + 梯度流可视化 + Loss 景观 3D + LoRA/QLoRA 可视化

### v3.x+ — 平台化与 AI 原生
- **v3.0**: 多人协作标注 + 架构搜索引擎 + 架构模板市场 + 架构 Diff 时间线
- **v3.1**: 自然语言查询 + 架构对话助手 + 架构推荐 + 自动论文关联
- **v4.0**: VR/AR 沉浸式 + 模型手术台 + 分布式训练全景 + 逐权重编辑

---

[← 返回目录](README.md)
