# HF Model Visualizer — 技术设计文档

> HuggingFace 模型结构 3D 交互式可视化 Web 服务 | v1.0

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
| [08-review-feedback](08-review-feedback.md) | 技术评审意见原文（3 轮）、延后处理项 |
| [09-architecture-decisions](09-architecture-decisions.md) | 架构决策记录（ADR 格式）、MVP 增量路径 |
| [10-performance-budget](10-performance-budget.md) | 性能预算目标、GPU 能力检测、FPS 自适应策略 |
| [11-design-review](11-design-review.md) | 三角色设计评审（资深前端设计师、产品总监、软件架构师） |
| [标杆产品分析](benchmark-analysis.md) | Netron/Model Explorer/Transformer Explainer/LLM Viz 源码级分析 |
| [并行策略设计](parallel-visualization-design.md) | v2.0 TP/PP/DP/EP/CP/SP 可视化设计 |

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
| 部署 | CPU-only torch + Docker 多阶段构建 | 镜像 ~1.5GB，保留 transformers 推理能力 |

## 目标用户

- **ML 研究员/工程师** (主要) — 快速理解架构、对比差异、论文插图
- **基础设施工程师** (主要) — 部署规划、量化评估、MoE 路由分析
- **学生/博主** (次要) — 理解 Transformer 机制、教学素材

## 性能预算

| 指标 | 目标 |
|---|---|
| 3D FPS (Mac 集成 GPU) | ≥ 30fps |
| API P95 (L0 缓存) | < 10ms |
| API P95 (L1 缓存) | < 50ms |
| API P95 (HF Hub) | < 3s |
| 3D 场景内存 | < 200MB |
| 首屏加载 | < 3s |

---

## 变更日志

| 日期 | 版本 | 变更内容 | 涉及文件 |
|---|---|---|---|
| 2026-04-23 | v1.0 | 初始文档拆分：从 2525 行单文件拆分为 14 个分文件 | 全部 |
| 2026-04-23 | v1.0 | 项目命名：HF Model X-Ray → HF Model Visualizer | 全部 |
| 2026-04-23 | v1.0 | 三角色设计评审（15 项建议全部采纳） | 01/02/03/04/05/06/10/11 |
| 2026-04-23 | v1.0 | 新增内容：GTM 策略、部署策略、缓存并发安全、SSE 进度推送、Analytics 集成 | 01/03/04/06 |
