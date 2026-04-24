# 架构决策记录 (ADR)

> [HF Model Visualizer](README.md) 技术设计文档 — 架构决策

---

## MVP 增量路径

```
MVP (Phase 1-3):
  后端 API + Next.js 脚手架 + 3D 基础架构树 + MoE 网格 + MLA 标注
  + 基本信息面板 + Bloom + 5 模型验证
  ↓
Phase 4: 端到端数据流动画（简化版粒子 + Guided Tour）
  ↓
Phase 5: MoE/MLA/量化专项深度交互
  ↓
Phase 6: 模型对比
  ↓
v1.2: 2D SVG 可视化模式
  ↓
v2.0: 并行策略可视化（TP/PP/EP/CP/SP + 训练数据流）
```

---

## 关键架构决策

### ADR-001: 后端依赖 — 保留 torch + transformers

- **决策**: 保留 torch + transformers 作为后端依赖
- **理由**: meta-device 加载真实模型树 + forward() 分析 + 零重写
- **替代方案**: 仅使用 huggingface_hub 轻量下载 config，放弃真实模型树
- **状态**: 已采纳

### ADR-002: 3D vs 2D 优先级 — 3D 优先

- **决策**: 3D 优先开发，2D 移至 v1.2
- **理由**: 核心差异化功能，对标 LLM Viz
- **替代方案**: 先实现 2D SVG 可视化，后补 3D
- **状态**: 已采纳

### ADR-003: 默认 3D 材质 — MeshStandardMaterial

- **决策**: 默认使用 MeshStandardMaterial
- **理由**: Mac 集成 GPU 性能优化（去 clearcoat/transmission）
- **替代方案**: MeshPhysicalMaterial（更真实但性能开销大）
- **状态**: 已采纳

### ADR-004: 后处理 — 仅 Bloom

- **决策**: 默认仅启用 Bloom 后处理
- **理由**: Mac 集成 GPU（去 Vignette/ChromaticAberration/Noise）
- **替代方案**: 全效果后处理链
- **状态**: 已采纳

### ADR-005: 粒子数量 — 默认 3K

- **决策**: 默认粒子上限 3K
- **理由**: Mac 集成 GPU 上限，高端设备可升至 10K
- **替代方案**: 统一 10K（低端设备卡顿）
- **状态**: 已采纳

### ADR-006: 渲染循环 — frameloop="demand"

- **决策**: 使用按需渲染模式
- **理由**: 按需渲染，idle 零 GPU 负载
- **替代方案**: 持续渲染（always）
- **状态**: 已采纳

### ADR-007: API 响应树结构 — 嵌套递归 TreeNode

- **决策**: 使用嵌套递归 TreeNode 结构
- **理由**: 匹配 explore_model.py 实际输出 + 前端层级展开需求
- **替代方案**: 扁平化 ID-parentID 列表
- **状态**: 已采纳

### ADR-008: 参数统计 — ParamStats 双形态统一

- **决策**: 精确(meta)和估算(config)共用一个模型 + `is_estimated` 标志
- **理由**: 统一前端消费接口，避免两套渲染逻辑
- **替代方案**: 分别定义 ExactParamStats 和 EstimatedParamStats
- **状态**: 已采纳

### ADR-009: 并行策略 — v2.0

- **决策**: 并行策略可视化移至 v2.0
- **理由**: 实现复杂度极高，v1.0 聚焦架构可视化
- **替代方案**: v1.0 包含基础 TP 可视化
- **状态**: 已采纳

### ADR-010: Zustand 状态范围 — 仅低频交互状态

- **决策**: Zustand 仅管理低频交互状态
- **理由**: 帧级数据走 useFrame + ref，避免 React 重渲染
- **替代方案**: 所有状态统一 Zustand 管理
- **状态**: 已采纳

### ADR-011: 缓存 — L0 内存 + L1 文件

- **决策**: 两层缓存架构（L0 进程内 LRU + L1 文件系统 JSON）
- **理由**: 跨重启持久化，v1.1 加 Redis
- **替代方案**: 纯内存 TTLCache（重启丢失）或直接引入 Redis（v1.0 过重）
- **状态**: 已采纳

---

[← 返回目录](README.md)
