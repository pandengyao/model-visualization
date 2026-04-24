# 评审意见与实施待办

> [HF Model Visualizer](README.md) 技术设计文档 — 评审记录

> 综合自审查、工程师评审、架构师评审三轮意见。
> 已解决的问题归档于 11.1，已吸收到正文的待办标注为 ✅，仍需关注的延后项列于 11.2。

---

## 11.1 已解决的问题（存档）

以下问题已通过正文修改解决：

| 问题 | 解决位置 |
|---|---|
| API 契约缺失 TreeNode/FlowStep 等定义 | Section 5.2 Pydantic Schema |
| 统一错误响应格式 | Section 5.2.1 `ErrorResponse` |
| 是否保留 torch + transformers | Section 2.1 已决策保留 |
| 3D 性能预算不切实际 | Section 6.6 材质与光影 — 双档规格（默认/高端） |
| 2D vs 3D 优先级 | Phase 2.5 已标注 3D 优先，2D 移至 v1.2 |
| 并行策略可视化版本 | Section 九 已移至 v2.0 |
| 安全规则与 SafeTensors 验证矛盾 | Section 5.4 已更新 |
| 前端状态分级策略 | ✅ Phase 2 Step 2.5 Zustand 状态分级 |
| 分层缓存替代纯内存 TTLCache | ✅ Section 5.3 三层缓存 + Step 1.9 cache.py |
| HF Hub 降级策略 | ✅ Section 5.3 HF Hub 降级策略表 |
| flow_generator v1.0 范围精简 | ✅ Step 1.7 模板表增加 v1.0 列 + SafeTensors 异步标注 |
| 前端组件按功能域分组 | ✅ Section 四 three/ 子目录已重构 |
| 参数估算准确性测试 | ✅ Step 1.5 + tests/test_param_estimator.py |
| 端到端动画分轮实现 | ✅ Phase 4 Step 4.1/4.3 Round 1/Round 2 标注 |
| `trust_remote_code` 安全边界 | ✅ Section 5.4 安全规则第 1 条 |
| model_id 注入防护 | ✅ Section 5.4 安全规则第 3 条 |
| 6.6 材质/后处理与 Phase 3 不一致 | ✅ 6.6 + 6.1.1C + E 统一为默认/高端双档 |

---

## 11.2 v1.1+ 延后处理

| # | 问题 | 方案 | 时机 |
|---|---|---|---|
| 1 | 模板系统改为 YAML 数据驱动 | 替代硬编码 Python 模板 | v1.1 |
| 2 | README HTML 清洗防 XSS | 后端 `nh3` 清洗 + 前端 `react-markdown` 严格模式 | v1.1 |
| 3 | Docker + CI/CD + 监控 | Dockerfile 多阶段构建 + GitHub Actions + structlog + Sentry | v1.1 |
| 4 | GSAP license 确认 | 确认 3.12 "no charge" 条款是否覆盖 SaaS | 编码前 |
| 5 | 3D 渲染三档自动降级 | High/Medium/Low 基于 `renderer.capabilities` 检测 | v1.1 |
| 6 | 响应式设计 | v1.0 仅桌面端(≥1280px)，移动端 v1.3 | v1.3 |
| 7 | 可访问性 | 3D 文字替代视图 + 色盲友好配色 | v1.2+ |
| 8 | 后端服务拆分 | v1.1 加异步任务队列，v1.2 按需微服务化 | v1.1+ |

---

[← 返回目录](README.md)
