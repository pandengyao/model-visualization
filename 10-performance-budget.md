# 性能预算

> [HF Model Visualizer](README.md) 技术设计文档 — 性能指标

---

## 核心性能预算

| 指标 | 目标 | 约束 |
|---|---|---|
| 3D 场景 FPS (Mac 集成 GPU) | ≥ 30fps | StandardMaterial + Bloom only + 3K 粒子 |
| API 响应 (L0 缓存) | < 10ms | 进程内 LRU |
| API 响应 (L1 缓存) | < 50ms | 文件系统 JSON |
| API 响应 (HF Hub) | < 3s | 包含 config 下载 + 检测 + 树构建 |
| 3D 场景内存 | < 200MB | |
| 首屏加载 | < 3s | |

---

## GPU 能力检测与自动材质选择

### GPU 能力检测（v1.1 计划）

通过 WebGL 扩展查询自动检测 GPU 能力，选择最优材质级别：

| 检测项 | 高端 GPU | Mac 集成 GPU | 低端/移动端 |
|---|---|---|---|
| `MAX_TEXTURE_SIZE` | ≥8192 | ≥4096 | <4096 |
| `WEBGL_compressed_texture_s3tc` | 有 | 可能有 | 无 |
| 材质 | MeshPhysicalMaterial | MeshStandardMaterial | MeshBasicMaterial |
| 后处理 | Bloom + Vignette + ACES | Bloom only | 无 |
| 粒子上限 | 10K | 3K | 1K |

---

## FPS 自适应策略

### FPS 自适应策略（v1.1 计划）

运行时 FPS 监控 + 自动降级：
- 连续 5 帧 < 25fps → 粒子数量减半
- 连续 10 帧 < 20fps → 关闭 Bloom 后处理
- 连续 20 帧 < 15fps → 切换到 MeshBasicMaterial
- 用户可手动覆盖自动降级（设置面板）

---

[← 返回目录](README.md)
