# 三、系统架构（概要视图）

> ⚠ **本文档为概要性摘要**。
> - 与 [09-backend-detailed-design.md](09-backend-detailed-design.md) 冲突时，**以 09 为准**（模块职责、阶段契约、数据模型、缓存分段、SSE 协议）。
> - **扩展点契约**以 [11-extension-points.md](11-extension-points.md) 为准（Adapter / Template / AnimationLayer / MemoryEstimator / ParallelismStrategy Registry 的接口签名与注册方式）。

本章仅给出「组件拓扑 + 两条数据流 + 扩展点集成位置 + 部署形态」的鸟瞰图，用于快速建立心智模型。

---

## 3.1 定位与原则回顾

- **非商业化**：内部工具，**单租户、无鉴权、无 SSO**，不引入 API 网关 / 服务发现 / 多租户隔离。
- **性能**：v1.0 除「交互响应 < 200ms」外，**不做任何性能优化**；不预留分布式组件（无 Kafka / Redis 集群 / 消息队列）。
- **可扩展**：v1.0 架构须最健壮——所有易变点走 Registry（见 11）。
- **Provenance 可追溯**：每个节点/边/估算值必须携带来源与置信度（见 09 §5.1.2）。

---

## 3.2 组件拓扑

```
┌───────────────────────────── 前端层 (Next.js 15 / React 19) ─────────────────────────────┐
│                                                                                          │
│   app/model/[org]/[repo]/page.tsx                                                        │
│   ┌──────────────────────────────┐   ┌────────────────────┐   ┌──────────────────────┐  │
│   │   R3F Scene (Canvas)         │   │   ConfigEditor     │   │   GPUSelector        │  │
│   │   ┌────────────────────────┐ │   │   (PATCH /config)  │   │   (读 GPU Catalog)   │  │
│   │   │ AnimationLayer 堆叠     │ │   └────────────────────┘   └──────────────────────┘  │
│   │   │  L1 Structure          │ │                                                       │
│   │   │  L2 DataFlow           │ │   ┌───────────────────────── Zustand Store ────────┐ │
│   │   │  L3 Heatmap  (v1.1)    │ │   │   graph / layout / revision / overrides /      │ │
│   │   │  L4 Parallelism(v1.2)  │ │   │   selectedGPU / streamState                    │ │
│   │   └────────────────────────┘ │   └────────────────────────────────────────────────┘ │
│   └──────────────────────────────┘                                                       │
└────────────────────┬──────────────────────────────────────────────┬──────────────────────┘
                     │ SSE  (冷启动 revision=1 / 2)                 │ WebSocket 或 SSE
                     │ GET  /api/v1/stream/{org}/{repo}             │ PATCH /api/v1/stream/{org}/{repo}/config
                     ▼                                              ▼
┌───────────────────────────────── 后端层 (FastAPI) ───────────────────────────────────────┐
│                                                                                          │
│   FastAPI Router  (routers/model.py, routers/stream.py)                                  │
│                                                                                          │
│   Pipeline — 五阶段纯函数（无副作用，I/O 除外，见 09 §5.1.1）                              │
│   ┌──── S1 parse ────┬──── S2 detect ────┬──── S3 synthesize ─┬─ S4 estimate ─┬─ S5 layout ┐
│   │ config.json      │ ArchitectureProfile│ edges / flows      │ params/mem/   │ 2D+3D      │
│   │ safetensors hdr  │                    │                    │ FLOPs         │ 坐标       │
│   └──────────────────┴────────────────────┴────────────────────┴───────────────┴────────────┘
│           ▲                   ▲                                      ▲                    │
│           │                   │                                      │                    │
│   ┌───────┴───────┐   ┌───────┴─────────┐                  ┌─────────┴────────────┐       │
│   │ ArchitectureAdapter │ TemplateContract│                  │ MemoryEstimator       │     │
│   │    Registry        │    Registry     │                   │   Registry           │     │
│   └────────────────────┘ (前端消费)       │                   └──────────────────────┘     │
│                                                                                           │
│   ┌──────────────────────────┐                                                            │
│   │ ParallelismStrategy      │  v1.0 仅接口就位，未接入主流程（见 11 §5）                  │
│   │    Registry              │                                                            │
│   └──────────────────────────┘                                                            │
└───────────────────────────────────┬─────────────────────────────────┬────────────────────┘
                                    │                                 │
                                    ▼                                 ▼
┌─────────────────── 数据层 ─────────────────┐       ┌───────────── 外部 ──────────────┐
│  L0 内存缓存 (cachetools / 进程内)         │       │   HuggingFace Hub               │
│  L1 文件缓存 (~/.cache/hf-visualizer)       │◄─────►│   (config.json / safetensors   │
│  GPU Catalog YAML (backend/data/gpus.yaml) │       │    header / repo metadata)     │
│   └─ 被 S4 / GPUSelector 消费               │       └─────────────────────────────────┘
└────────────────────────────────────────────┘
```

**依赖方向**：Router → Pipeline（纯函数） → Registry/数据层；Registry 与数据层之间**不直接互调**。

---

## 3.3 数据流

### 3.3.1 冷启动路径（SSE 两段推送）

```
Browser
  │  GET /api/v1/stream/{org}/{repo}   (Accept: text/event-stream)
  ▼
FastAPI Router
  │
  ├─► S1 parse          config.json + safetensors header  (L0/L1 缓存命中则复用)
  │
  ├─► S2 detect         ArchitectureAdapter Registry 选出 adapter
  │                     → ArchitectureProfile + ModuleGraph 骨架
  │
  │◄── emit SSE { revision: 1, is_final: false }    ← 首屏可渲染快照
  │
  ├─► S3 synthesize     补 edges / data-flow
  ├─► S4 estimate       params / memory / FLOPs  (MemoryEstimator Registry + GPU Catalog)
  ├─► S5 layout         2D+3D 坐标
  │
  └─► emit SSE { revision: 2, is_final: true }      ← 最终完整快照
```

- 每段 SSE 都是**完整可渲染快照**（不是增量/进度条）。前端收到 revision=1 立即首屏，收到 revision=2 整体替换。
- 数据模型（ModuleGraph / ArchitectureProfile / Provenance / StageOutcome）见 09 §5.1.2。

### 3.3.2 热更新路径（PATCH /config）

```
用户在 ConfigEditor 修改 hidden_size / num_layers / …
  │
  ▼
PATCH /api/v1/stream/{org}/{repo}/config   body: { overrides: {...} }
  │
  ▼
Router 复用原始 config + overrides（合并后为 effective_config）
  │
  ├─► 跳过 S1（config 已在内存，safetensors header 不受 overrides 影响）
  ├─► S2 detect      （adapter 通常不变，快路径直通）
  ├─► S3 synthesize
  ├─► S4 estimate
  └─► S5 layout
  │
  └─► 通过 WebSocket 或 SSE 推送 { revision: N+1, is_final: true }
       ── 后端预算 **< 200ms**（交互响应要求，见 3.1）
```

---

## 3.4 扩展点集成位置

所有扩展的**接口签名与注册方式以 11 为准**，本节仅标注它们在架构图中的位置与被调用时机。

| 扩展点 | Registry 位置 | 被谁消费 | 调用时机 |
|---|---|---|---|
| **ArchitectureAdapter** | 后端 `backend/adapters/` | S2 `detect_features` | S1 完成后、S2 开始时选型 |
| **TemplateContract** | 前端 `frontend/src/templates/` | R3F Scene | 收到 SSE 后前端渲染选型 |
| **AnimationLayer (L1–L4)** | 前端 `frontend/src/animations/` | R3F Scene | 渲染时按模板声明叠加（L1 基底，L2–L4 可选层） |
| **MemoryEstimator** | 后端 `backend/estimators/` | S4 `estimate_resources` | 每个估算策略一个实例，按 profile 匹配 |
| **ParallelismStrategy** | 后端 `backend/parallel/` | 预留接口，**v1.0 未接入主流程** | v1.2 起在 S4/S5 之间调用 |
| **GPU Catalog** | 后端 `backend/data/gpus.yaml` | S4（显存判定）+ 前端 GPUSelector | 全流程只读数据依赖 |

---

## 3.5 部署形态

### 3.5.1 定位

单租户内部工具，**单实例 Docker 容器即可**：
- 不做多区域、不做高可用、不做多副本负载均衡。
- **K8s 是可选项**，不作为前置条件；裸 Docker 足够。
- 不引入 API 网关、服务注册中心、消息队列。

### 3.5.2 开发模式

```
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  FastAPI (uvicorn --reload)  │  HTTP  │  Next.js dev server           │
│  :8000                       │◄──────►│  npm run dev   :3000          │
└──────────────────────────────┘        └──────────────────────────────┘
        两个进程独立启动，前端 proxy 到后端 8000
```

### 3.5.3 生产模式

**单 Docker 镜像，多阶段构建**：

```
┌─ Stage 1: python:3.12-slim      ─► 安装 backend 依赖（CPU-only torch）
├─ Stage 2: node:20-slim          ─► next build  → standalone 静态产物
└─ Stage 3: python:3.12-slim (终镜像)
     ├── /app/backend              （FastAPI + uvicorn）
     ├── /app/frontend/.next/standalone  + /static + /public
     └── entrypoint.sh：启动 uvicorn :8000 + node server.js :3000
```

- Next.js 采用 **`output: 'standalone'`** 模式，最终镜像只需很薄的 Node 运行时即可承载静态产物与少量 SSR。
- 镜像大小按当前依赖估算 ~1.3–2.0 GB（torch CPU + transformers 占主要体积，以实际构建为准）。

---

[← 返回目录](README.md)
