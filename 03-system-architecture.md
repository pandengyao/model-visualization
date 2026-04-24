# 三、系统架构与项目结构

> [HF Model Visualizer](README.md) 技术设计文档 — 章三+四

### 3.1 整体架构

```
┌─────────────────┐     HTTP/JSON      ┌─────────────────┐
│                 │ ←───────────────── │                 │
│   Next.js App   │                    │  FastAPI Server  │
│   (React + R3F) │ ──────────────────→ │                 │
│                 │  /api/v1/model/*   │  ┌───────────┐  │
│  ┌───────────┐  │                    │  │ config    │  │
│  │ 3D Scene  │  │                    │  │ _parser   │  │
│  │ (R3F+Drei)│  │                    │  ├───────────┤  │
│  ├───────────┤  │                    │  │ detectors │  │
│  │ Info Panel│  │                    │  ├───────────┤  │
│  │ (React)   │  │                    │  │ tree_     │  │
│  ├───────────┤  │                    │  │ builder   │  │
│  │ Controls  │  │                    │  ├───────────┤  │
│  └───────────┘  │                    │  │ param_    │  │
│                 │                    │  │ estimator │  │
└─────────────────┘                    │  ├───────────┤  │
                                       │  │ model_    │  │
                                       │  │ card      │  │
                                       │  ├───────────┤  │
                                       │  │ flow_     │  │
                                       │  │ generator │  │
                                       │  └───────────┘  │
                                       │                 │
                                       │  TTL Cache      │
                                       │  ↕               │
                                       │  HF Hub API     │
                                       └─────────────────┘
```

### 3.2 数据流

```
用户输入 "moonshotai/Kimi-K2.6"
    │
    ▼
Next.js Route → /model/moonshotai/Kimi-K2.6
    │
    ▼ fetch
FastAPI: GET /api/v1/model/moonshotai/Kimi-K2.6
    │
    ├─ 检查 TTL Cache → 命中则直接返回
    │
    ├─ httpx 下载 config.json (from HF Hub)
    ├─ huggingface_hub.model_info() 获取元数据
    ├─ httpx 下载 README.md（完整内容）
    │
    ├─ config_parser.extract_key_config(config_dict)
    ├─ detectors.detect_moe(config_dict)
    ├─ detectors.detect_mla(config_dict)
    ├─ detectors.detect_quantization(config_dict)
    ├─ detectors.detect_sub_configs(config_dict)
    │
    ├─ tree_builder.build_synthetic_tree(config_dict)
    ├─ param_estimator.estimate_params_from_config(config_dict)
    ├─ flow_generator.generate_data_flow(config_dict)
    │
    ▼ 写入 Cache + 返回 JSON
React 接收 JSON → 渲染 3D 场景
```

---

## 四、项目结构

```
/Users/frank/work/hf-model-visualizer/
│
├── backend/                              # FastAPI 后端
│   ├── pyproject.toml
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI 入口 + CORS + 路由挂载
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── model.py                  # GET /api/v1/model/{org}/{repo}[/config|tree|params]
│   │   │   └── compare.py               # POST /api/v1/compare
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── config_parser.py          # async 下载 + 解析 config.json
│   │   │   ├── detectors.py              # MoE/MLA/量化/子配置检测
│   │   │   ├── tree_builder.py           # 合成树生成
│   │   │   ├── param_estimator.py        # 参数估算
│   │   │   ├── model_card.py             # HF 模型卡片 + README 完整内容
│   │   │   └── flow_generator.py         # 推理数据流步骤生成
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py               # Pydantic 响应模型
│   │   └── cache.py                      # TTLCache 封装
│   └── tests/
│       ├── test_config_parser.py
│       ├── test_detectors.py
│       ├── test_param_estimator.py        # 参数估算准确性测试（含 ground truth）
│       └── test_api.py
│
├── frontend/                             # Next.js + R3F 前端
│   ├── package.json
│   ├── next.config.js
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── postcss.config.js
│   ├── public/
│   │   └── favicon.svg
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx                # Root layout (dark theme)
│   │   │   ├── page.tsx                  # 首页：搜索框 + 热门模型卡片
│   │   │   └── model/
│   │   │       └── [org]/
│   │   │           └── [repo]/
│   │   │               └── page.tsx      # 模型可视化主页面
│   │   ├── components/
│   │   │   ├── ui/                       # 通用 UI 组件
│   │   │   │   ├── SearchBar.tsx
│   │   │   │   ├── ModelHeader.tsx       # 模型基本信息 + 徽章
│   │   │   │   ├── ParamTable.tsx        # 参数统计表
│   │   │   │   ├── ModeToggle.tsx        # 2D/3D 模式切换按钮
│   │   │   │   ├── Sidebar.tsx           # 五视图上下文侧边栏
│   │   │   │   ├── TimelineControl.tsx   # Guided Tour 时间轴控制条
│   │   │   │   ├── SearchPanel.tsx       # 正则搜索面板（4 种匹配模式）
│   │   │   │   ├── BookmarkBar.tsx       # 书签栏（1-9 快捷键）
│   │   │   │   └── LoadingSpinner.tsx
│   │   │   ├── two-d/                    # 2D 可视化组件
│   │   │   │   ├── Graph2D.tsx           # SVG DAG 图主容器
│   │   │   │   ├── Node2D.tsx            # 2D 节点（圆角矩形 + 颜色编码）
│   │   │   │   ├── Edge2D.tsx            # 2D 边（贝塞尔曲线 + 箭头）
│   │   │   │   ├── DataFlowAnim2D.tsx    # 2D 数据流动画（SVG 路径动画）
│   │   │   │   ├── SplitPane.tsx         # 分屏对比面板
│   │   │   │   └── ExportSVG.tsx         # SVG/PNG 导出
│   │   │   └── three/                    # 3D 可视化组件（按功能域分组）
│   │   │       ├── core/                 # 场景基础设施
│   │   │       │   ├── Scene.tsx         # Canvas + 灯光 + 后处理 + 相机
│   │   │       │   ├── CameraRig.tsx     # 弹簧物理相机控制 + 飞越动画
│   │   │       │   ├── ConnectionLines.tsx# 层间连线（QuadraticBezierLine）
│   │   │       │   ├── LayerBlock.tsx    # 单层 3D 块（RoundedBox + 材质）
│   │   │       │   └── InfoPanel.tsx     # Html overlay 信息面板
│   │   │       ├── architecture/         # 架构可视化
│   │   │       │   ├── ModelArchitecture.tsx # 完整模型 3D 架构布局
│   │   │       │   ├── MoEGrid.tsx       # MoE 专家 3D 网格 (InstancedMesh)
│   │   │       │   └── MLAFlow.tsx       # MLA 漏斗 3D 动画
│   │   │       ├── dataflow/             # 数据流动画
│   │   │       │   ├── DataFlowParticles.tsx # 数据流发光粒子系统
│   │   │       │   ├── InputVisualization.tsx # 数据预处理动画
│   │   │       │   ├── OutputVisualization.tsx# 输出可视化
│   │   │       │   └── TensorShape3D.tsx # 张量形状 3D 变形动画
│   │   │       └── tour/                 # Guided Tour
│   │   │           └── GuidedTour.tsx    # 自动播放 Guided Tour (GSAP Timeline)
│   │   ├── lib/
│   │   │   ├── api.ts                    # 后端 API 客户端
│   │   │   ├── types.ts                  # TypeScript 类型定义
│   │   │   ├── colors.ts                 # 颜色编码常量
│   │   │   └── layout.ts                 # 3D 布局计算
│   │   └── stores/
│   │       └── useModelStore.ts          # Zustand 状态管理
│   └── .env.local                        # NEXT_PUBLIC_API_URL=http://localhost:8000
│
└── README.md
```

### 3.3 部署策略

#### Docker 多阶段构建

```dockerfile
# Stage 1: Python 依赖
FROM python:3.12-slim AS backend-deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers huggingface_hub fastapi uvicorn httpx cachetools

# Stage 2: 前端构建
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 3: 最终镜像
FROM python:3.12-slim
COPY --from=backend-deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=frontend-build /app/frontend/.next /app/frontend/.next
COPY backend/ /app/backend/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 关键优化

| 优化项 | 方案 | 效果 |
|---|---|---|
| **CPU-only torch** | 使用 `pytorch.org/whl/cpu` 索引 | 镜像 ~1.5GB (vs CUDA 版 ~5GB) |
| **层缓存** | 将 pip install 和 npm ci 放在 COPY 源码之前 | 依赖层可复用，构建 <2min |
| **预缓存热门模型** | 启动时异步预加载 Top-100 模型配置 | 冷启动后首次请求 <500ms |
| **transformers 保留** | 后端依赖 torch+transformers | meta-device 真实模型树 + forward() 分析 |

> **注意**: CPU-only torch 已足够支持 `torch.device("meta")` 加载和 transformers 推理验证，无需 CUDA。

---

[← 返回目录](README.md)