# 二、技术栈

> [HF Model Visualizer](README.md) 技术设计文档 — 章二

### 2.1 技术选型

| 层 | 技术 | 选型理由 |
|---|---|---|
| **前端框架** | Next.js 15 (React 19) | 最大生态、SSR/ISR 支持、API Routes |
| **3D 渲染** | React Three Fiber (R3F) + Drei | 最成熟的声明式 3D 框架（Drei 9.6K stars），60+ 现成组件 |
| **3D 引擎** | Three.js | Web 3D 事实标准，InstancedMesh 处理 10K-100K 对象 |
| **动画** | GSAP (100% 免费) | 时间轴编排、ScrollTrigger 滚动驱动、MorphSVG |
| **数据映射** | D3.js (d3-scale, d3-color) | 颜色映射、比例尺计算 |
| **UI 样式** | Tailwind CSS | 实用优先的 CSS |
| **后端** | FastAPI | 异步 IO（关键：HF Hub API 200-500ms 延迟）、Pydantic 自动验证 |
| **HTTP 客户端** | httpx | 异步下载 config.json |
| **模型信息** | huggingface_hub | 模型元数据 API |
| **缓存** | cachetools (TTLCache) | 内存缓存，24h TTL |

> **GSAP 使用边界**: GSAP 仅用于 HTML overlay 层的动画（侧边栏展开、面板过渡、时间轴控制器等）。3D 场景内的动画统一使用 `react-spring` (通过 `@react-three/drei`) 和 R3F 的 `useFrame` hook，避免 GSAP 直接操作 3D 对象导致的 React 渲染冲突。

### 2.2 为什么选 React + R3F 而非 Svelte + Threlte

| 维度 | React + R3F + Drei | Svelte + Threlte |
|---|---|---|
| 3D 生态成熟度 | **10/10** (Drei 9.6K stars) | 8/10 (60+ 组件) |
| 社区/文档 | **10/10** (最大社区) | 7/10 (较新) |
| 框架性能 | 7/10 (虚拟 DOM) | 10/10 (零虚拟 DOM) |
| 示例丰富度 | **10/10** | 7/10 |
| 综合 | **9/10** | 9/10 |

选择 R3F 的决定因素：**3D 开发中遇到问题时，R3F 社区的解决方案数量远超 Threlte**。对于 3D 密集型项目，生态成熟度比框架性能更重要。

### 2.3 关键依赖版本

**前端**:
```json
{
  "next": "^15",
  "react": "^19",
  "@react-three/fiber": "^8",
  "@react-three/drei": "^9",
  "three": "^0.170",
  "gsap": "^3.12",
  "d3-scale": "^4",
  "d3-color": "^3",
  "d3-shape": "^3",
  "dagre": "^0.8",
  "zustand": "^5",
  "tailwindcss": "^4"
}
```

**后端**:
```toml
[dependencies]
fastapi = ">=0.115"
uvicorn = ">=0.34"
httpx = ">=0.28"
huggingface_hub = ">=0.28"
cachetools = ">=5.5"
transformers = ">=4.57"
torch = ">=2.4"
```

**保留 torch + transformers** — 理由：
1. `transformers.AutoConfig` 提供标准化的配置对象，所有检测函数可直接复用
2. `torch.device("meta")` 零权重加载获取**真实模型结构树**（非合成树）
3. 模型类 `forward()` 方法是生成准确数据流的最可靠来源
4. 零重写工作量，直接复用 `explore_model.py` 全部函数

Docker 镜像 ~2.5GB，通过多阶段构建 + 层缓存优化；冷启动 ~30s，通过预缓存 Top-100 热门模型缓解。

### 2.4 3D 技术栈详解

#### Drei 关键组件映射

| 可视化需求 | Drei 组件 | 说明 |
|---|---|---|
| 384 MoE 专家 | `<Instances>` / `<InstancedMesh>` | 单次 draw call 渲染 384 个几何体 |
| 3D 文字标签 | `<Text>` (troika-three-text SDF) | 任意缩放清晰 |
| 2D 信息面板 | `<Html>` (CSS2DRenderer) | 3D 场景中嵌入 React DOM |
| 相机控制 | `<OrbitControls>` | 旋转/缩放/平移 |
| 相机动画 | `<CameraControls>` | 平滑相机过渡 |
| 连线 | `<Line>` / `<QuadraticBezierLine>` | 层间连接线 |
| 环境 | `<Environment>` | HDR 环境贴图 |
| 后处理 | `<EffectComposer>` + `<Bloom>` (postprocessing) | 选择性辉光 |
| 圆角盒 | `<RoundedBox>` | 层块几何体 |
| 渐变材质 | `<GradientTexture>` | 颜色渐变 |
| 加载状态 | `<Loader>` | 3D 场景加载指示器 |
| 性能监控 | `<Perf>` | FPS/GPU 监控面板 |

#### 动画方案

| 动画类型 | 技术 | 实现 |
|---|---|---|
| **数据流粒子** | Three.js Points + GLSL shader | 发光粒子沿贝塞尔路径移动 |
| **相机飞越** | GSAP + R3F useFrame | 从 Embedding 飞到 LM Head |
| **层展开/折叠** | react-spring (@react-three/drei) | 弹簧物理动画 |
| **MoE 专家激活** | InstancedMesh color attribute | 8 个活跃专家发光 |
| **Tensor shape 变形** | GSAP timeline | 3D 块 scale tween |
| **MLA 漏斗粒子** | Points + 路径动画 | 粒子通过漏斗几何体 |
| **滚动驱动推理** | GSAP ScrollTrigger | 滚动→相机移动→步骤高亮 |

#### 视觉风格

> 详见 **6.6 现代感视觉设计规范**。核心要点:

```
背景色: #06060f (近黑深蓝，非纯黑)
环境光: HemisphereLight(#1a1a3e, #000000, 0.3)
主方向光: 0.8 强度 + 蓝色补光 0.2
后处理: 仅 Bloom（Mac 集成 GPU 优化，去 Vignette/ChromaticAberration/Noise）
色调映射: ACES Filmic

材质: MeshStandardMaterial（Mac 集成 GPU 优化，去 clearcoat/transmission）
UI: Glassmorphism 毛玻璃 + blur(20px)
字体: Inter (UI) + JetBrains Mono (数值)
动画: cubic-bezier(0.4, 0, 0.2, 1) + 弹簧物理
加载: 线框→充能→实体→连线→标签→Bloom 六段序列
粒子: 默认 3K（Mac 集成 GPU 上限）
渲染: frameloop="demand"（按需渲染，idle 零 GPU）
```

> **样式隔离**: Tailwind CSS 仅用于 HTML UI 层（侧边栏、面板、工具栏）。Canvas/WebGL 内部不依赖 Tailwind，3D 场景的颜色/材质通过 Three.js Material 系统管理。

---

[← 返回目录](README.md)