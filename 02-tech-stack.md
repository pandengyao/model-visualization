# 二、技术栈

> [HF Model Visualizer](README.md) 技术设计文档 — 章二

> **定位声明**：本项目为**内部教学/可视化工具**，非商业化产品。不做多租户、不做鉴权 SSO、不做多区域部署。技术选型的唯一目标是：**在单机 / 单部署节点上，呈现结构正确、视觉精美、教学深入的 3D 模型可视化**。

### 2.1 技术选型（现代化 3D 前端栈）

| 层 | 技术 | 版本（锁定） | 选型理由 |
|---|---|---|---|
| **前端框架** | Next.js (React 19) | `next ^15` / `react ^19` | SSR/ISR、API Routes、最大 React 生态 |
| **3D 声明式层** | React Three Fiber (R3F) | `@react-three/fiber ^9` | R3F v9 是唯一兼容 React 19 的版本（v8 不兼容） |
| **3D 组件库** | Drei | `@react-three/drei ^10` | Drei v10 与 R3F v9 对齐，60+ 3D 组件 |
| **3D 引擎** | Three.js | `three 0.169.0` **(exact)** | 钉精确版本，避免 minor API churn；升级需同步 R3F/Drei/postprocessing peer |
| **后处理** | postprocessing | `postprocessing ^6.35` | Drei `<EffectComposer>` 的 peer 依赖；Bloom/SSAO/ToneMapping 必需 |
| **HTML overlay 动画** | GSAP | `gsap ^3.13` | 2024 Webflow 收购后 **100% 免费**，含 Club 插件 MorphSVG/DrawSVG/SplitText，无商业限制 |
| **3D 场景动画** | react-spring (three) | `@react-spring/three ^9` | 弹簧物理，React 渲染管线原生集成 |
| **状态管理** | Zustand | `zustand ^5` | 轻量、无 Provider、R3F 社区事实标准 |
| **UI 样式** | Tailwind CSS | `tailwindcss ^3.4` **(锁 v3，不上 v4)** | v4 生态第三方 UI 库适配未完成，v1.0 稳定优先 |
| **数据映射** | D3 (scale/color/shape) | `d3-scale ^4` 等 | 颜色映射、比例尺、路径 |
| **布局算法** | dagre | `dagre ^0.8` | 有向图层级布局 |
| **类型系统** | TypeScript **strict mode** | — | 全项目强制 `strict: true`，无 `any` 逃生舱 |

> **版本兼容性的关键约束**：
> - `react 19` ⇔ `@react-three/fiber ^9` ⇔ `@react-three/drei ^10` 三者必须同步升级，任一 minor 偏离都会触发 peer 冲突。
> - `three` 钉精确版本 `0.169.0`，升级前需验证 R3F / Drei / postprocessing 三者 peer 范围。
> - `tailwindcss` 锁 `^3.4`，**禁止**升级到 v4（生态未就绪）。

### 2.2 视觉质感要求（对齐原则 2：精美 3D 风格）

本项目**拒绝"能跑就行"的审美**。任何 3D 场景必须满足以下硬性要求：

#### 2.2.1 材质与光照（PBR 强制）

- **材质基线**：必须使用 `MeshStandardMaterial` 或更高等级（`MeshPhysicalMaterial`），启用 metalness / roughness。
- **禁止**：`MeshBasicMaterial`（无光照响应）、`MeshLambertMaterial`（无高光）在主视觉元素上使用。
- **环境光**：必须使用 Drei `<Environment preset="city" | "studio" | "sunset" />` 提供 HDR 环境贴图（IBL），不允许只有 `ambientLight + directionalLight` 裸奔。
- **阴影**：主模块块体启用 `castShadow` + `receiveShadow`（集成 GPU 环境下可通过 `shadows="soft"` 控制质量）。

#### 2.2.2 后处理管线（必选 pass）

必须在 `<EffectComposer>` 中启用**至少以下三个 pass**：

| Pass | 用途 |
|---|---|
| `<Bloom>` | 发光元素（MoE 激活专家、数据流粒子）辉光效果 |
| `<SSAO>` | 屏幕空间环境光遮蔽，增强块体之间的体积感 |
| `<ToneMapping mode={ACESFilmicToneMapping}>` | 电影级色调映射，避免死白/死黑 |

#### 2.2.3 微交互（Motion Design）

任何可交互的 3D / HTML 元素必须具备：
- **Hover**：scale（1.0 → 1.05）+ 辉光（emissive intensity 提升）
- **Click**：ripple 波纹（3D 为 shader ring，HTML 为 Tailwind 过渡）
- **Focus**：可见 outline（键盘可达性）
- **过渡缓动**：`cubic-bezier(0.4, 0, 0.2, 1)` 或 react-spring 弹簧配置

#### 2.2.4 排版与配色

- **字体栈**：UI 使用系统级无衬线（`-apple-system, "Inter", "Segoe UI", sans-serif`），**数值/代码**使用等宽字体 `"JetBrains Mono", "Fira Code", monospace`。
- **配色**：**深色模式优先**，禁止使用 Tailwind 默认灰阶（`gray-100` … `gray-900`）作为主视觉。
  - 背景：`#06060f`（近黑深蓝）
  - 主色：深空蓝 / 霓虹青 / 品红辉光，具备"科技感"语义。
  - 所有颜色需登记在 `tokens.ts` 并通过 Tailwind `theme.extend.colors` 暴露，避免魔法字符串。

### 2.3 3D 组件映射（Drei / Three 对照表）

| 可视化抽象 | 3D 实现 | 说明 |
|---|---|---|
| **主模块块体** | `<Box>` / `<RoundedBox>` | Transformer Block、Embedding、LM Head |
| **连接线** | `<Line>` / `<Tube>` / `<QuadraticBezierLine>` | 层间数据流路径 |
| **Label（标签）** | Drei `<Html>` | 3D 场景中嵌入 React DOM，保持可交互 |
| **3D 文字** | Drei `<Text>` (troika SDF) | 大号模型名、分区标题 |
| **相机** | `<PerspectiveCamera>` | 主视角，FOV 45° |
| **控制器** | `<OrbitControls>` | 旋转 / 缩放 / 平移 |
| **后处理** | `<EffectComposer>` (postprocessing) | Bloom + SSAO + ACES ToneMapping |
| **环境光** | Drei `<Environment>` | HDR 预设，驱动 PBR 材质反射 |
| **MoE 专家批渲染** | Drei `<Instances>` / `<InstancedMesh>` | 单 draw call 渲染 384 专家 |
| **动画** | `useFrame` + `react-spring` + GSAP Timeline | 3D 内用 spring/useFrame，HTML overlay 用 GSAP |
| **加载指示** | Drei `<Loader>` | 场景资源就绪前占位 |

> **动画职责边界**：
> - **3D 场景内**：`useFrame`（逐帧）+ `react-spring`（弹簧物理）
> - **HTML overlay**：GSAP Timeline（侧边栏、面板过渡、时间轴 UI 控制器）
> - **相机过渡**：Drei `<CameraControls>` 的内置平滑插值
>
> 禁止 GSAP 直接操作 `three` 对象，避免与 React 渲染管线冲突。

### 2.4 为什么选 React + R3F 而非 Svelte + Threlte

| 维度 | React + R3F + Drei | Svelte + Threlte |
|---|---|---|
| 3D 生态成熟度 | **10/10** (Drei 9.6K stars) | 8/10 (60+ 组件) |
| 社区 / 文档 | **10/10** | 7/10 |
| 框架性能 | 7/10 (虚拟 DOM) | 10/10 |
| 示例 / demo 丰富度 | **10/10** | 7/10 |

决定因素：**3D 开发遇到问题时，R3F 社区的解法数量远超 Threlte**。对于 3D 密集型项目，生态成熟度 > 框架性能。

### 2.5 后端技术栈

| 层 | 技术 | 版本 | 选型理由 |
|---|---|---|---|
| **Web 框架** | FastAPI (async) | `>=0.115` | 全异步，HF Hub API 200–500ms 延迟下协程收益显著 |
| **ASGI 服务器** | uvicorn | `>=0.34` | FastAPI 官方推荐 |
| **Schema + Provenance** | Pydantic v2 | `>=2.6` | 所有 API I/O 类型强制，`ProvenanceField` 可追溯（对齐原则 9） |
| **模型加载** | transformers + torch | `transformers>=4.57`, `torch>=2.4` (CPU-only) | `AutoConfig` 标准化、`torch.device("meta")` 零权重加载**真实模型结构树** |
| **权重 header 解析** | safetensors | `>=0.4` | **仅读 header**，不加载权重数据，获取 shape / dtype |
| **L0 内存缓存** | cachetools | `>=5.5` | `TTLCache`，内部工具无需 Redis（原则 1：不做分布式） |
| **JSON 序列化** | **orjson** | `>=3.10` | **交互响应硬约束**（原则 5 例外）：大结构树序列化 5–10× 快于 stdlib |
| **HTTP 客户端** | httpx | `>=0.28` | 异步下载 config.json / model index |
| **HF 元数据** | huggingface_hub | `>=0.28` | 模型元信息 API |
| **配置文件读取** | PyYAML | `>=6.0` | 读取 `gpu-catalog.yaml`（GPU 规格目录） |
| **热更新推送** | WebSocket (FastAPI 原生) | — | `PATCH /config` 后向前端推送增量，驱动 3D 场景即时重渲 |

> **CPU-only + meta-device**：后端不需要 GPU，`torch.device("meta")` 仅构造结构不分配权重内存，Docker 镜像不含 CUDA（控制体积 ~1.3–2.0 GB）。

### 2.6 扩展点依赖（对齐原则 6：v1.0 架构最健壮）

v1.0 必须具备以下插件化扩展点，但**扩展机制本身拒绝过度工程**：

| 扩展点 | 注册机制 | 明确禁止 |
|---|---|---|
| **Adapter**（模型架构检测） | Python `Protocol` + **显式注册表** `ADAPTER_REGISTRY: dict[str, AdapterProtocol]` | `setuptools entry_points`、`pluggy`、任何自动发现机制 |
| **Template**（前端 3D 模板，含 Template G：LLaMA / LLaMA-MoE / DeepSeek-MoE 底线） | 显式 `import` + 数组注册 `TEMPLATES: Template[] = [llamaTemplate, deepseekMoeTemplate, ...]` | 动态 `import()`、glob 扫描、约定优于配置 |
| **AnimationLayer** | 显式注册表 | 同上 |
| **ParallelismStrategy** | 显式注册表 | 同上 |
| **MemoryEstimator** | 显式注册表 | 同上 |

> **设计原则**：注册表必须是**可 `grep` 出来的显式代码**。v1.0 Adapter/Template 数量 ~5–10 个，隐式插件机制带来的复杂度远大于收益。

### 2.7 依赖清单

**前端 `package.json`（关键部分）**:
```json
{
  "dependencies": {
    "next": "^15",
    "react": "^19",
    "react-dom": "^19",
    "@react-three/fiber": "^9",
    "@react-three/drei": "^10",
    "three": "0.169.0",
    "postprocessing": "^6.35",
    "gsap": "^3.13",
    "@react-spring/three": "^9",
    "zustand": "^5",
    "tailwindcss": "^3.4",
    "d3-scale": "^4",
    "d3-color": "^3",
    "d3-shape": "^3",
    "dagre": "^0.8"
  },
  "devDependencies": {
    "typescript": "^5.4",
    "vitest": "^1.6",
    "@vitest/coverage-v8": "^1.6",
    "@testing-library/react": "^16",
    "@testing-library/jest-dom": "^6",
    "playwright": "^1.47",
    "@playwright/test": "^1.47",
    "eslint": "^8.57",
    "eslint-config-next": "^15",
    "@typescript-eslint/parser": "^8",
    "@typescript-eslint/eslint-plugin": "^8",
    "prettier": "^3.3"
  }
}
```

TypeScript `tsconfig.json` 硬性：
```json
{ "compilerOptions": { "strict": true, "noUncheckedIndexedAccess": true } }
```

**后端 `pyproject.toml`（关键部分）**:
```toml
[project.dependencies]
fastapi = ">=0.115"
uvicorn = ">=0.34"
pydantic = ">=2.6"
httpx = ">=0.28"
huggingface_hub = ">=0.28"
cachetools = ">=5.5"
transformers = ">=4.57"
torch = ">=2.4"       # CPU-only wheel
safetensors = ">=0.4"
orjson = ">=3.10"
pyyaml = ">=6.0"
websockets = ">=13"

[project.optional-dependencies.dev]
pytest = ">=7.4"
pytest-asyncio = ">=0.23"
pytest-cov = ">=5"
httpx = ">=0.28"      # TestClient 复用
ruff = ">=0.6"        # linter + formatter
mypy = ">=1.11"
```

### 2.8 测试 / Lint 工具链（对齐原则 3/5/9）

**前端**：
- **Vitest**：单元 / 集成测试；覆盖 Adapter/Template/Registry 等纯函数与 React hooks。`vitest run --coverage` 在 CI 强制 ≥ 70%（v1.0 DoD）
- **@testing-library/react**：组件测试（Provenance 徽标、Config 编辑器、GPU 选择器）
- **Playwright**：E2E + 视觉回归；承载 07 验证矩阵 21 模型快照与 05 §5.4 视觉规范验证（36 张）
- **ESLint + `@typescript-eslint`**：强制 `strict` + `no-floating-promises` + `no-explicit-any`
- **Prettier**：代码风格统一（2 空格缩进、单引号、100 列）

**后端**：
- **pytest + pytest-asyncio**：SSE / WebSocket / Pipeline 异步测试
- **pytest-cov**：覆盖率 ≥ 80%（v1.0 DoD）
- **Ruff**：linter + formatter 合一（替代 flake8 / black / isort），规则集 `E, F, W, I, B, C4, UP, SIM`
- **mypy**：类型检查，strict 模式，pydantic plugin 启用

**CI 关卡（Phase 0 起必达）**：
```
前端：pnpm lint && pnpm typecheck && pnpm test:unit && pnpm test:e2e
后端：ruff check . && ruff format --check . && mypy src/ && pytest --cov=src --cov-fail-under=80
```

Docker 镜像预计 **1.3–2.0 GB**（多阶段构建 + 层缓存）。冷启动 ~30 s，通过预缓存 Top-100 热门模型缓解。v1.1 可评估 web（Node 静态）/ api（Python）镜像拆分。

---

[← 返回目录](README.md)
