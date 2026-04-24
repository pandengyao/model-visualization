# 六、2D/3D 双模式可视化设计

> [HF Model Visualizer](README.md) 技术设计文档 — 章六

## 目录

- [6.0 双模式架构](#60-双模式架构)
  - [模式切换按钮](#模式切换按钮)
  - [共享数据层 vs 独立渲染层](#共享数据层-vs-独立渲染层)
- [6.0.0 WebGL 降级策略](#600-webgl-降级策略)
- [6.0.1 2D 模式设计（借鉴 Netron + Model Explorer）](#601-2d-模式设计借鉴-netron--model-explorer)
  - [渲染技术](#渲染技术)
  - [2D 节点设计](#2d-节点设计)
  - [2D 数据流动画](#2d-数据流动画)
  - [2D 专属功能](#2d-专属功能)
- [6.1 3D 模式设计（主体，保持原有设计）](#61-3d-模式设计主体保持原有设计)
- [6.1.1 深度借鉴 LLM Viz 的 3D 实现细节](#611-深度借鉴-llm-viz-的-3d-实现细节)
  - [A. 3D 空间隐喻（核心布局哲学）](#a-3d-空间隐喻核心布局哲学)
  - [B. 残差流 (Residual Stream) 可视化](#b-残差流-residual-stream-可视化)
  - [C. 多 Pass 渲染 + 选择性 Bloom](#c-多-pass-渲染--选择性-bloom)
  - [D. 多尺度网格线 + LOD 系统](#d-多尺度网格线--lod-系统)
  - [E. 块体光照模型 (Block Lighting)](#e-块体光照模型-block-lighting)
  - [F. 线程连线渲染 (Thread Rendering)](#f-线程连线渲染-thread-rendering)
  - [G. 弹簧物理相机 (Spring Camera)](#g-弹簧物理相机-spring-camera)
  - [H. 多列布局换行](#h-多列布局换行)
- [6.2 交互设计](#62-交互设计)
- [6.3 MoE 3D 专家网格](#63-moe-3d-专家网格)
- [6.4 MLA 3D 漏斗](#64-mla-3d-漏斗)
- [6.5 端到端 3D 数据流可视化](#65-端到端-3d-数据流可视化)
  - [阶段一：数据预处理（Input → Embedding）](#阶段一数据预处理input--embedding)
  - [阶段二：模型内部数据流转（Decoder Layers）](#阶段二模型内部数据流转decoder-layers)
  - [阶段三：最终输出可视化（Norm → LM Head → Tokens）](#阶段三最终输出可视化norm--lm-head--tokens)
  - [自动动画播放模式](#自动动画播放模式)
  - [Tensor Shape 3D 实时标注](#tensor-shape-3d-实时标注)
- [6.6 现代感视觉设计规范](#66-现代感视觉设计规范)
  - [设计语言：「深空科技」](#设计语言深空科技)
  - [色彩系统](#色彩系统)
  - [材质与光影](#材质与光影)
  - [UI 组件现代感规范](#ui-组件现代感规范)
  - [动画与过渡](#动画与过渡)

### 6.0 双模式架构

```
┌──────────────────────────────────────────────────────────┐
│  工具栏                                                    │
│  [🔍 搜索] [📐 2D ◀▶ 3D] [▶ Guided Tour] [⚙ 设置]       │
│                                                            │
│  ┌──────────────────────────────┬───────────────┐  │
│  │                                      │               │  │
│  │     可视化区域                        │   侧边栏      │  │
│  │     (2D: SVG Canvas)                 │   (五视图)    │  │
│  │     (3D: R3F Canvas)                 │               │  │
│  │                                      │   属性面板     │  │
│  │                                      │   步骤列表     │  │
│  │                                      │   参数统计     │  │
│  └──────────────────────────────┴───────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  时间轴（Guided Tour 模式）                            │  │
│  │  ◀ ⏸ ▶  ───●──────────────── 1:30 / 5:00  [1x]      │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

#### 模式切换按钮

```
┌─────────────────────────────┐
│  📐 2D  │  🧊 3D  │         │  ← 分段控制器(Segmented Control)
└─────────────────────────────┘

切换行为:
  - 2D → 3D: 当前展开状态保留，2D 节点位置映射到 3D 坐标
  - 3D → 2D: 当前展开状态保留，3D 布局投影到 2D DAG
  - 过渡动画: 0.5s ease-out，节点从 2D 位置飞到 3D 位置（或反向）
  - URL 参数: ?mode=2d / ?mode=3d
  - 默认: 3D 模式
```

#### 共享数据层 vs 独立渲染层

```
┌─────────────────────────────────────────┐
│  共享数据层 (Zustand Store)              │
│                                          │
│  - modelData: API 返回的完整 JSON        │
│  - expandedNodes: Set<string>            │
│  - selectedNode: string | null           │
│  - searchQuery: string                   │
│  - overlayData: NDP 叠加数据             │
│  - tourState: { step, playing, speed }   │
│  - bookmarks: SavedState[9]              │
│  - sidebarView: 'model'|'node'|'tensor'  │
└──────────┬──────────────┬───────────────┘
           │              │
    ┌──────┴──────┐  ┌───┴───────────┐
    │   2D 渲染    │  │   3D 渲染     │
    │              │  │               │
    │  SVG/Canvas  │  │  R3F + Three  │
    │  + D3 Force  │  │  + Drei       │
    │  + dagre     │  │  + GSAP       │
    │              │  │               │
    │  搜索高亮    │  │  Bloom 高亮   │
    │  SVG 导出    │  │  PNG 导出     │
    │  分屏对比    │  │  粒子数据流   │
    └─────────────┘  └───────────────┘
```

### 6.0.0 WebGL 降级策略

不是所有用户都有支持 WebGL 的浏览器或足够的 GPU 性能。必须提供优雅降级：

```
Level 0 (默认): 完整 3D + Bloom + 粒子数据流
  条件: WebGL2 + 独立/集成 GPU
  体验: 全功能 3D 可视化

Level 1 (简化 3D): 3D 几何体 + 无后处理 + 无粒子
  条件: WebGL1 或 GPU 检测到低性能
  体验: 可旋转的 3D 架构树，无动画效果

Level 2 (2D SVG): dagre 布局 + SVG 渲染
  条件: 无 WebGL 或用户主动选择
  体验: 经典 2D DAG 图，仍可交互展开/折叠

Level 3 (纯文本): ASCII 结构树 + 参数表格
  条件: 极端环境（无头浏览器、屏幕阅读器）
  体验: tree_text + Markdown 表格，零 JS 依赖
```

**检测逻辑** (首次加载时执行):
1. `document.createElement('canvas').getContext('webgl2')` → Level 0/1
2. 无 WebGL2 → 尝试 `webgl` → Level 1
3. 均失败 → Level 2
4. GPU 性能检测: 渲染 100 个 Instanced 立方体，FPS < 15 → 降级一级
5. 用户可在设置中手动覆盖降级级别

### 6.0.1 2D 模式设计（借鉴 Netron + Model Explorer）

#### 渲染技术

```
SVG 层级（6 层，借鉴 Netron）:
  Layer 0: 背景网格（极淡参考线）
  Layer 1: 层组矩形（GroupNode 边框）
  Layer 2: 边路径（贝塞尔曲线 + 箭头）
  Layer 3: 边命中区域（不可见宽路径，用于点击检测）
  Layer 4: 节点（圆角矩形 + 颜色编码）
  Layer 5: 标签 + 叠加数据条

布局算法:
  dagre (Sugiyama 层级布局)
    - nodesep: 20
    - ranksep: 50
    - edgesep: 20
    - >3000 节点自动降级到 longest-path ranker
    - Web Worker 中计算布局
```

#### 2D 节点设计

```
┌─────────────────────────────────────┐
│ ▼ MLA Attention                [📋] │  ← 头部: 类型图标 + 名称 + 展开/折叠按钮
├─────────────────────────────────────┤
│ 类型: DeepseekV3Attention          │  ← 属性区
│ 参数: 101.12M (9.8%)              │
│ ████████████████░░░░  101.12M      │  ← NDP 数据条
├─────────────────────────────────────┤
│ 输入: (B, S, 7168)                 │  ← 端口区
│ 输出: (B, S, 7168)                 │
└─────────────────────────────────────┘

颜色编码（同 3D）:
  Attention: #4a9eff 蓝色边框 + 浅蓝底
  MLP:       #34d399 绿色边框 + 浅绿底
  MoE:       #f59e0b 橙色边框 + 浅橙底
  Norm:      #fbbf24 黄色边框 + 浅黄底
  Embedding: #a78bfa 紫色边框 + 浅紫底
  Vision:    #22d3ee 青色边框 + 浅青底
```

#### 2D 数据流动画

```
2D 模式的数据流不使用粒子，而是使用:
  - SVG 路径动画: strokeDashoffset 驱动的"线条绘制"效果
  - 渐变路径: 线性渐变从源节点色到目标节点色
  - 脉冲效果: 边上的亮点沿路径移动（SVG <animateMotion>）
  - 残差连接: 虚线 + 半透明
  - Hover 高亮: 选中节点的所有输入边=绿色，输出边=红色
```

#### 2D 专属功能

| 功能 | 说明 |
|---|---|
| **SVG 导出** | 内联 CSS 的自包含 SVG，可直接用于论文 |
| **分屏对比** | 两个模型并排 SVG 面板，同步导航 |
| **层级颜色条** | 折叠层底部的子节点颜色分布条 |
| **平铺全部层** | 移除所有分组，展示原始节点连接 |
| **弹出层窗口** | 多个可移动面板同时查看不同层内部 |
| **边叠加层** | 自定义 JSON 定义的额外边集合 |

### 6.1 3D 模式设计（主体，保持原有设计）

```
相机默认视角：45° 俯视，距离中心 15 单位

纵向堆叠布局（Y 轴向下为推理方向）：

       ┌──────────────┐
       │  Embedding   │ ← 紫色 RoundedBox
       │ (163840×7168)│
       └──────┬───────┘
              │ 发光连线
       ┌──────┴───────┐
       │  Layer 0     │ ← 展开后看到:
       │  (Dense)     │   ┌──────┐ ┌──────┐
       └──────┬───────┘   │ MLA  │ │ MLP  │
              │           │ (蓝) │ │ (绿) │
              │           └──────┘ └──────┘
       ┌──────┴──────────────────────────┐
       │  Layers 1-60 (MoE × 60)        │ ← 折叠显示为单块
       │  点击展开 ↓                      │
       │  ┌─────────┐  ┌────────────────┐│
       │  │  MLA    │  │  MoE Block     ││
       │  │ Attn    │  │ ┌──────────────┐││
       │  │  (蓝)   │  │ │  24×16 专家  │││ ← InstancedMesh
       │  │         │  │ │  网格 (384)  │││
       │  └─────────┘  │ └──────────────┘││
       │               │ + 共享专家(大块) ││
       │               └────────────────┘│
       └──────┬──────────────────────────┘
              │
       ┌──────┴───────┐
       │  RMSNorm     │ ← 黄色
       └──────┬───────┘
       ┌──────┴───────┐
       │  LM Head     │ ← 紫色
       │ (7168→163840)│
       └──────────────┘

多模态分支（X 轴偏移 -5）：

  ┌─────────────┐
  │ MoonViT     │ ← 青色
  │ (27层 ViT)  │
  └──────┬──────┘
  ┌──────┴──────┐
  │ PatchMerger │ ← 青色
  │ (投影器)    │
  └──────┬──────┘
         │ 粒子流汇入主干
         └──────────→ Embedding 之后
```

### 6.1.1 深度借鉴 LLM Viz 的 3D 实现细节

LLM Viz 是目前 3D 模型可视化领域的标杆，以下逐项深度借鉴其核心技术并适配到 R3F 体系：

#### A. 3D 空间隐喻（核心布局哲学）

```
LLM Viz 原始设计:
  - Y 轴 (向下): 推理方向 (token 从上到下穿越各层)
  - X 轴 (水平): 计算分支 (Attention 向左, MLP 向右)
  - Z 轴 (深度): 并行结构 (多个注意力头沿 Z 轴展开)

我们的适配 (支持更复杂的架构):
  - Y 轴 (向下): 推理方向 — 保持不变
  - X 轴 (水平):
    - 左: Attention 分支
    - 中: 残差主干 (Residual Stream) — 直线垂直主干
    - 右: MLP/MoE 分支
    - 远左: 多模态分支 (Vision/Audio Encoder)
  - Z 轴 (深度):
    - 注意力头 (64 heads) 沿 Z 轴排列
    - MoE 专家 (384 experts) 在 XZ 平面形成网格
    - MLA 漏斗在 Z 方向展示压缩/解压

空间语义:
  - 越靠近 Y 轴主干 = 越核心的计算
  - X 距离主干越远 = 越专项的计算
  - Z 深度越大 = 并行度越高
```

#### B. 残差流 (Residual Stream) 可视化

```
借鉴 LLM Viz 的核心创新: 将残差连接渲染为贯穿整个模型的"垂直主干"

实现方式 (R3F):
  <group name="residual-stream">
    // 半透明发光圆柱体，从 Embedding 贯穿到 LM Head
    <Cylinder args={[0.15, 0.15, totalHeight]} position={[0, -totalHeight/2, 0]}>
      <meshPhysicalMaterial
        color="#818cf8"
        transparent opacity={0.3}
        emissive="#818cf8" emissiveIntensity={0.15}
        transmission={0.4}
      />
    </Cylinder>

    // 每一层在主干上标记一个节点（发光环）
    {layers.map((_, i) => (
      <Ring args={[0.2, 0.35, 32]} position={[0, -layerY(i), 0]} rotation={[Math.PI/2, 0, 0]}>
        <meshBasicMaterial color="#818cf8" transparent opacity={0.5} />
      </Ring>
    ))}
  </group>

视觉效果:
  - Attention/MLP 计算作为"分支"从主干伸出
  - 计算完成后结果"汇回"主干
  - 整个推理过程 = 信号沿主干向下传播，每层被分支"加工"一次
  - 残差主干始终微微发光，表示信息持续流动
```

#### C. 多 Pass 渲染 + 选择性 Bloom

```
借鉴 LLM Viz 的 8 Pass 渲染管线，简化为 R3F 4 层:

Pass 1 — 几何层 (Opaque):
  层块 RoundedBox + 连线 + 残差主干
  渲染顺序: renderOrder={0}
  Bloom 层: layers.set(0) — 不参与 Bloom

Pass 2 — 发光层 (Emissive):
  活跃模块 + 数据流粒子 + MoE 激活专家
  Bloom 层: layers.set(1) — 参与 Bloom
  通过 <SelectiveBloom> 只对这一层施加辉光

Pass 3 — 标签层 (Overlay):
  <Html> overlay 信息面板
  <Text> 3D 文字标签
  渲染顺序: renderOrder={10}

Pass 4 — 后处理:
  // 默认 (Mac 集成 GPU): 仅 Bloom
  <EffectComposer>
    <Bloom luminanceThreshold={0.8} intensity={1.5} radius={0.4} />
  </EffectComposer>

  // 高端 GPU 自动升级: + Vignette + ChromaticAberration + Noise
  // （通过 renderer.capabilities 检测，v1.1 实现三档自动降级）

关键: 按需渲染 (borrowing LLM Viz):
  <Canvas frameloop="demand" ...>
  // 仅在状态变化时重绘，idle 时 0 GPU 负载
  // 通过 invalidate() 手动触发重绘
```

#### D. 多尺度网格线 + LOD 系统

```
借鉴 LLM Viz 的多尺度 GLSL 网格:

3D 地面参考网格:
  <gridHelper args={[100, 100]} position={[0, groundY, 0]}>
    // 主网格线 (每 16 格): opacity 0.1
    // 次网格线 (每 1 格): opacity 0.03
    // 通过相机距离动态调整可见性

双重 LOD 策略 (借鉴 LLM Viz):

  LOD Level 0 (远距离 — 相机距离 > 30):
    - 所有层块合并为单色条
    - 隐藏所有文字标签
    - 60 个 MoE 层显示为一个带数字 "×60" 的块
    - 连线简化为直线
    - >12 层自动分列排列（防止过长单列，借鉴 LLM Viz）

  LOD Level 1 (中距离 — 相机距离 10-30):
    - 每层显示为独立块体 + 颜色编码
    - 显示层编号 + 类型名称
    - MoE 网格显示为单个块（标注 "384 experts"）
    - 连线显示箭头方向

  LOD Level 2 (近距离 — 相机距离 < 10):
    - 展开层内部结构（Attention + MLP/MoE）
    - 显示完整参数标注
    - MoE 展开为 24×16 网格
    - MLA 显示漏斗结构
    - 粒子数据流可见

  LOD 过渡:
    - 使用 Drei <LOD> 组件 + 自定义距离阈值
    - 过渡时材质 opacity 渐变 (300ms)
    - 远距离节点颜色混向背景色 (借鉴 Model Explorer)
```

#### E. 块体光照模型 (Block Lighting)

```
借鉴 LLM Viz 的块体专用光照:

每个 3D 层块的 6 个面使用不同亮度:
  顶面: 1.0 × baseColor (最亮，接收主方向光)
  前面: 0.85 × baseColor
  侧面: 0.70 × baseColor
  底面: 0.50 × baseColor (最暗)

R3F 实现:
  <RoundedBox args={[width, height, depth]} radius={0.1} smoothness={4}>
    <meshStandardMaterial
      color={moduleColor}
      metalness={0.15}
      roughness={0.25}
      // 默认 MeshStandardMaterial (Mac GPU)
      // 高端 GPU 升级为 MeshPhysicalMaterial + clearcoat + transmission
      envMapIntensity={0.5}
    />
  </RoundedBox>

活跃状态 (Hover/Selected):
  // 添加 emissive + 轮廓发光
  emissive={moduleColor}
  emissiveIntensity={isActive ? 0.4 : isHover ? 0.2 : 0}
  // 外加一个稍大的透明外壳 (glow shell)
  <RoundedBox args={[width+0.02, height+0.02, depth+0.02]}>
    <meshBasicMaterial
      color={moduleColor}
      transparent opacity={isActive ? 0.15 : 0}
      side={BackSide}  // 只渲染内表面
    />
  </RoundedBox>
```

#### F. 线程连线渲染 (Thread Rendering)

```
借鉴 LLM Viz 的数据连线风格:

主数据流连线:
  <QuadraticBezierLine
    start={sourcePos} end={targetPos}
    mid={midPoint}  // 微微弯曲，避免与块体重叠
    color={flowColor}
    lineWidth={2}
    transparent opacity={0.6}
    // 加上发光效果
    onUpdate={(line) => { line.material.emissive = flowColor; line.material.emissiveIntensity = 0.3 }}
  />

残差连接线 (borrowing LLM Viz 的虚线残差):
  <Line
    points={[layerTop, layerBottom]}
    color="#818cf8"
    lineWidth={1}
    dashed
    dashSize={0.1}
    gapSize={0.05}
    transparent opacity={0.4}
  />

分支连线 (Attention ← 主干 → MLP):
  水平弯曲贝塞尔，表示数据从主干"分流"到计算分支

汇合连线 (计算结果 → + residual → 主干):
  箭头指向主干，表示结果与残差相加

连线动画:
  - 生长效果: 新连线从起点"画"向终点 (GSAP drawSVG 概念映射到 3D)
  - 数据流脉冲: 周期性亮度变化沿连线传播 (shader uniform)
```

#### G. 弹簧物理相机 (Spring Camera)

```
借鉴 LLM Viz 的临界阻尼相机系统:

参数 (直接借鉴):
  mass: 1
  tension: 170    // 弹簧刚度
  friction: 2 * sqrt(1 * 170) ≈ 26  // 临界阻尼
  过渡时间: ~667ms

R3F 实现:
  import { useSpring } from '@react-spring/three'
  import { CameraControls } from '@react-three/drei'

  const cameraRef = useRef<CameraControls>()

  // 点击模块时飞向目标
  const flyTo = (targetPos: Vector3, targetLookAt: Vector3) => {
    cameraRef.current?.setLookAt(
      targetPos.x, targetPos.y + 3, targetPos.z + 8,  // 相机位置
      targetLookAt.x, targetLookAt.y, targetLookAt.z,  // 看向目标
      true  // 启用平滑过渡
    )
  }

  // Guided Tour 中的相机轨迹
  const tourPath = [
    { pos: [0, 5, 20], lookAt: [0, 0, 0], duration: 2 },      // 全景
    { pos: [0, 2, 8],  lookAt: [0, -1, 0], duration: 1.5 },   // Embedding
    { pos: [-2, -5, 6], lookAt: [-2, -6, 0], duration: 2 },    // Layer 0
    { pos: [3, -8, 5],  lookAt: [3, -9, 0], duration: 2 },     // MoE 网格
    // ...
  ]

球坐标系统 (borrowing LLM Viz):
  // 用户拖拽时使用球坐标而非笛卡尔坐标
  // 保证旋转始终以模型中心为轴心
  // OrbitControls 已内置此行为
```

#### H. 多列布局换行

```
借鉴 LLM Viz 的分列策略:

当 num_layers > 12 时自动分列:

  单列 (≤12 层):        双列 (13-24 层):       三列 (25-36 层):
  ┌─────┐               ┌─────┐ ┌─────┐       ┌─────┐ ┌─────┐ ┌─────┐
  │  L0 │               │  L0 │ │ L13 │       │  L0 │ │ L13 │ │ L25 │
  │  L1 │               │  L1 │ │ L14 │       │  L1 │ │ L14 │ │ L26 │
  │ ... │               │ ... │ │ ... │       │ ... │ │ ... │ │ ... │
  │ L11 │               │ L12 │ │ L24 │       │ L12 │ │ L24 │ │ L35 │
  └─────┘               └─────┘ └─────┘       └─────┘ └─────┘ └─────┘

  列间有连线: 第一列底部 → 第二列顶部（弯曲贝塞尔）

  Kimi-K2.6 (61 层) → 4 列:
  列1: L0-L15 | 列2: L16-L30 | 列3: L31-L45 | 列4: L46-L60
  每列间距: X 偏移 6 单位
  列间连线: 从上一列底部弯向下一列顶部

  相同层自动检测 (borrowing Model Explorer):
  - 选中 Layer 5 (MoE) → 自动虚线高亮 Layer 1-60 所有同结构层
  - Layer 0 (Dense) 不被高亮（结构不同）
  - 折叠显示: "Layer 1-60 (×60, 相同结构)" 带展开按钮
```

### 6.2 交互设计

| 操作 | 效果 |
|---|---|
| **鼠标拖拽** | 旋转 3D 场景（OrbitControls） |
| **滚轮** | 缩放（平滑过渡） |
| **点击层块** | 展开子模块（GSAP 动画：子块从中心向外扩散） |
| **再次点击** | 折叠回去 |
| **Hover 层块** | 高亮发光 + Html overlay 显示简要信息 |
| **点击 MoE 块** | 展开 384 专家网格 |
| **点击单个专家** | Html overlay 显示专家参数详情 |
| **播放按钮** | 相机飞越 + 数据流粒子动画 |
| **时间轴拖动** | 跳到推理的任意步骤 |
| **侧边栏** | 模型信息、参数统计、配置详情 |

### 6.3 MoE 3D 专家网格

```
384 个专家排列为 24×16 3D 网格：

  InstancedMesh (单次 draw call)
  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤  × 24 行
  ├─┼─┼─╋═╋─┼─┼─╋═╋─┼─┼─╋═╋─┼─┼─╋═╋  ← 8 个活跃专家(亮色+bloom)
  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

  活跃专家: 亮色 + emissive + Bloom
  非活跃:   暗色半透明 (opacity: 0.2)
  共享专家: 独立大块，始终亮 (旁边标注 "Always Active")
```

### 6.4 MLA 3D 漏斗

```
3D 漏斗几何体（CylinderGeometry 变形）：

  ┌────────────────────┐  ← 宽 (7168)
  │   kv_a_proj_with   │
  │       _mqa         │
  └────────┬───────────┘
           │
     ┌─────┴─────┐        ← 窄 (576 = 512 + 64)
     │ kv_latent  │          "KV Cache 只需缓存这里"
     │ + k_rope   │          ← 黄色高亮标注
     └─────┬─────┘
           │
  ┌────────┴───────────┐  ← 宽 (16384)
  │    kv_b_proj       │
  │ k_nope(128) v(128) │
  └────────────────────┘

  粒子动画：发光点从宽口进入 → 挤压通过窄口 → 从另一端宽口出来
  标注："28.4:1 压缩比"
```

### 6.5 端到端 3D 数据流可视化

整个推理过程分为三个阶段，每个阶段都有独立的 3D 可视化和自动动画：

#### 阶段一：数据预处理（Input → Embedding）

```
═══════════════════ 数据入口区域（场景顶部） ═══════════════════

  ┌──────────────────────────────────────────────┐
  │           📝 输入文本 (3D 浮动文字)           │
  │   "Hello, how are you today?"                │
  │   ↓                                          │
  │   ┌──────────────────────────────┐            │
  │   │  Tokenizer (3D 切割动画)     │ ← 白色半透明
  │   │  文本 → ["Hello", ",", ...]  │
  │   │  动画: 文字被"切割刀"分段    │
  │   └──────┬───────────────────────┘            │
  │          │                                    │
  │   ┌──────┴──────────────────────────────────┐ │
  │   │  Token IDs (3D 数字方块阵列)             │ │
  │   │  [15496, 11, 703, 389, 345, 1909, 30]   │ │
  │   │  每个 token 是一个小方块，上面显示 ID     │ │
  │   │  动画: 方块从文字位置飞入排列            │ │
  │   └──────┬──────────────────────────────────┘ │
  │          │                                    │
  │   ┌──────┴──────────────────────────────────┐ │
  │   │  Embedding Lookup (3D 矩阵切片动画)     │ │
  │   │  163840 × 7168 大矩阵 (紫色半透明)      │ │
  │   │  动画: Token ID 方块射入矩阵 →           │ │
  │   │  对应行高亮发光 → 行向量"抽出"为         │ │
  │   │  → (B, 7, 7168) 3D 张量块                │ │
  │   └──────┬──────────────────────────────────┘ │
  │          │                                    │
  │   ┌──────┴──────────────────────────────────┐ │
  │   │  + Position Embedding (YaRN RoPE)       │ │
  │   │  动画: 正弦波形 3D 表面叠加到向量上      │ │
  │   │  旋转螺旋效果表示旋转位置编码            │ │
  │   └──────┬──────────────────────────────────┘ │
  └──────────│────────────────────────────────────┘
             ↓
    inputs_embeds: (B, S, 7168) → 进入主干

  ═══ 多模态分支（如有图像/视频输入）═══

  ┌──────────────────────────────────┐
  │  🖼️ 输入图像 (3D 纹理平面)       │
  │  动画: 图像飘入场景               │
  │  ↓                                │
  │  Patch 切割 (16×16 网格动画)      │
  │  动画: 图像被网格线"切割"为 patch  │
  │  ↓                                │
  │  MoonViT 编码 (27 层动画)         │
  │  ↓                                │
  │  PatchMerger (2×2 合并动画)       │
  │  每 4 个 patch 合并为 1 个         │
  │  ↓                                │
  │  Linear 投影 (4608 → 7168)        │
  │  颜色从青色渐变为紫色             │
  └──────────┬───────────────────────┘
             │ 粒子流汇入主干
             └──→ masked_scatter 到 inputs_embeds
```

#### 阶段二：模型内部数据流转（Decoder Layers）

```
═══════════════════ 主干推理区域（场景中部） ═══════════════════

  inputs_embeds: (B, S, 7168) → 3D 张量块（紫色）
             │
             │ 数据流粒子开始
             ▼
  ┌─────────────────────────────────────────────┐
  │  Layer 0 (Dense) — 展开内部:                 │
  │                                              │
  │  ┌─ RMSNorm ──────────────────────────────┐  │
  │  │  动画: 张量块"脉动"表示归一化           │  │
  │  └─────────┬──────────────────────────────┘  │
  │            ↓                                  │
  │  ┌─ MLA Attention (蓝色区域) ──────────────┐  │
  │  │                                          │  │
  │  │  Q 路径:                                 │  │
  │  │  (B,S,7168) →[下投影]→ (B,S,1536)        │  │
  │  │  →[RMSNorm]→ [上投影]→ (B,S,64,192)      │  │
  │  │  → 分裂为 nope(128) + rope(64)            │  │
  │  │  → 螺旋动画: RoPE 旋转                    │  │
  │  │                                          │  │
  │  │  KV 路径 (MLA 漏斗 3D 动画):              │  │
  │  │  (B,S,7168) →[压缩]→ (B,S,576)           │  │
  │  │  黄色高亮: "KV Cache 576 dim"             │  │
  │  │  →[解压]→ (B,S,64,256)                    │  │
  │  │                                          │  │
  │  │  Attention 计算:                          │  │
  │  │  Q×K^T → Softmax → ×V                    │  │
  │  │  动画: 注意力权重热力图（悬浮 2D 面板）    │  │
  │  │                                          │  │
  │  │  → o_proj → (B, S, 7168)                  │  │
  │  └─────────┬────────────────────────────────┘  │
  │            ↓ + residual (虚线回路)              │
  │  ┌─ RMSNorm ──────────────────────────────┐  │
  │  └─────────┬──────────────────────────────┘  │
  │            ↓                                  │
  │  ┌─ Dense MLP (绿色区域) ─────────────────┐  │
  │  │  gate(7168→18432) → SiLU               │  │
  │  │  up(7168→18432)                         │  │
  │  │  → 张量块"膨胀"动画 (7168 → 18432)     │  │
  │  │  → SiLU × up → down(18432→7168)        │  │
  │  │  → 张量块"收缩"回 7168                  │  │
  │  └─────────┬──────────────────────────────┘  │
  │            ↓ + residual                       │
  └────────────│──────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────────────────┐
  │  Layers 1-60 (MoE × 60) — 折叠/展开:                │
  │                                                      │
  │  折叠态: 60 层堆叠为一个大块，粒子快速穿过            │
  │  展开单层: 同 Layer 0 的 Attention + MoE 替代 MLP     │
  │                                                      │
  │  ┌─ MoE 路由动画 ────────────────────────────────┐   │
  │  │  (B,S,7168) → Gate Linear → sigmoid 评分       │   │
  │  │  动画: 384 维评分条形图实时"生长"               │   │
  │  │  → Top-8 选择: 8 个专家发光点亮                 │   │
  │  │  → 归一化 × 2.827                               │   │
  │  │                                                  │   │
  │  │  数据分流动画:                                   │   │
  │  │  粒子流从 Gate 分裂为 8 条支流                    │   │
  │  │  → 8 个活跃专家各接收一条支流                    │   │
  │  │  → 每个专家内部: SwiGLU 膨胀/收缩               │   │
  │  │  → 8 条支流汇合 + 加权求和                       │   │
  │  │  + 共享专家始终处理完整流（独立大块）             │   │
  │  │  → 两路合并输出                                  │   │
  │  └──────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────┘
```

#### 阶段三：最终输出可视化（Norm → LM Head → Tokens）

```
═══════════════════ 输出区域（场景底部） ═══════════════════

  ┌──────────────────────────────────────────────┐
  │  Final RMSNorm                                │
  │  动画: 张量块最后一次"脉动"归一化             │
  │  (B, S, 7168) → (B, S, 7168)                 │
  └──────┬───────────────────────────────────────┘
         │
  ┌──────┴───────────────────────────────────────┐
  │  LM Head (7168 → 163840)                     │
  │  3D 可视化: 张量块急剧"膨胀"                  │
  │  7168 维 → 163840 维（词表大小）              │
  │  动画: 窄长条扩展为宽巨幅面板                 │
  │  颜色: 从内部颜色渐变到紫色                   │
  └──────┬───────────────────────────────────────┘
         │
  ┌──────┴───────────────────────────────────────┐
  │  Logits → Softmax (概率分布可视化)            │
  │                                               │
  │  3D 柱状图 / 粒子喷泉:                        │
  │  ┌─────────────────────────────────────┐      │
  │  │  163840 个 token 概率                │      │
  │  │  Top-K 突出显示:                     │      │
  │  │                                      │      │
  │  │  ████████████ "you"     (p=0.32)    │      │
  │  │  ████████     "doing"   (p=0.18)    │      │
  │  │  ██████       "today"   (p=0.12)    │      │
  │  │  ████         "?"       (p=0.08)    │      │
  │  │  ███          "feeling" (p=0.06)    │      │
  │  │  ·····(其余概率极小，半透明)·····    │      │
  │  │                                      │      │
  │  │  3D 效果: Top 词汇的柱状条从平面     │      │
  │  │  "生长"出来，高度=概率，颜色=置信度  │      │
  │  │  最高概率的柱子发光 + Bloom           │      │
  │  └─────────────────────────────────────┘      │
  │                                               │
  │  采样动画:                                     │
  │  一个发光粒子从概率分布中"被选中"              │
  │  → 飞到输出文字区域                            │
  │  → 显示为生成的 token: "you"                   │
  └───────────────────────────────────────────────┘
         │
  ┌──────┴───────────────────────────────────────┐
  │  📤 输出文本 (3D 浮动文字)                    │
  │  "you" ← 新生成的 token 以打字机效果出现      │
  │                                               │
  │  自回归循环（可选动画）:                        │
  │  新 token 反馈到顶部 → 再次走完整流程          │
  │  粒子从底部回流到顶部，循环动画                │
  └───────────────────────────────────────────────┘
```

#### 自动动画播放模式

**"Guided Tour" 自动播放**（借鉴 LLM Viz 的 walkthrough 系统）：

```
时间轴控制条:  ◀ ⏸ ▶  ───●──────────────── 2:30 / 5:00

步骤编排（GSAP Master Timeline）:

  T=0s   相机俯瞰全貌 → 淡入标题 "端到端推理流程"
  T=3s   相机飞向输入区 → 文字输入动画 → Tokenizer 切割
  T=8s   Token ID 方块排列 → 飞入 Embedding 矩阵
  T=12s  相机跟随 → Embedding 向量"抽出" → 位置编码叠加
  T=16s  （多模态）相机切到视觉分支 → 图像 Patch 切割 → ViT 编码
  T=22s  视觉粒子汇入主干 → 相机回到主流
  T=25s  相机进入 Layer 0 → Attention 展开
  T=30s  MLA 漏斗动画 → Q/K/V 路径分离与合并
  T=38s  → Dense MLP 膨胀/收缩
  T=42s  相机进入 MoE 层 → Gate 评分动画 → 专家网格点亮
  T=48s  数据分流 → 8 专家并行处理 → 汇合
  T=55s  快进穿越 Layer 2-59（粒子加速流动）
  T=60s  Layer 60 结束 → Final Norm
  T=63s  LM Head 膨胀 → 概率分布"生长"
  T=68s  采样粒子飞出 → 输出 token 显示
  T=72s  自回归回路动画（粒子回流）
  T=75s  相机拉远 → 全景 → 结束

交互控制:
  - ⏸ 暂停：任何时刻可暂停，自由旋转/缩放探索
  - ◀▶ 跳转：点击时间轴跳到任意步骤
  - 🔄 速度：0.5x / 1x / 2x / 4x 播放速度
  - 📍 步骤列表：侧边栏显示所有步骤，点击跳转
```

#### Tensor Shape 3D 实时标注

```
数据流中每个节点旁边悬浮 3D 标签:

  ┌──────────────┐
  │ (B, 7, 7168) │ ← <Html> overlay，半透明背景
  │ 50,176 params│    跟随 3D 块移动
  └──────────────┘

张量形状变化时的 3D 变形动画:
  - 维度增加: 块"膨胀"（scale.x/y/z 按比例增长）
  - 维度减少: 块"收缩"
  - 维度分裂（如 Q 分 64 头）: 块"分裂"为多个小块
  - 维度合并（如 MoE 汇合）: 多个小块"融合"为一个

颜色映射值范围:
  - 高维度 → 更鲜艳 + 更大发光强度
  - 低维度（如压缩后的 576）→ 暗色 + 黄色标注 "压缩"
```

### 6.6 现代感视觉设计规范

#### 设计语言：「深空科技」

整体视觉风格追求**高端科技感 + 数据可视化美学**，参考 Apple Vision Pro UI、Bloomberg Terminal 现代化重设计、Stripe 产品页面的品质标准。

#### 色彩系统

```
═══ 背景层 ═══
主背景:      #06060f (近黑深蓝) — 不是纯黑，带微蓝色调
次背景:      #0d0d1a (深空蓝) — 卡片/面板背景
表面色:      #141428 (暗靛蓝) — 悬浮元素背景
边框:        rgba(255,255,255,0.06) — 极微弱白色边框

═══ 主色调 ═══
品牌蓝:      #3b82f6 → #60a5fa (渐变) — 主要交互色
辉光蓝:      #818cf8 — Bloom 效果主色

═══ 功能色彩（3D 场景内） ═══
Attention:   #4a9eff → #2563eb (蓝色系)
MLP/FFN:     #34d399 → #10b981 (翠绿系)
MoE Router:  #f59e0b → #f97316 (橙金系)
Norm:        #fbbf24 → #eab308 (明黄系)
Embedding:   #a78bfa → #8b5cf6 (紫色系)
Vision/Conv: #22d3ee → #06b6d4 (青色系)
输出/Logits: #f472b6 → #ec4899 (粉色系)

═══ 状态色 ═══
活跃/激活:   当前模块色 + emissive 0.4 + Bloom
非活跃:      模块色 × 0.15 opacity
Hover:       模块色 + emissive 0.2 + 白色边框 glow
数据流粒子:  当前阶段色 + additive blending

═══ 渐变 ═══
全局渐变:    从 #06060f (顶) 到 #0a0a2e (底) — 微妙深度感
卡片渐变:    从 rgba(255,255,255,0.03) 到 rgba(255,255,255,0.01)
```

#### 材质与光影

```
═══ 3D 材质 ═══

默认材质 (Mac 集成 GPU — v1.0 默认):
层块材质: MeshStandardMaterial {
  metalness: 0.15       // 轻微金属感
  roughness: 0.25       // 较光滑
  envMapIntensity: 0.5
}
// 注: 去 transmission/clearcoat，性能提升 2-3x

高端材质 (独立 GPU — 自动检测升级):
层块材质: MeshPhysicalMaterial {
  metalness: 0.15
  roughness: 0.25
  transmission: 0.15    // 微透明（看到内部粒子流动）
  thickness: 0.8
  clearcoat: 0.3        // 清漆层增加光泽
  clearcoatRoughness: 0.2
  envMapIntensity: 0.5
}

活跃层块: 上述 + {
  emissive: 模块颜色
  emissiveIntensity: 0.3
}

粒子材质: PointsMaterial + 自定义 shader {
  blending: AdditiveBlending
  transparent: true
  vertexColors: true
  size: 根据距离衰减
  // 片段着色器: 径向渐变圆点 + 发光尾迹
}

═══ 光照 ═══
环境光:     HemisphereLight(#1a1a3e, #000000, 0.3)
主方向光:   DirectionalLight(#ffffff, 0.8) position=[10, 15, 10]
补光:       DirectionalLight(#3b82f6, 0.2) position=[-5, 5, -5]  // 蓝色补光
点光源:     PointLight(#818cf8, 0.5) — 跟随活跃模块移动

═══ 后处理 ═══

默认 (Mac 集成 GPU — v1.0 默认):
Bloom: {
  threshold: 0.8
  intensity: 1.5
  radius: 0.4
}
// 仅 Bloom，去 Vignette/ChromaticAberration/Noise

高端 (独立 GPU — 自动检测升级):
Bloom: { threshold: 0.7, intensity: 2.0, radius: 0.5, luminanceSmoothing: 0.1 }
Vignette: { darkness: 0.4, offset: 0.3 }
ChromaticAberration: { offset: [0.0003, 0.0003] }
Noise: { opacity: 0.015 }
```

#### UI 组件现代感规范

```
═══ 信息面板 (glassmorphism 毛玻璃风格) ═══
background: rgba(13, 13, 26, 0.75)
backdrop-filter: blur(20px) saturate(150%)
border: 1px solid rgba(255, 255, 255, 0.06)
border-radius: 16px
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4)
padding: 24px

═══ 搜索框 ═══
background: rgba(20, 20, 40, 0.8)
border: 1px solid rgba(99, 102, 241, 0.2)
border-radius: 12px
transition: border-color 0.3s, box-shadow 0.3s
&:focus {
  border-color: rgba(99, 102, 241, 0.5)
  box-shadow: 0 0 20px rgba(99, 102, 241, 0.15)
}

═══ 徽章 (MoE / MLA / 量化等标签) ═══
background: linear-gradient(135deg, rgba(color, 0.15), rgba(color, 0.05))
border: 1px solid rgba(color, 0.3)
border-radius: 8px
font-size: 12px
font-weight: 500
letter-spacing: 0.05em
text-transform: uppercase

═══ 按钮 ═══
background: linear-gradient(135deg, #3b82f6, #6366f1)
border: none
border-radius: 10px
box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3)
transition: transform 0.2s, box-shadow 0.2s
&:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) }

═══ 数据表格 ═══
font-family: "JetBrains Mono", "SF Mono", monospace  // 数值用等宽字体
th { color: rgba(255,255,255,0.5); font-weight: 400; text-transform: uppercase; font-size: 11px }
td { color: rgba(255,255,255,0.9); font-variant-numeric: tabular-nums }
tr:hover { background: rgba(255,255,255,0.03) }

═══ 字体 ═══
标题: "Inter", sans-serif — font-weight: 600
正文: "Inter", sans-serif — font-weight: 400
数值/代码: "JetBrains Mono", monospace — font-weight: 400
3D 标签: troika-three-text (SDF 渲染) — "Inter" 字体
```

#### 动画与过渡

```
═══ 全局过渡曲线 ═══
默认 ease:    cubic-bezier(0.4, 0, 0.2, 1)    // Material Design standard
弹性 ease:    cubic-bezier(0.34, 1.56, 0.64, 1)  // 轻微弹跳
减速 ease:    cubic-bezier(0, 0, 0.2, 1)      // 进入动画

═══ 页面加载序列 ═══
1. 场景背景渐入 (0 → 0.3s)
2. 3D 模型骨架线框渐入 (0.2s → 0.8s)
3. 块体材质"充能"效果: 线框 → 半透明 → 实体 (0.5s → 1.5s)
4. 连线"生长"动画 (1.0s → 1.8s)
5. 标签和面板滑入 (1.5s → 2.0s)
6. Bloom 后处理渐入 (1.8s → 2.2s)

═══ 层块展开动画 ═══
持续时间: 600ms
编排:
  0ms:    父块开始收缩
  100ms:  子块从父块中心"爆出"
  200ms:  子块移向目标位置（弹簧物理）
  300ms:  连线"生长"连接子块
  400ms:  标签渐入
  600ms:  稳定

═══ 微交互 ═══
Hover 光晕:     200ms ease-out, emissive 从 0 → 0.2
点击涟漪:       300ms, 从点击点向外扩散的半透明环
进度条发光:     持续脉动, opacity 0.6 → 1.0 → 0.6
粒子尾迹:       每个粒子后面 3-5 个渐隐副本
```

---

[← 返回目录](README.md)
