# 附录 A：标杆产品技术深度分析与借鉴

> 本文档从主设计文档 `serialized-seeking-clock.md` Section 1.1.1 提取。
> 包含 Netron、Google Model Explorer、Transformer Explainer、LLM Viz 四个标杆产品的源码级分析。

---

## 开源仓库与许可证

| 项目 | GitHub 仓库 | License | 核心技术栈 | 关键源码入口 |
|---|---|---|---|---|
| **Netron** | `lutzroeder/netron` | **MIT** ✅ | Vanilla JS + Electron + 自研 dagre | `source/grapher.js` (图渲染), `source/dagre.js` (布局) |
| **Model Explorer** | `google-ai-edge/model-explorer` | **Apache-2.0** ✅ | Angular + Three.js 0.134 + RxJS | `src/ui/src/components/visualizer/webgl_renderer.ts` (主渲染), `webgl_rounded_rectangles.ts` (SDF 节点) |
| **Transformer Explainer** | `poloclub/transformer-explainer` | **MIT** ✅ | SvelteKit + D3 + GSAP + @xenova/transformers | `src/components/Attention.svelte`, `Sankey.svelte`, `AttentionMatrix.svelte` |
| **LLM Viz** | `bbycroft/llm-viz` | **⚠️ 无 License** (All Rights Reserved) | Next.js 13 + 自研 WebGL/WebGPU | `src/llm/GptModelLayout.ts` (3D 布局), `src/llm/render/modelRender.ts` (渲染管线) |

> **⚠️ LLM Viz 许可证风险**: LLM Viz 仓库无 LICENSE 文件，默认版权保留。**不可直接复制代码**，只能借鉴设计思路和算法原理。我们的实现基于 R3F/Three.js 全新编写，仅参考其空间布局哲学和视觉风格。

## 关键源文件参考指南

开发时可直接查阅以下文件学习具体实现：

**Netron** (`source/`):
- `grapher.js` — 节点/边渲染核心，包含双路径边命中检测实现
- `dagre.js` — 完整的 Sugiyama 布局算法（25 步流水线）
- `view.js` — 缩放/平移/视图管理
- `onnx.js`, `pytorch.js` — 格式解析器（参考其模型数据抽象模式）

**Model Explorer** (`src/ui/src/components/visualizer/`):
- `webgl_renderer.ts` — WebGL 主渲染器，包含 InstancedBufferAttribute 实例化渲染
- `webgl_renderer_threejs_service.ts` — Three.js 桥接层（最接近我们 R3F 的用法）
- `webgl_rounded_rectangles.ts` — SDF 圆角矩形片段着色器
- `webgl_edges.ts` — 边渲染（贝塞尔曲线 + 箭头）
- `webgl_texts.ts` — GPU 文字渲染
- `worker/` — Web Worker 离线布局计算

**Transformer Explainer** (`src/`):
- `components/Attention.svelte` — 注意力机制完整可视化（含 Q/K/V 分解动画）
- `components/AttentionMatrix.svelte` — 注意力权重热力图（D3 色阶 + hover 交叉高亮）
- `components/Sankey.svelte` — Sankey 数据流图（`getBoundingClientRect` 动态定位 + 贝塞尔带状路径）
- `components/ProbabilityBars.svelte` — 输出概率分布柱状图
- `store/` — Svelte stores 全局状态管理（对应我们的 Zustand store）

**LLM Viz** (`src/llm/`，仅供思路参考，不可复制代码):
- `GptModelLayout.ts` — 3D 空间布局核心（残差流 Y 轴、分支 X 轴、并行 Z 轴）
- `Camera.ts` — 弹簧物理相机（临界阻尼参数 mass=1, tension=170）
- `Annotations.ts` — 3D 标签/注解系统
- `render/modelRender.ts` — 8 Pass 渲染管线编排
- `render/blockRender.ts` — Transformer 块渲染（含 6 面差异光照）
- `render/fontRender.ts` — MSDF GPU 字体渲染
- `render/blurRender.ts` — Bloom/模糊后处理
- `render/threadRender.ts` — 数据流连线渲染
- `gpu/WebGpuMain.ts` — WebGPU 入口（v2.0 WebGPU 迁移参考）

## 各产品数据模型模式（指导我们的 JSON 结构设计）

| 产品 | 图数据表示 | 状态管理 | 对我们的启示 |
|---|---|---|---|
| Netron | 扁平节点/边列表 + 类型映射 | 全局单例 View 对象 | 后端 API 返回扁平 `tree: TreeNode[]` 即可 |
| Model Explorer | 嵌套 GroupNode 树 + 每组独立子图 | RxJS Observable 流 | 支持层级展开的树结构必须有 `children` 嵌套 |
| Transformer Explainer | 硬编码 GPT-2 结构 + Svelte stores | Svelte writable stores | 前端 Zustand store 设计参考其分层 store 模式 |
| LLM Viz | 枚举式模型定义 + 布局计算分离 | React state + 自定义事件 | 数据(ModelDef)与布局(ModelLayout)分离是好模式 |

## Netron（SVG + 自研 Dagre 布局）

| 技术要点 | 我们的借鉴 |
|---|---|
| **自研 Dagre 布局引擎**：25 步 Sugiyama 算法流水线，含 5 层前瞻交叉减少 + fan-out/fan-in 校正 | 我们使用 R3F 3D 布局，不需要 Dagre 图布局。但 **Web Worker 计算布局** 的模式直接采纳 |
| **双路径边命中检测**：每条边渲染两个 `<path>`（可见细线 + 不可见宽线用于点击），解决细线难点击问题 | 直接采纳：3D 连线的射线检测(raycasting)使用更大的命中体积 |
| **操作元数据系统**：每个 op 类型有 `-metadata.json`，提供人类可读的文档、输入输出规格 | **重要借鉴**：为 HF 模型的每种模块类型（Attention、MLP、MoE、Norm 等）构建类似的元数据文档系统，点击模块时显示详细说明 |
| **>3000 节点自动降级**：大图自动从 network-simplex 切换到 longest-path 布局算法 | 借鉴思路：大模型（>100 层）自动简化显示策略 |
| **SVG 6 层分层渲染**：cluster → edge-path → edge-hit → edge-label → node → tunnel | 借鉴：3D 场景使用 render order 分层（连线 → 块体 → 标签 → 高亮 → 后处理） |

## Google Model Explorer（WebGL2 + Three.js + 自定义 SDF 着色器）

| 技术要点 | 我们的借鉴 |
|---|---|
| **GPU 实例化渲染 + SDF 片段着色器**：单个 `PlaneGeometry(1,1)` 模板 + `InstancedBufferAttribute` 渲染所有节点，片段着色器用 SDF 绘制圆角矩形，无需纹理 | **核心借鉴**：R3F 的 `<Instances>` 组件本质上就是这个技术。MoE 384 专家网格直接使用此方案 |
| **分屏同步对比**：双面板独立缩放/平移，节点名匹配 + 自定义 JSON 映射 + 差异高亮（红=删除，绿=新增） | **直接采纳**：模型对比页面实现双 Canvas 同步交互 |
| **层级布局分治**：每个 GroupNode 独立布局，而非整张图一次布局 | **直接采纳**：按层级（模型 → 层 → Attention/MLP → 子组件）分级展开和布局 |
| **LOD 文字/颜色混合**：远距离节点背景色混向中性色，文字标签超出阈值隐藏 | **直接采纳**：R3F 的 `<LOD>` 组件 + 自定义距离回调实现多级细节 |
| **Node Data Provider 叠加**：在节点上方直接渲染量化数据分布条 | **借鉴**：在 3D 层块上叠加参数量比例条或量化状态指示器 |
| **Angular zone 优化**：鼠标事件监听器注册在 zone 之外，避免不必要的变更检测 | 对应到 React：使用 `useFrame` 而非 React state 驱动高频动画，避免 React 重渲染 |

## Transformer Explainer（Svelte + D3 + GSAP + 浏览器内推理）

| 技术要点 | 我们的借鉴 |
|---|---|
| **浏览器内实时推理**：GPT-2 通过 `@xenova/transformers` + `onnxruntime-web` 在客户端运行，用户输入文字后实时显示真实注意力权重和激活值 | **未来演进（v2.0）**：使用 Transformers.js 实现小模型的浏览器内推理，展示真实中间 tensor |
| **GSAP 分段动画序列**：展开组件时按时间线编排（隐藏摘要 → 渐入矩阵 → 线条绘制效果 → 缩放入圆点 → 滑入 mask → 渐入路径） | **直接采纳**：层展开动画使用 GSAP timeline 编排多步骤序列，创造叙事感 |
| **Sankey 风格连接路径**：通过 `getBoundingClientRect()` 查询 DOM 元素位置，动态生成贝塞尔带状路径连接组件 | 借鉴思路：3D 场景中信息面板的连接线可使用类似动态定位 |
| **注意力矩阵可视化**：SVG 圆点（非矩形），`d3.scaleLinear` 白到紫渐变，hover 交叉高亮 Q/K 向量 | **借鉴**：MLA 可视化中的 Q/K/V 矩阵热力图使用类似的交互式色阶映射 |
| **SVG 渐变线性路径 + 虚线残差连接**：数据流路径使用线性渐变，残差连接使用虚线 | **直接采纳**：3D 连线区分数据流（实线发光）和残差连接（虚线半透明） |

## LLM Viz（纯自研 WebGL2 引擎 + 逐权重可视化）

| 技术要点 | 我们的借鉴 |
|---|---|
| **3D 空间隐喻**：残差流作为垂直主干（Y 轴向下），Attention/MLP 水平分支（X 轴），注意力头深度展开（Z 轴） | **核心借鉴**：我们的 3D 布局直接采用此空间隐喻 — Y 轴为推理方向，X 轴为计算分支，Z 轴为并行结构（头/专家） |
| **逐权重数据纹理采样**：将权重值存为 GPU 纹理，片段着色器通过 `texelFetch` + `u_accessMtx` 矩阵映射 3D 块坐标到纹理坐标，正值=蓝色、负值=黑色、零=灰色 | **未来演进（v2.0）**：真正的 "X-Ray" 模式，展示逐权重值的 3D 热力体 |
| **多尺度网格线**：16 格和 256 格边界线，通过 GLSL `fwidth()` + `smoothstep` 实现 LOD 平滑出现/消失 | 借鉴：3D 场景的参考网格使用类似的多尺度 LOD 渲染 |
| **引导式 Walkthrough**：10 个步骤组件精确对应 forward pass，每步控制相机位置、块体高亮（透明度 0-1）、文字标注 | **直接采纳**：实现 "Guided Tour" 模式 — 相机沿推理路径飞越，逐步高亮当前模块并显示说明 |
| **多 Pass 渲染**：Blur/Glow Pass → Geometry → Block Lighting → Thread → Opaque → Arrow → Overlay → Overlay2D | 简化采纳：R3F 通过 `<EffectComposer>` + `<Bloom>` + `<SelectiveBloom>` 实现类似的多 Pass 效果 |
| **弹簧物理相机**：临界阻尼（`2 * sqrt(mass * tension)`）相机动画，667ms 过渡 | **直接采纳**：R3F `<CameraControls>` + react-spring 弹簧物理驱动相机飞越 |
| **按需渲染**：非持续渲染循环，仅状态变化时重绘 | **直接采纳**：R3F Canvas 设置 `frameloop="demand"`，仅需要时渲染 |

## 借鉴优先级矩阵

| 优先级 | 技术 | 来源 | 实现阶段 | 影响 |
|---|---|---|---|---|
| P0 | GPU 实例化渲染（MoE 专家网格） | Model Explorer | Phase 3 | 性能基础 |
| P0 | 3D 空间隐喻布局 | LLM Viz | Phase 3 | 核心体验 |
| P0 | Web Worker 布局计算 | Netron + Model Explorer | Phase 3 | 响应性 |
| P0 | 层级展开/折叠 | Model Explorer | Phase 3 | 大模型可用性 |
| P1 | GSAP 分段动画序列 | Transformer Explainer | Phase 4 | 叙事感 |
| P1 | Guided Tour 相机飞越 | LLM Viz | Phase 4 | 教育价值 |
| P1 | LOD 多级细节 | Model Explorer | Phase 3 | 多尺度体验 |
| P1 | 分屏同步对比 | Model Explorer | Phase 6 | 差异化功能 |
| P2 | 模块元数据文档系统 | Netron | Phase 5 | 信息深度 |
| P2 | 选择性 Bloom 发光 | LLM Viz | Phase 3 | 视觉品质 |
| P2 | 按需渲染 | LLM Viz | Phase 3 | 性能优化 |
| P3 | 浏览器内推理 + 真实 tensor | Transformer Explainer | v2.0 | 终极 "X-Ray" |
| P3 | 逐权重数据纹理可视化 | LLM Viz | v2.0 | 终极深度 |

## 完整借鉴特性清单（36 项，按优先级分层）

### TIER 1: 必须实现（10 项）

| # | 特性 | 来源 | 2D/3D | 实现要点 |
|---|---|---|---|---|
| 1 | **渐进式三层展开** | Transformer Explainer | Both | 模型→层→子组件三级折叠/展开，点击渐进揭示细节 |
| 2 | **3D 空间隐喻布局** | LLM Viz | 3D | Y=推理方向，X=计算分支，Z=注意力头/专家并行 |
| 3 | **Guided Walkthrough** | LLM Viz | Both | 10 步骤逐步解说 + 相机动画 + 时间轴控制 |
| 4 | **NDP 数据叠加层** | Model Explorer | Both | 在节点上叠加数值条（参数量、延迟、量化位数等） |
| 5 | **分屏同步对比** | Model Explorer | 2D | 双面板独立操作 + 节点匹配 + 差异红绿高亮 |
| 6 | **逐元素颜色映射** | LLM Viz | Both | 权重/激活值的颜色编码，正=蓝，负=黑，零=灰 |
| 7 | **五视图上下文侧边栏** | Netron | Both | 模型/子图/节点/连接/张量 五种属性面板动态切换 |
| 8 | **正则搜索（4 种匹配）** | Model Explorer | Both | 按标签/属性/输入/输出正则搜索，结果高亮+跳转 |
| 9 | **SVG/PNG 导出** | Netron + Model Explorer | Both | 2D 导出 SVG（内联 CSS），3D 导出 PNG（Canvas 截图） |
| 10 | **URL 状态持久化** | Model Explorer | Both | 模型 ID + 展开状态 + 缩放 + 选中节点编码到 URL |

### TIER 2: 高价值（10 项）

| # | 特性 | 来源 | 2D/3D | 实现要点 |
|---|---|---|---|---|
| 11 | **注意力权重热力图** | Transformer Explainer + BertViz | Both | Hover token 高亮权重，Head/Model/Neuron 三视图 |
| 12 | **温度/采样交互控制** | Transformer Explainer | Both | 温度滑块实时更新概率分布，Top-K/Top-P 切换 |
| 13 | **面包屑子图导航** | Netron | Both | 栈式进入/退出子图 + 跨边界隧道连线 |
| 14 | **书签系统（1-9）** | Model Explorer | Both | 保存/恢复最多 9 个图状态，缩略图预览 |
| 15 | **层级颜色分布条** | Model Explorer | 2D | 折叠层底部显示子节点颜色比例条 |
| 16 | **相同层自动检测** | Model Explorer | Both | 选中一层自动高亮所有结构相同的层 |
| 17 | **输入/输出链追踪** | Model Explorer | Both | 选中节点高亮完整祖先+后代链，其余变暗 |
| 18 | **弹簧物理相机** | LLM Viz | 3D | 临界阻尼相机过渡（667ms），球坐标系统 |
| 19 | **双重 LOD 策略** | LLM Viz | 3D | 小块隐藏 + 多头合并，>12 层自动简化 |
| 20 | **独立 HTML 导出** | BertViz | Both | 下载为自包含 HTML 文件，离线可查看 |

### TIER 3: 锦上添花（6 项，v1.1+）

| # | 特性 | 来源 | 2D/3D | 实现要点 |
|---|---|---|---|---|
| 21 | **暗色/亮色主题切换** | Netron | Both | 自动检测 OS + 手动切换 + localStorage 持久化 |
| 22 | **滚轮行为切换** | Netron | Both | 缩放 vs 滚动可选（Cmd/Ctrl+M） |
| 23 | **节点样式器** | Model Explorer | Both | 正则规则自定义节点颜色，规则可导入/导出 |
| 24 | **多列布局换行** | LLM Viz | 3D | >12 层自动分列显示，防止过长单列 |
| 25 | **出版级 SVG 导出** | NN-SVG | 2D | 面向论文的干净标注 SVG 图 |
| 26 | **拖放 + URL 加载** | Netron | Both | 拖放本地文件 + `?model=xxx` URL 参数加载 |
