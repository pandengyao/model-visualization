# 五、3D 可视化设计（v1.0 纯 3D）

> **v1.0 范围**：纯 3D 模式（含 WebGL 不可用时的文字 Fallback）。**2D SVG 模式整节已删除**，移至 v1.1+（见本文件末尾 TODO）。
>
> **对齐产品原则**：2（精美 3D 风格）/ 3（结构与数据流 100% 正确）/ 4（教学深度与动画精细度超越竞品）/ 5（前期不做性能优化，交互响应除外）/ 6（AnimationLayer 插件式叠加）/ 7（真实模型优先）/ 8（Template A/B/C/G 架构广度底线）/ 9（Provenance 强制展示）。
>
> **契约权威**：AnimationLayer 四层分工以 [11-extension-points.md §3](11-extension-points.md) 为准，本文件是其视觉落地。

> [HF Model Visualizer](README.md) 技术设计文档 — 章五

## 目录

- [5.1 3D 场景总览与空间隐喻](#51-3d-场景总览与空间隐喻)
- [5.2 AnimationLayer 四层叠加模型](#52-animationlayer-四层叠加模型)
- [5.3 v1.0 Stage-2 数据流动画（三项最终范围）](#53-v10-stage-2-数据流动画三项最终范围)
  - [① Attention Q/K/V 分解动画](#-attention-qkv-分解动画)
  - [② MoE 路由动画](#-moe-路由动画)
  - [③ Residual flow 动画](#-residual-flow-动画)
- [5.4 视觉规范（精美硬约束，对齐原则 2）](#54-视觉规范精美硬约束对齐原则-2)
  - [配色系统（深色优先 + 玫瑰金强调）](#配色系统深色优先--玫瑰金强调)
  - [材质](#材质)
  - [光照](#光照)
  - [后处理（必选）](#后处理必选)
  - [微交互](#微交互)
  - [排版](#排版)
  - [相机](#相机)
  - [动效缓动](#动效缓动)
- [5.5 Template A/B/C/G 视觉设计（对齐原则 8）](#55-template-abcg-视觉设计对齐原则-8)
- [5.6 Provenance 徽标规范（对齐原则 9）](#56-provenance-徽标规范对齐原则-9)
- [5.7 Config 编辑器 UI 规范（对齐原则 6 PATCH /config）](#57-config-编辑器-ui-规范对齐原则-6-patch-config)
- [5.8 GPU 选择器 UI 规范（对齐原则 6 GPU Catalog）](#58-gpu-选择器-ui-规范对齐原则-6-gpu-catalog)
- [5.9 交互设计](#59-交互设计)
- [5.10 WebGL 不可用 Fallback](#510-webgl-不可用-fallback)
- [5.11 v1.1+ TODO（本文件范围内）](#511-v11-todo本文件范围内)

---

## 5.1 3D 场景总览与空间隐喻

```
┌──────────────────────────────────────────────────────────────────┐
│  顶部工具栏                                                        │
│  [🔍 搜索] [🧊 Template A/B/C/G 指示]                              │
│  [⚡ GPU: A100-80G ▼] [▶ Guided Tour] [⚙ Animation Layers]        │
│                                                                    │
│  ┌───────────────────────────────────┬────────────────────────┐  │
│  │                                     │  右侧浮动面板          │  │
│  │        R3F Canvas (3D 场景)         │  ───────────────────   │  │
│  │                                     │  Provenance Summary    │  │
│  │                                     │  ───────────────────   │  │
│  │                                     │  Config 编辑器         │  │
│  │                                     │  （实时 PATCH /config） │  │
│  │                                     │  ───────────────────   │  │
│  │                                     │  节点属性 / 张量形状    │  │
│  │                                     │  MemoryBreakdown        │  │
│  └───────────────────────────────────┴────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  AnimationLayer 时间轴                                         ││
│  │  [L1 Structure  ▢]  [L2 DataFlow  ▢]                          ││
│  │  [L3 Heatmap v1.1 🔒]  [L4 Parallelism v1.2 🔒]               ││
│  │  ◀ ⏸ ▶  ───●──────────────── 1:30 / 5:00  [1x]                ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

**空间隐喻（以 `meta-llama/Llama-3-8B` 32 层解码器为例）**：

```
  Y 轴向下 = 推理方向（token 从上到下穿越）
  X 轴水平 = 计算分支（左: Attention / 中: Residual 主干 / 右: MLP 或 MoE）
  Z 轴深度 = 并行结构（heads / experts 沿 Z 轴展开）

  ┌──────────────┐
  │  Embedding   │  emb: (vocab=128256, hidden=4096) ← Template A 标准入口
  └──────┬───────┘
         │
  ┌──────┴──────────────────────────────────┐
  │  Decoder Block × 32（展开后）             │
  │   ┌─────────────┐         ┌──────────┐  │
  │   │  Attention  │◀══ 主干 ═▶│   MLP   │  │   注意力蓝 / MLP 紫
  │   │  (Q/K/V)    │  residual │(SwiGLU) │  │
  │   └─────────────┘         └──────────┘  │
  └──────┬──────────────────────────────────┘
         │
  ┌──────┴───────┐
  │  Final Norm  │
  └──────┬───────┘
  ┌──────┴───────┐
  │   LM Head    │  (hidden=4096 → vocab=128256)
  └──────────────┘

  真实模型示例（v1.0 必须全部跑通）:
    - meta-llama/Llama-3-8B          → Template A
    - mistralai/Mixtral-8x7B         → Template B（8 experts, top-2）
    - deepseek-ai/DeepSeek-V3        → Template C（MLA + 256 experts + shared）
    - 任何未识别的 *ForCausalLM      → Template G（通用回退 + INFERRED 徽标）
```

---

## 5.2 AnimationLayer 四层叠加模型

> **契约对齐**：本节严格对齐 [11-extension-points.md §3](11-extension-points.md)。层间通过共享 `AnimationContext` 时间轴协调，禁止硬编码依赖；每层可独立 toggle，关闭任一层不得破坏其他层。

### 5.2.1 四层分工

| 层 | 名称 | v1.0 范围 | 职责 | 叠加依赖 |
|---|---|---|---|---|
| **L1** | StructureAnimation | ✅ v1.0 | 模块展开 / 收起 / 层级过渡 | — |
| **L2** | DataFlowAnimation | ✅ v1.0（三项子集） | Attention Q/K/V + MoE 路由 + Residual flow | L1 |
| **L3** | NumericalHeatmap | v1.1 | Attention 权重热力、激活值分布 | L1 + L2 |
| **L4** | ParallelismAnimation | v1.2 | TP / PP / DP / EP / CP / SP 通信原语 | L1 + L2 |

### 5.2.2 前端 UI（强制要求）

- 工具栏底部 / 时间轴上方必须提供四层**独立开关**（toggle）+ **独立时间轴进度条**。
- L3 / L4 在 v1.0 显示为 🔒 灰态（hover 提示"v1.1/v1.2 交付"）。
- 开关触发后，`AnimationContext.timeline` 重编排，不重建场景。
- 同屏多层同时启用时，**时间轴共享**（master timeline），各层通过 `(startTime, duration, targetNodeId, tween)` 元组声明自身片段。

### 5.2.3 AnimationContext 共享时间轴（TS 草案）

```typescript
interface AnimationContext {
  master: gsap.core.Timeline;      // 全局主时间轴
  layers: Record<'L1_structure' | 'L2_dataflow' | 'L3_heatmap' | 'L4_parallelism',
                 { enabled: boolean; sub: gsap.core.Timeline }>;
  graph: ModuleGraph;              // 当前 ModuleGraph（PATCH /config 后重新注入）
  now: number;                     // 秒
}
```

**约束**：
- L2 读取 L1 提供的节点最终 world position，但**不得**通过直接引用 R3F ref 耦合 —— 必须通过 `AnimationContext.graph` 的 layout 结果。
- L1 关闭时，L2 在折叠态节点上以简化形式播放（例如残差只走主干直线）。

---

## 5.3 v1.0 Stage-2 数据流动画（三项最终范围）

> **最终决议（对齐原则 4：教学深度与动画精细度大幅超越竞品）**：
>
> 原先被移至 v1.1 的 Stage-2 三项核心动画，**全部拉回 v1.0 必交付范围**：
> 1. **Attention Q/K/V 分解**
> 2. **MoE 路由**
> 3. **Residual flow（含 Pre/Post-LN 位置）**
>
> **v1.1+ 保留**：脉动 / 膨胀 / 螺旋 / 热力图 / token residual 细粒度 / 反向传播动画。

### ① Attention Q/K/V 分解动画

> 演示模型：`meta-llama/Llama-3-8B`（32 heads, hidden=4096, head_dim=128）

展示完整计算链，**一镜到底**：

```
Step 1: tokens (B, S, 4096) 发光紫色张量块从上方流入
Step 2: 三条并行投影分叉 —— Q / K / V，分别显影为 蓝 / 青 / 绿
           tokens → W_Q → Q (B, S, 32, 128)
           tokens → W_K → K (B, S, 32, 128)
           tokens → W_V → V (B, S, 32, 128)
Step 3: Q 与 K^T 做矩阵乘 —— 3D 方格矩阵在空中旋转对齐
           QK^T: (B, 32, S, S)   ← 注意力得分矩阵
Step 4: Softmax —— 每行以暖色系（玫瑰金 → 橙）渐变高亮，最大值粒子发光
           attention_weights: (B, 32, S, S)
Step 5: × V —— 权重矩阵与 V 相乘，粒子流按权重比例分配到 V 的各 token 行
           output: (B, S, 32, 128) → concat → (B, S, 4096)
Step 6: output 流回主干（residual add，见 ③）
```

**视觉要点**：
- Q/K/V 三条支流以**材质色区分**（蓝/青/绿 `MeshStandardMaterial`），非 opacity 分层。
- Softmax 激活格用 **Bloom emissive** 强化最大值感知。
- 所有张量形状变化用 Drei `<Html>` 标签实时跟随（如 `(B, S, 4096) → (B, 32, S, 128)`）。
- 每个动作附讲解字幕（毛玻璃 Html overlay）。

### ② MoE 路由动画

> 演示模型：`mistralai/Mixtral-8x7B`（8 experts, top-2）与 `deepseek-ai/DeepSeek-V3`（256 routed + 1 shared, top-8）

```
Step 1: token 张量 (B, S, hidden) 进入 Gate
Step 2: Gate Linear → softmax → top-k
           Mixtral: top-2 of 8
           DeepSeek-V3: top-8 of 256（+ shared expert 始终激活）
Step 3: 被选中的 Experts 高亮（橙色 emissive + Bloom outline）；
        未选中 Expert 降低到 opacity 0.15 的"休眠态"
Step 4: token 粒子流沿 Gate → 选中 Experts 并行分流（N 条亮色轨迹）
Step 5: 每个激活 Expert 内部完成 SwiGLU 前向（膨胀/收缩由子动画呈现）
Step 6: 加权求和 —— 各 Expert 输出按 gate 权重回流、在主干前汇合
        DeepSeek-V3 额外显示 Shared Expert 的恒定贡献支路
```

**视觉要点**：
- Experts 使用 **InstancedMesh**（`count = num_experts`），激活态通过 instance attribute 切换 emissive。
- Gate top-k 概率条形图悬浮于 Gate 上方（`<Html>` 毛玻璃条）。
- 权重数字（如 `0.42`）贴在每条支流上，等宽字体 JetBrains Mono。

### ③ Residual flow 动画

> 演示模型：所有三个 Template（A/B/C）均需展示

展示 **residual stream 如何跨 block 传递 + Pre/Post-LN 位置**：

```
残差主干（贯穿 Embedding → 所有 Decoder Block → Final Norm）:
  一根半透明发光圆柱，玫瑰金色 #ec4899，始终微微脉动（opacity 0.3~0.5）

每个 Block 内部（以 LLaMA 的 Pre-LN 为例）:
  x ─┬────────────────────────────────────────────── x + ΔA ──┐
     │                                                          │
     └─→ RMSNorm → Attention → ΔA ─────────────────────────────┘
                                (支流汇回主干：粒子沿弧线回流 + "+" 图标闪烁)

  x + ΔA ─┬────────────────────────────────────── (x + ΔA) + ΔM ──
          │
          └─→ RMSNorm → MLP/MoE → ΔM ────────────────────────────┘

Pre-LN 位置标注: RMSNorm 紧贴 Attention/MLP 的 **入口侧**
Post-LN（若存在）: 标注在汇回点 **出口侧**（目前 v1.0 三个 Template 均 Pre-LN）
```

**视觉要点**：
- 残差主干始终可见，是场景的"脊柱"。
- Pre-LN / Post-LN 用 3D Text 小标签直接贴在 Norm 节点上方。
- "+" 合并点用一个小八面体 + Bloom，点击可查看张量形状 `(B, S, hidden)` 保持不变的契约。

---

## 5.4 视觉规范（精美硬约束，对齐原则 2）

> 原则 2：**视觉质感必须精美（材质、光照、后处理、微交互、排版、动效经得起截图分享）**。以下为硬约束，不允许"能跑就行"。

### 配色系统（深色优先 + 玫瑰金强调）

**禁止 Tailwind 默认灰阶堆砌**；以下为基线，冷色调主色 + 玫瑰金强调。

```
═══ 背景层（深色模式默认，无浅色模式 v1.0） ═══
主背景      #0b1220 (近黑冷蓝)
次背景      #1e293b (Slate 900 基准主色)
表面色      #111827 (悬浮卡片背景)
玻璃拟态    backdrop-blur-md bg-white/10 border-white/15

═══ 强调色（二选一，项目启动时冻结） ═══
方案 A · 玫瑰金:  #ec4899   ← 残差主干 / 关键高光 / 进度轴
方案 B · 科技蓝:  #3b82f6   ← 残差主干 / 关键高光 / 进度轴
（v1.0 冻结：方案 A 玫瑰金，除非后续评审翻案）

═══ 模块类别色（3D 场景内材质 baseColor，禁止更改色相） ═══
Attention       #3b82f6 (科技蓝)
MLP             #8b5cf6 (紫)
MoE Expert      #f97316 (橙)
Norm            #94a3b8 (灰)
Embedding       #10b981 (绿)

═══ Provenance 徽标色（见 §5.6） ═══
EXACT           #22c55e (绿色实心圆)
INFERRED        #3b82f6 (蓝色空心圆)
ESTIMATED       #eab308 (黄色三角)
```

### 材质

所有模块**统一**使用 `MeshStandardMaterial`：

```tsx
<meshStandardMaterial
  color={categoryColor}      // 严格按 §配色 模块类别色
  roughness={0.3}
  metalness={0.6}            // 轻微金属感
  envMapIntensity={1.0}
/>
```

- **禁止** `MeshBasicMaterial`、**禁止** 默认灰色 PBR。
- 活跃态：`emissive = categoryColor; emissiveIntensity = 0.3;` + outline pass。

### 光照

使用 Drei `<Environment>` HDR 环境光 + 一个方向光 + 一个点光：

```tsx
<Environment preset="studio" />   {/* 或 "city" */}
<directionalLight position={[10, 15, 8]} intensity={0.8} castShadow />
<pointLight position={[-6, 4, -4]} intensity={0.5} color="#ec4899" />
```

### 后处理（必选）

使用 `@react-three/postprocessing`：

```tsx
<EffectComposer>
  <Bloom threshold={0.9} intensity={0.4} radius={0.8} />
  <SSAO intensity={1.2} />
  <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
  <ChromaticAberration offset={[0.0002, 0.0002]} />   {/* 极轻微，可选 */}
</EffectComposer>
```

### 微交互

| 交互 | 视觉反馈 |
|---|---|
| **悬停节点** | `scale 1.0 → 1.02` + `emissiveIntensity 0 → 0.3` + 外描边发光（Drei `<Outlines>`） |
| **点击节点** | 点击位置产生 ripple 向外扩散（自定义 shader Ring expanding 600ms）+ 右侧信息面板从右滑入（react-spring `config.gentle`） |
| **选中态** | Postprocessing `OutlinePass` 高亮选中节点 + 其他节点 opacity 0.6 |
| **时间轴 scrub** | 按 16ms/frame 重绘，禁止 throttle |

### 排版

```
UI 字体:        系统级无衬线 (ui-sans-serif, -apple-system, "Inter" fallback)
代码/数字:      "JetBrains Mono", ui-monospace（所有张量形状、数值、shape 元组必须等宽）
3D 标签层:      Drei <Html> 半透明玻璃拟态
                className="backdrop-blur-md bg-white/10 border border-white/15
                           rounded-xl px-3 py-2 text-sm"
3D 文字:        Drei <Text> + Inter SDF
```

### 相机

```tsx
<PerspectiveCamera fov={45} position={[0, 20, 40]} makeDefault />
<OrbitControls
  enableDamping
  dampingFactor={0.05}
  makeDefault
/>
```

- 相机阻尼必开；禁止 `dampingFactor=0`。
- 初始视角：俯视偏前，Y=20, Z=40，看向原点。

### 动效缓动

- **所有过渡必须用** GSAP `power2.inOut` **或** react-spring `config.gentle`。
- **禁止 `linear`**（除非是恒速循环粒子发射，且必须显式注释）。
- 默认过渡时长：组件展开 600ms / 面板滑入 300ms / 悬停 200ms。

---

## 5.5 Template A/B/C/G 视觉设计（对齐原则 8）

| 模板 | 目标模型族 | 3D 视觉结构 | Provenance |
|---|---|---|---|
| **A** | `meta-llama/Llama-3-8B` / `Qwen/Qwen2.5-7B`（LLaMA 族 Decoder） | 垂直堆叠 Decoder Block × N；每 Block 内 **Attention（左）∥ MLP（右）** 并列，residual 主干居中 | EXACT（全部字段来自 config + safetensors） |
| **B** | `mistralai/Mixtral-8x7B`（LLaMA-MoE） | 在 A 的基础上 **MLP → MoE 扇形展开**：Gate 位于分支入口，N 个 Expert 围绕 Gate 呈扇形排列（Mixtral=8，呈半圆；更多 experts 使用径向网格） | EXACT |
| **C** | `deepseek-ai/DeepSeek-V3`（DeepSeek-MoE） | 在 B 的基础上新增三项：<br>• **MLA 压缩头**：Attention 区显示双锥漏斗（q_lora / kv_lora 压缩维度可视化）<br>• **Shared Expert**：独立大块，位于 Routed Experts 扇形**外侧**，始终激活（emissive 常亮）<br>• **Routed Experts 分组**：按 expert_group 着色微差（同组同色深浅） | EXACT |
| **G** | 通用回退（任何未识别 `*ForCausalLM`） | • 仅展示 **config 已知字段**对应节点（num_layers / hidden_size / vocab_size / num_heads 若存在）<br>• 通用 Decoder 骨架（Embedding → N × Block → Norm → LM Head）<br>• **醒目全局 `INFERRED` 徽标**（右上角大号）<br>• **免责浮层**：顶部条幅："⚠ 该架构未识别，仅基于 config 字段推断" —— 点击关闭后仍保留徽标 | INFERRED |

**切换策略**：`ArchitectureAdapter.detect()` 判定返回 `template_id`，前端 `TemplateContract.canRender()` 匹配后挂载对应 `<Scene>`。Template G 永远兜底，**禁止**默认回退到 A 伪装 LLaMA。

---

## 5.6 Provenance 徽标规范（对齐原则 9）

### 5.6.1 节点级徽标

每个模块节点的**右上角**以 Drei `<Html>` 形式显示徽标：

| Provenance | 形状 | 颜色 | 说明 |
|---|---|---|---|
| `EXACT` | 实心圆 ● | 绿 `#22c55e` | 来自 config + safetensors 的确切字段 |
| `INFERRED` | 空心圆 ○ | 蓝 `#3b82f6` | 基于 model_type 或启发式推断 |
| `ESTIMATED` | 三角 ▲ | 黄 `#eab308` | 基于公式估计（如 activations/KV cache 估算） |

**悬停徽标 → Tooltip** 显示 Provenance 详情：

```
┌─────────────────────────────────────┐
│  ▲ ESTIMATED                         │
├─────────────────────────────────────┤
│  source:  memory_estimator.v1       │
│  caveats: 基于 bsz=1, seq=4096      │
│           激活值按 checkpointing=off │
│           估算；真实训练可能 ±20%    │
└─────────────────────────────────────┘
```

### 5.6.2 全局 Provenance Summary

右侧浮动面板**顶部**显示全局 `provenance_summary`（由后端 ModuleGraph 汇总）：

```
┌─ Provenance Summary ────────────────────┐
│  Template: C (DeepSeek-V3)              │
│  ● EXACT:     142 nodes                  │
│  ○ INFERRED:  3 nodes                    │
│  ▲ ESTIMATED: 2 fields (activations,    │
│                         kv_cache)       │
│  Source: config.json + safetensors      │
│  Revision: 1                             │
└─────────────────────────────────────────┘
```

**禁止**：任何"估算即展示，不标来源"的设计。凡是屏幕上的数字/箭头/结构，必须能追溯。

---

## 5.7 Config 编辑器 UI 规范（对齐原则 6 PATCH /config）

```
┌─ Config Editor (浮动右侧) ──────────────────────┐
│                                                  │
│  Model: deepseek-ai/DeepSeek-V3                  │
│  Template: C                                      │
│                                                  │
│  num_hidden_layers        61   → 61              │
│    ──●────────────── [slider 1-128]              │
│                                                  │
│  hidden_size            7168   → 7168            │
│    [input number]                                 │
│                                                  │
│  num_experts             256   → 128  (修改中)   │
│    [input number]        ⟳ 重算中...              │
│                                                  │
│  num_experts_per_tok       8   → 8               │
│    [input number]                                 │
│                                                  │
│  moe_top_k                 8   → 8               │
│    ...                                            │
│                                                  │
│  [⟲ Reset to original config]                     │
└──────────────────────────────────────────────────┘
```

**行为约束**：

1. 字段变化 → **300ms debounce** → `PATCH /api/v1/stream/{org}/{repo}/config`。
2. 请求中显示 **loading spinner**（玫瑰金色旋转环）；端到端必须 **< 300ms**（原则 5 例外条款硬约束）。
3. 每个字段旁显示 **"原值 → 当前值"**；当前值与原值不同时以玫瑰金强调色高亮。
4. **Reset 按钮**恢复到原 config（触发一次 PATCH，overrides=空）。
5. 白名单字段严格对齐 [11-extension-points.md §8.1](11-extension-points.md)。
6. **并发竞态防护**（对齐 04 §4.7.1 revision 单调）：
   - 前端维护 `lastAppliedRevision`；收到 WS `graph_update` 时若 `msg.revision <= lastAppliedRevision` → **丢弃**（防 debounce 连发 PATCH 后乱序到达）
   - 每次 PATCH 前本地递增 `pendingRequestId`；WS 消息无 requestId 时只靠 revision 单调排序；PATCH 的 HTTP 响应若携带 `accepted_revision < lastAppliedRevision` → 忽略该次响应的 UI 副作用（保留服务端状态）
   - 快速连改多个字段在 300ms 内最终合并为**一次** PATCH（debounce 天然合并）；若用户在 PATCH pending 期间继续修改 → 不取消已发起的，等 WS 回推后再发下一次（顺序保证 + 最终一致）

---

## 5.8 GPU 选择器 UI 规范（对齐原则 6 GPU Catalog）

```
┌─ GPU Selector (顶部工具栏下拉) ────────────────────────────┐
│  ⚡ GPU: A100-80G (80GB, 2TB/s, 312 TFLOPS BF16)  ▼        │
└────────────────────────────────────────────────────────────┘

下拉展开（从 GET /api/v1/gpus 动态拉取）:
┌────────────────────────────────────────────────────────────┐
│  NVIDIA                                                      │
│    A100-40G   (40GB,  1.5TB/s, 312 TFLOPS BF16)            │
│  ✓ A100-80G   (80GB,  2.0TB/s, 312 TFLOPS BF16)            │
│    H100-80G   (80GB,  3.4TB/s, 989 TFLOPS BF16)            │
│    H200-141G  (141GB, 4.8TB/s, 989 TFLOPS BF16)            │
│    B200       (192GB, 8.0TB/s, 2250 TFLOPS BF16)           │
│    4090-24G   (24GB,  1.0TB/s, 165 TFLOPS BF16)            │
│    3090-24G   (24GB,  0.9TB/s, 71 TFLOPS BF16)             │
│    L40S-48G   (48GB,  0.9TB/s, 362 TFLOPS BF16)            │
│  国产                                                        │
│    昇腾 910B  (64GB,  ...)                                  │
│    寒武纪 MLU370 (...)                                      │
│    昆仑芯 P800 / R200 (...)                                 │
└────────────────────────────────────────────────────────────┘
```

**行为约束**：

1. 选项显示格式：`<name> (<mem>, <bandwidth>, <tflops> BF16)`。
2. 选择后自动触发 `PATCH /config` 携带 `gpu_id` → MemoryEstimator 重算 → 右侧 MemoryBreakdown 面板实时更新。
3. 清单完全从 `backend/data/gpu-catalog.yaml` 加载，**禁止**前端硬编码 GPU 列表。

---

## 5.9 交互设计

| 操作 | 效果 |
|---|---|
| 鼠标拖拽 | 旋转 3D 场景（`OrbitControls` + damping 0.05） |
| 滚轮 | 缩放（平滑过渡） |
| 点击层块 | 展开子模块（L1 StructureAnimation：子块从中心向外扩散 600ms，GSAP `power2.inOut`） |
| 再次点击 | 折叠回去 |
| Hover 层块 | scale 1.02 + emissive 0.3 + 外描边发光 + `<Html>` 简要信息 |
| 点击 MoE 块 | 展开 Expert 扇形（Mixtral 8 / DeepSeek-V3 256） |
| 点击单个专家 | 信息面板滑入显示 expert 参数 + Provenance |
| 点击徽标 | Tooltip 显示 source / caveats |
| 播放按钮 | Guided Tour：相机飞越 + L2 数据流动画自动串联播放 |
| 时间轴 scrub | 16ms/frame 重绘；支持跳到任意步骤 |
| AnimationLayer 开关 | 独立 toggle L1/L2；L3/L4 锁定态（v1.1/v1.2） |

---

## 5.10 WebGL 不可用 Fallback

> **原则 5 例外**：本节不是"性能降级"（已按产品原则删除 LOD 降级设计），而是 WebGL 完全不可用时的**底线可访问性**。

```
if (!document.createElement('canvas').getContext('webgl2')) {
  → 渲染 ASCII 结构树 + Markdown 参数表（零 JS 依赖）
  → 顶部条幅："您的浏览器不支持 WebGL2。请升级至 Chrome/Edge 最新版"
  → 仍可正常调用 PATCH /config 编辑与查看 Provenance Summary
}
```

**已删除**：
- ~~Level 0/1/2/3 多档 LOD 降级~~（原则 5：前期不做性能优化）
- ~~GPU 性能检测自动降级~~
- ~~低端机"简化 3D"模式~~
- ~~Mac 集成 GPU 默认/高端 GPU 自动升级双档材质~~ → v1.0 统一 `MeshStandardMaterial(roughness=0.3, metalness=0.6)`。

---

## 5.11 v1.1+ TODO（本文件范围内）

> 以下章节在本次修订中**已删除**，待相应版本回填。

- [ ] **TODO v1.1**：2D SVG 模式（dagre 布局、SVG 6 层、节点/边/数据流动画、SVG 导出、分屏对比、弹出层窗口、边叠加层） —— 整节移出。
- [ ] **TODO v1.1**：L3 NumericalHeatmap 动画层（Attention 权重热力、激活值分布 2D 面板）。
- [ ] **TODO v1.1**：Stage-2 细粒度动画（脉动 / 膨胀 / 螺旋 / RoPE 旋转 / token residual 细粒度）。
- [ ] **TODO v1.1**：反向传播动画（`DataFlowDirection.backward` + forward/backward split-screen）。
- [ ] **TODO v1.2**：L4 ParallelismAnimation（TP / PP / DP / EP / CP / SP + 通信原语 AllReduce / AllGather / ReduceScatter / All2All / P2P）。
- [ ] **TODO v1.2**：2D↔3D 切换过渡动画。
- [ ] **TODO v1.2**：模型对比分屏（两个真实 HF 模型并排）。

---

[← 返回目录](README.md)
