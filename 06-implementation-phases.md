# 七、实现阶段

> [HF Model Visualizer](README.md) 技术设计文档 — 章七

---

### Phase 1: 后端 API（复用 explore_model.py + FastAPI 封装）

**Prerequisites**: Python 3.11+, Node.js 20+, Git, HuggingFace 账号（可选，用于私有模型访问）

**Definition of Done**: 后端 API 可对 5 个验证模型返回完整 JSON 响应（tree、params、moe_info、mla_info、quant_info、data_flow），所有单元测试通过。

#### Step 1.1: 项目初始化
```bash
mkdir -p /Users/frank/work/hf-model-visualizer/backend/app/{routers,services,models}
mkdir -p /Users/frank/work/hf-model-visualizer/backend/tests
```
- `pyproject.toml`: fastapi, uvicorn, httpx, huggingface_hub, cachetools, transformers, torch
- **保留 torch + transformers**：直接复用 explore_model.py 的全部检测函数和 meta-device 加载能力

#### Step 1.2: config_parser.py
- `async fetch_config(model_id, token=None) -> AutoConfig`
  - 使用 `AutoConfig.from_pretrained(model_id)` 加载配置
  - 处理认证、404、网络错误
- `extract_key_config(config) -> dict`
  - 直接复用 explore_model.py:307

#### Step 1.3: detectors.py
- 直接复用 explore_model.py 中的 4 个函数（接收 AutoConfig 对象）：
  - `detect_moe(config)` (原 L128)
  - `detect_mla(config)` (原 L194)
  - `detect_quantization(config)` (原 L220)
  - `detect_sub_configs(config)` (原 L268)

#### Step 1.4: tree_builder.py
- 直接复用（接收 AutoConfig 对象）：
  - `build_synthetic_tree(config)` (原 L578) — config 构建合成树（meta 加载失败时的降级方案）
  - `load_model_meta(config)` — meta device 加载获取**真实模型结构树**
  - `tree_to_text()` (原 L943)
- 优先使用 meta-device 真实模型树；失败时降级到合成树

#### Step 1.5: param_estimator.py
- 直接复用（接收 AutoConfig 对象）：
  - `estimate_params_from_config(config)` (原 L428)
  - `_estimate_attention_params(config)` (原 L347)
  - `_estimate_mlp_params(h, intermediate_size)` (原 L383)
  - `_estimate_vision_params(vision_cfg)` (原 L390)
  - `format_params(count)` (原 L964)
- meta-device 加载成功时使用精确参数统计；失败时使用 config 估算
- **编写准确性测试** (`tests/test_param_estimator.py`):
  ```python
  GROUND_TRUTH = {
      "meta-llama/Llama-3.1-8B": {"total": 8_030_261_248, "tolerance": 0.01},
      "Qwen/Qwen3-30B-A3B": {"total": 30_534_287_360, "tolerance": 0.02},
      "moonshotai/Kimi-K2.6": {"total": 1_026_850_000_000, "tolerance": 0.05},
  }
  # 对每个模型: 估算值与 ground truth 差异 < tolerance
  ```

#### Step 1.6: model_card.py
- `async fetch_model_card(model_id, token=None) -> dict`
  - `huggingface_hub.model_info()` 获取元数据
  - httpx 下载 README.md **完整内容**（不截断）
  - 提取 transformers 版本要求

#### Step 1.7: flow_generator.py — 新增（核心：数据流准确性保证）

数据流生成采用 **三层保证架构：模板选择 + 配置参数化 + SafeTensors 验证**。

##### 7a. 架构模板系统（7 个模板覆盖 95%+ 模型）

> **v1.0 范围**: 仅实现模板 A + C，覆盖 80%+ 热门模型。其余 5 个模板 v1.1 补全。

| 模板 | 架构族 | 覆盖的 model_type | 核心差异 | v1.0 |
|---|---|---|---|---|
| **A: LLaMA Decoder** | 标准 Post-2023 Decoder-Only | `llama`, `mistral`, `qwen2`, `qwen3`, `gemma`, `phi3`, `olmo`, `starcoder2`, `cohere` 等 60+ | RMSNorm + RoPE + SwiGLU + GQA | ✅ |
| **B: LLaMA-MoE** | 标准 MoE (Router + Experts) | `mixtral`, `qwen2_moe`, `qwen3_moe`, `phimoe`, `olmoe`, `jetmoe` 等 20+ | 在 A 基础上 MLP 替换为 Sparse MoE | v1.1 |
| **C: DeepSeek-MoE** | MLA + MoE + Shared Experts | `deepseek_v2`, `deepseek_v3` + `kimi_k2`(via architectures) | MLA 压缩注意力 + 共享专家 | ✅ |
| **D: GPT-2 Decoder** | Pre-LLaMA 架构 | `gpt2`, `gpt_neo`, `codegen`, `gptj` 等 10+ | LayerNorm + 绝对位置 + Conv1D + 单 c_attn | v1.1 |
| **E: Hybrid Mamba** | Mamba SSM + Attention 交替 | `jamba`, `bamba`, `zamba`, `falcon_mamba`, `mamba2` 等 15+ | 由 `layer_types` 决定每层是 Attention 还是 Mamba | v1.1 |
| **F: Encoder-Decoder** | 编码器-解码器 | `bart`, `t5`, `mt5`, `marian`, `pegasus` 等 | 双向编码 + 交叉注意力 | v1.1 |
| **G: Multimodal VLM** | 视觉语言模型 | `qwen2_vl`, `llava`, `gemma3`, `llama4`(vision) 等 | Vision Encoder + Projector + 文本解码器(A/B/C) | v1.1 |

##### 7b. 模板选择算法

```python
def select_template(config: dict) -> str:
    model_type = config.get("model_type", "")
    architectures = config.get("architectures", [])

    # 1. 直接匹配 model_type
    if model_type in TEMPLATE_MAP:
        return TEMPLATE_MAP[model_type]

    # 2. 通过 architectures 回退匹配
    for arch in architectures:
        if "DeepseekV3" in arch or "DeepseekV2" in arch:
            return "C"  # DeepSeek-MoE
        if "ForCausalLM" in arch:
            # 检查 MoE 特征
            if config.get("n_routed_experts") or config.get("num_local_experts"):
                if config.get("q_lora_rank"):
                    return "C"  # DeepSeek-MoE (MLA + MoE)
                return "B"  # LLaMA-MoE
            return "A"  # LLaMA Decoder

    # 3. 特征指纹匹配
    if config.get("q_lora_rank") and config.get("kv_lora_rank"):
        return "C"
    if config.get("layer_types") and "mamba" in str(config.get("layer_types")):
        return "E"

    return "A"  # 默认 LLaMA
```

##### 7c. 配置参数化（决定每层内部结构的关键 config 字段）

| 类别 | 字段 | 影响 |
|---|---|---|
| **逐层结构** | `first_k_dense_replace` | 前 K 层用 Dense MLP，其余用 MoE |
| | `layer_types` / `layers_block_type` | 逐层指定类型（Attention/Mamba/SlidingWindow） |
| | `mlp_only_layers` | 明确的 Dense 层索引列表 |
| | `moe_layers` | 明确的 MoE 层索引列表 |
| **注意力变体** | `q_lora_rank` ≠ null | 触发 MLA（压缩 Q 路径） |
| | `kv_lora_rank` | KV 压缩维度 |
| | `num_key_value_heads < num_attention_heads` | GQA 分组查询注意力 |
| | `sliding_window` | 滑动窗口注意力 |
| **MoE 路由** | `n_routed_experts` | 总专家数 |
| | `num_experts_per_tok` | Top-K 激活数 |
| | `n_shared_experts` | 始终活跃的共享专家数 |
| | `scoring_func` | `sigmoid` vs `softmax` |

##### 7d. SafeTensors Index 验证（可选异步验证）

> **v1.0**: SafeTensors 验证为**可选异步任务**，不阻塞主响应。首次请求返回数据流后，后台异步下载 index 并验证，结果缓存后续请求使用。

**下载 `model.safetensors.index.json`**（仅几 MB，不下载权重），提取所有权重路径并验证：

```python
async def validate_flow(model_id: str, generated_flow: list) -> ValidationResult:
    # 下载 safetensors index（仅权重名→分片映射）
    index = await fetch_safetensors_index(model_id)
    weight_names = set(index["weight_map"].keys())

    # 提取唯一模式（替换层/专家编号为通配符）
    patterns = extract_unique_patterns(weight_names)

    # 逐层验证
    for layer_idx in range(num_layers):
        has_moe = f"model.layers.{layer_idx}.mlp.gate.weight" in weight_names
        has_dense = f"model.layers.{layer_idx}.mlp.gate_proj.weight" in weight_names
        has_mla_q = f"model.layers.{layer_idx}.self_attn.q_a_proj.weight" in weight_names
        has_std_q = f"model.layers.{layer_idx}.self_attn.q_proj.weight" in weight_names

        # 对比模板断言
        assert flow[layer_idx].is_moe == has_moe
        assert flow[layer_idx].is_mla == has_mla_q
```

**Kimi-K2 验证实例**：safetensors index 含 139,644 个权重名，验证结果：
- Layer 0 有 `mlp.gate_proj` 无 `mlp.gate` → 确认 Dense ✓
- Layers 1-60 有 `mlp.gate.weight` + `mlp.experts.E.*` → 确认 MoE ✓
- 所有层有 `self_attn.q_a_proj` 无 `self_attn.q_proj` → 确认 MLA ✓
- 专家索引 0-383 → 确认 384 专家 ✓

##### 7e. 关键发现：`base_model_pp_plan`

transformers 每个 config 类都内置了 `base_model_pp_plan`（流水线并行计划），它是**机器可读的顶层数据流规格**：

```python
# LlamaConfig.base_model_pp_plan:
{
    "embed_tokens": (["input_ids"], ["inputs_embeds"]),
    "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    "norm": (["hidden_states"], ["hidden_states"]),
}
```

所有 decoder-only LLM 的顶层流完全相同：`embed_tokens → layers → norm → lm_head`。所有变化发生在 `layers` 内部。

##### 7f. 准确性等级标注

| 等级 | 条件 | 界面显示 |
|---|---|---|
| ✅ **已验证** | 模板匹配 + SafeTensors 验证通过 | 绿色徽章 "Verified" |
| ⚠️ **高置信** | 模板匹配 + config 一致但未验证 safetensors | 黄色徽章 "High Confidence" |
| ❓ **推断** | 未知 model_type，通过特征指纹推断 | 橙色徽章 "Inferred" |
| ❌ **不支持** | auto_map 且无法匹配任何模板 | 红色警告 "Unsupported Architecture" |

#### Step 1.8: schemas.py — Pydantic 模型
（见上方 API 设计部分）

#### Step 1.9: cache.py — 分层缓存
```python
from cachetools import TTLCache, LRUCache
import hashlib, json, os

DATA_DIR = os.environ.get("CACHE_DIR", "/data/cache")

# L0: 进程内 LRU (maxsize=100, ttl=300s) — 零延迟热点
l0_cache = TTLCache(maxsize=100, ttl=300)

# L1: 文件系统 JSON — 持久化跨重启, 24h TTL
L1_TTL = 86400

def _l1_path(model_id: str, sha: str | None = None) -> str:
    key = f"{model_id}:{sha}" if sha else model_id
    return os.path.join(DATA_DIR, hashlib.md5(key.encode()).hexdigest() + ".json")

async def get_cached(model_id: str) -> dict | None:
    # L0
    if model_id in l0_cache:
        return l0_cache[model_id]
    # L1
    path = _l1_path(model_id)
    if os.path.exists(path):
        data = json.loads(open(path).read())
        l0_cache[model_id] = data  # 提升到 L0
        return data
    return None

async def set_cached(model_id: str, data: dict, sha: str | None = None):
    l0_cache[model_id] = data
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_l1_path(model_id, sha), "w") as f:
        json.dump(data, f, ensure_ascii=False)
```

#### Step 1.10: routers/model.py + compare.py
（见上方 API 设计部分）

#### Step 1.11: main.py
- FastAPI app + CORS (allow localhost:3000) + 挂载路由

#### Step 1.12: 验证
```bash
cd backend && uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/api/v1/model/moonshotai/Kimi-K2.6 | python3 -m json.tool
```
验证返回 JSON 包含完整的 tree、params、moe_info、mla_info、quant_info、data_flow。

---

### Phase 2: Next.js + R3F 脚手架 + 首页

**Prerequisites**: Phase 1 后端 API 已通过验证，Node.js 20+, npm 10+

**Definition of Done**: 首页可正常加载，搜索框输入模型名可跳转到模型页面并显示基本信息面板，Zustand 状态管理就绪，Analytics 集成完毕。

#### Step 2.1: Next.js 初始化
```bash
npx create-next-app@latest frontend --typescript --tailwind --app --src-dir
cd frontend
npm install @react-three/fiber @react-three/drei three @types/three
npm install gsap d3-scale d3-color zustand
npm install @react-three/postprocessing postprocessing
```

#### Step 2.2: API 客户端 (lib/api.ts)
```typescript
export async function fetchModel(org: string, repo: string): Promise<ModelVisualization> {
  const res = await fetch(`${API_URL}/api/v1/model/${org}/${repo}`);
  if (!res.ok) throw new Error(`Model not found: ${res.status}`);
  return res.json();
}
```

#### Step 2.3: 首页 (app/page.tsx)

```
┌──────────────────────────────────────────────────────────────┐
│                                                                │
│              HF Model Visualizer                               │
│              输入 HuggingFace 模型路径，探索 3D 架构           │
│                                                                │
│     ┌──────────────────────────────────────────────┐           │
│     │  🔍  moonshotai/Kimi-K2.6                     │           │
│     └──────────────────────────────────────────────┘           │
│                                                                │
│     热门模型                                                    │
│     ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                       │
│     │Kimi  │ │DSV3  │ │Qwen3 │ │Llama │                       │
│     │K2.6  │ │0324  │ │30B   │ │3.1-8B│                       │
│     │1T MoE│ │671B  │ │MoE   │ │Dense │                       │
│     └──────┘ └──────┘ └──────┘ └──────┘                       │
│                                                                │
│     一句话说明: "零下载，零安装，3 秒看清任意 HF 模型架构"       │
└──────────────────────────────────────────────────────────────┘
```

- 搜索框（输入 HF 模型路径，Enter 或按钮提交）
- 热门模型卡片列表（Kimi-K2.6, DeepSeek-V3, Qwen3-30B-A3B, Llama-3.1-8B）
  - 每张卡片显示：模型名、参数量、架构标签（MoE/MLA/Dense）
  - 点击直接跳转到可视化页面
- 深色主题，Glassmorphism 风格
- URL 支持 `?model=xxx` 参数直接跳转

#### Step 2.3.1: 渐进式加载策略 (LoadingSequence)

模型可视化页面加载时采用三阶段渐进式渲染，确保用户尽早看到有用信息：

```
阶段 1 — 文字信息 (0-1s):
  API 返回 → 立即渲染信息面板:
  - ModelHeader (模型名 + 徽章)
  - ParamTable (参数统计表)
  - KeyConfig (关键配置)
  用户 Quick Check 需求在此阶段即可满足

阶段 2 — 2D 骨架 (1-2s):
  - 渲染 ASCII 结构树 (tree_text)
  - 数据流步骤列表 (FlowStep[])
  - 3D Canvas 显示加载进度条

阶段 3 — 3D 场景 (2-4s):
  - R3F Canvas 初始化 + 灯光
  - 层块几何体渐入（线框→实体动画）
  - 连线"生长"动画
  - Bloom 后处理渐入
  - Guided Tour 按钮就绪
```

#### Step 2.4: 模型页面 (app/model/[org]/[repo]/page.tsx)
- 数据加载（调后端 API）
- ModelHeader：模型名、描述、参数量、徽章（MoE/MLA/量化）
- 左侧：3D 场景（Scene 组件）
- 右侧：信息面板（参数统计、配置、数据流步骤列表）

#### Step 2.5: Zustand 状态分级策略 (stores/useModelStore.ts)

R3F 场景中必须严格区分 React 声明式状态与 Three.js 命令式状态，否则帧率严重下降。

```typescript
// stores/useModelStore.ts — Zustand 仅管理低频交互状态
interface ModelStore {
  // 点击级状态（低频，触发 React 重渲染）
  modelData: ModelVisualization | null;
  expandedNodes: Set<string>;
  selectedNode: string | null;
  viewMode: '2d' | '3d';
  tourState: { step: number; playing: boolean; speed: number };
  sidebarView: 'model' | 'node' | 'tensor';
}

// 帧级状态（高频，通过 useFrame + ref 管理，绝不放 Zustand）
// particleProgress, hoverGlow, animationTime, cameraTarget
// 粒子位置、shader uniform、动画进度
```

此规则写入 `ARCHITECTURE.md`，作为前端开发硬性约束。

#### Step 2.X: Analytics 集成
- 匿名使用统计（模型搜索量、3D/2D 模式切换比例、Guided Tour 完成率）
- 反馈按钮（"这个可视化有帮助吗？"）
- 实现：轻量级前端埋点 + 后端 `/api/analytics` 端点

#### Step 2.6: 验证
```bash
cd frontend && npm run dev
# 访问 http://localhost:3000
# 输入 moonshotai/Kimi-K2.6 → 看到基本信息
```

---

### ~~Phase 2.5: 2D 可视化模式~~ → 移至 Roadmap v1.2

> **决策**：3D 是核心差异化功能（对标 LLM Viz），优先开发 3D。2D SVG/dagre 可视化移至 v1.2 版本 roadmap。

---

### Phase 3: 3D 架构可视化（核心功能，优先开发）

**Prerequisites**: Phase 2 前端脚手架就绪，Zustand store 可正常获取后端数据，R3F 基础 Canvas 可渲染。

**Definition of Done**: Kimi-K2.6 完整 3D 架构树渲染正常，旋转/缩放流畅（≥30fps on Mac 集成 GPU），点击展开/折叠交互正常，MoE 网格可见。

> **目标设备**: 个人 Mac（Apple Silicon 集成 GPU，无独立显卡）
> **性能策略**: 默认 MeshStandardMaterial + 仅 Bloom 后处理 + 3K 粒子上限

#### Step 3.1: Scene.tsx — 3D 场景容器
```tsx
<Canvas camera={{ position: [0, 5, 15], fov: 50 }} frameloop="demand">
  <ambientLight intensity={0.4} />
  <directionalLight position={[5, 10, 5]} intensity={0.8} />
  <OrbitControls />
  <ModelArchitecture data={modelData} />
  <EffectComposer>
    <Bloom luminanceThreshold={0.8} intensity={1.5} radius={0.4} />
  </EffectComposer>
</Canvas>
```

**Mac 集成 GPU 性能适配**:
- 材质: `MeshStandardMaterial`（去 transmission/clearcoat，性能 2-3x 提升）
- 后处理: 仅 Bloom（去 Vignette、ChromaticAberration、Noise）
- 粒子: 默认 3K（检测到高端 GPU 时升级到 10K）
- 渲染: `frameloop="demand"`（按需渲染，idle 时零 GPU 负载）
- 文字: `<Html>` DOM overlay 优先（减少 GPU draw calls）
- LOD: 距离 >20 时合并层块，>40 时隐藏文字标签

#### Step 3.2: ModelArchitecture.tsx — 架构布局
- 根据 tree 数据计算 3D 位置
- 纵向排列层块（Y 轴递减）
- 多模态分支 X 轴偏移

#### Step 3.3: LayerBlock.tsx — 3D 层块
- `<RoundedBox>` + `<meshPhysicalMaterial>`
- 颜色编码 + hover 发光
- 点击展开子模块（动画）

#### Step 3.4: MoEGrid.tsx — 专家网格
- `<Instances>` 渲染 384 个小块
- 活跃专家高亮 + bloom

#### Step 3.5: ConnectionLines.tsx — 层间连线
- `<QuadraticBezierLine>` 贝塞尔连线

#### Step 3.6: 多模态分支
- Vision encoder 侧面布局
- 投影器连接线

#### 验证
- Kimi-K2.6 完整 3D 架构渲染
- 旋转/缩放流畅
- 点击展开/折叠

---

### Phase 4: 端到端 3D 数据流动画

**Prerequisites**: Phase 3 的 3D 架构树渲染正常，LayerBlock/ConnectionLines 组件可用，GSAP 已引入。

**Definition of Done**: Round 1 动画完整播放（Embedding → 粒子流 → MoE 分流 → LM Head），Guided Tour 基础自动播放正常，基本模型对比（config diff 表格）可用。

> **分轮实现策略**: InputVisualization 和 OutputVisualization 各自相当于独立 3D 应用。
> - **Round 1 (Phase 4)**: 张量块膨胀/收缩 + 简单粒子流 + 基础 Guided Tour
> - **Round 2 (Phase 5 后)**: 精细动画（Tokenizer 切割、概率分布 3D 柱状图、自回归回路等）

> **注意**: 基本模型对比功能（config diff 表格）已移至本阶段实现，完整 3D 分屏对比仍在 Phase 6。

#### Step 4.1: InputVisualization.tsx — 数据预处理 3D 动画

**Round 1 (Phase 4)**:
- **Embedding 查找**: 大矩阵（紫色半透明体）中对应行高亮并"抽出"为向量
- **位置编码**: 简单的 RoPE 标注（静态标签）

**Round 2 (Phase 5 后)**:
- **文字输入**: 3D 浮动文字（troika-three-text）以打字机效果逐字出现
- **Tokenizer 切割**: 文字被"光刃"分段的动画，每段成为独立 3D 对象
- **Token ID 方块**: 每个 token 变成带数字的 3D 小方块，飞入排列
- **位置编码**: RoPE 螺旋动画叠加到向量上
- **多模态输入**: 图像纹理平面飘入 → 16×16 网格切割 → ViT 编码 → PatchMerger 合并

#### Step 4.2: DataFlowParticles.tsx — 模型内部数据流粒子
- Three.js Points + 自定义 vertex/fragment shader
- 粒子沿贝塞尔路径流动，颜色随阶段变化（文本=蓝，视觉=金）
- 层内数据流: MLA 双路径（Q 路径 + KV 路径）并行粒子
- MoE 分流: 粒子从 Gate 分裂为 8 条支流 → 汇合
- Residual 回路: 半透明虚线粒子
- additive blending + Bloom 发光尾迹

#### Step 4.3: OutputVisualization.tsx — 最终输出 3D 动画

**Round 1 (Phase 4)**:
- **LM Head 膨胀**: 张量块从 7168 维急剧膨胀到 163840 维（scale 动画）
- **简单标注**: Top-K token 文字标签 + 概率值

**Round 2 (Phase 5 后)**:
- **概率分布 3D 柱状图**: Top-K token 的 3D 柱子从平面"生长"出来
  - 高度 = 概率值，颜色 = 置信度渐变
  - 最高概率柱子 Bloom 发光
  - 每根柱子上方浮动 token 文字标签
- **采样动画**: 发光粒子从概率分布中"被选中"→ 飞向输出区
- **输出文字**: 新 token 以打字机效果出现在 3D 输出面板
- **自回归回路**: 新 token 反馈，粒子从底部回流到顶部（可选循环）

#### Step 4.4: GuidedTour.tsx — 自动播放 Guided Tour
- GSAP Master Timeline 编排完整推理流程（~75s）
- 相机自动飞越 + 逐步高亮 + tensor shape 标注
- 播放/暂停/进度条/速度控制（0.5x/1x/2x/4x）
- 侧边栏步骤列表，点击跳转到任意步骤
- 借鉴 LLM Viz 的 walkthrough 系统

#### Step 4.5: CameraRig.tsx — 弹簧物理相机
- react-spring 驱动的临界阻尼相机过渡
- 每到一层暂停，展示 tensor shape 3D 标注
- Hover 模块时相机微调对焦

#### Step 4.6: TensorShape3D.tsx — 张量变形动画
- 块的 scale 按维度比例变化（膨胀/收缩/分裂/融合）
- GSAP timeline 编排维度变化
- 高维 = 鲜艳发光，低维 = 暗色 + "压缩" 标注

#### Step 4.7: 基本模型对比（config diff 表格）
- 后端 compare API 返回两模型 config 差异
- 前端渲染 config diff 表格（高亮不同字段）
- 参数量、层数、注意力头数等关键指标并排对比

#### 验证 (Round 1)
- Embedding 查找动画 → 张量块生成
- 粒子从 Embedding 流过所有层到 LM Head
- MoE 层数据分流 → 8 专家并行 → 汇合动画
- LM Head 膨胀 + Top-K 标注
- Guided Tour 基础自动播放，播放/暂停/跳转正常
- Config diff 表格可对比两个模型

---

### Phase 5: MoE/MLA/量化专项

**Prerequisites**: Phase 4 数据流粒子动画基本可用，MoE 网格和 MLA 标注已在 Phase 3 中初步实现。

**Definition of Done**: MoE Gate 管线动画完整，MLA 漏斗 3D 可视化可交互，量化颜色编码覆盖所有支持的量化类型，参数分布 3D 柱状图可见。

#### Step 5.1: MoE 深度交互
- Gate 管线动画（sigmoid → top-K → normalize）
- 共享 vs 路由专家分流

#### Step 5.2: MLAFlow.tsx — 3D 漏斗
- 漏斗几何体（宽→窄→宽）
- 粒子穿越漏斗
- MHA vs MLA 对比面板

#### Step 5.3: 量化颜色编码
- 3D 架构中按量化状态着色

#### Step 5.4: 参数分布
- 3D 柱状图

---

### Phase 6: 模型对比（完整 3D 分屏）

**Prerequisites**: Phase 4 中基本 config diff 表格已可用，Phase 5 专项可视化完成，后端 compare API 就绪。

**Definition of Done**: 3D 并排对比可同步旋转/缩放，差异高亮（Bloom 颜色区分）正常，2D 分屏对比 SVG 面板就绪。

#### Step 6.1: 后端 compare API
#### Step 6.2: 2D 分屏对比（借鉴 Model Explorer）
- SplitPane.tsx: 两个独立 SVG 面板 + 可拖拽分隔条
- 同步导航: 选中一侧节点自动匹配另一侧
- 差异高亮: 删除=红色边框，新增=绿色边框
- 自定义映射 JSON 上传（1:1, 1:N, N:M）
#### Step 6.3: 3D 并排对比
- 两个 Canvas 并排
- 同步旋转/缩放
- 差异高亮（Bloom 颜色区分）

---

[← 返回目录](README.md)
