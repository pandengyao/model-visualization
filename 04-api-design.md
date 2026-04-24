# 五、API 设计

> [HF Model Visualizer](README.md) 技术设计文档 — 章五

### 5.1 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/model/{org}/{repo}` | 完整可视化数据 |
| GET | `/api/v1/model/{org}/{repo}/config` | 仅配置 + 检测 |
| GET | `/api/v1/model/{org}/{repo}/tree` | 仅架构树 |
| GET | `/api/v1/model/{org}/{repo}/params` | 仅参数统计 |
| POST | `/api/v1/compare` | 两模型对比 |
| GET | `/api/v1/popular` | 预缓存热门模型列表 |

### 5.2 完整 API 契约（Pydantic Schema）

> 基于 `explore_model.py` 实际输出结构设计，确保前后端契约精确一致。
> 所有 schema 对应后端 `app/models/schemas.py` 文件。

#### 5.2.1 统一错误响应

```python
class ErrorResponse(BaseModel):
    """所有错误统一使用此格式"""
    error: str                                # 错误消息
    code: str                                 # 错误代码: "not_found", "auth_required", "rate_limited", "parse_error", "internal"
    fix: str | None = None                    # 用户可操作的修复建议
```

#### 5.2.2 主响应模型

```python
class ModelVisualization(BaseModel):
    """GET /api/v1/model/{org}/{repo} 的完整响应"""

    # === 基础信息 ===
    model_name: str                           # 输入的模型标识，如 "moonshotai/Kimi-K2.6"
    model_class: str | None = None            # 模型类名（meta 加载成功时），如 "KimiK25ForConditionalGeneration"

    # === 模型卡片 ===
    model_card: ModelCard | None = None       # HF 模型卡片信息（API 调用失败时为 None）

    # === 配置与检测 ===
    key_config: KeyConfig                     # 关键配置（结构化）
    moe_info: MoEInfo                         # MoE 检测结果（is_moe=False 时其余字段缺省）
    mla_info: MLAInfo                         # MLA 检测结果（is_mla=False 时其余字段缺省）
    quant_info: QuantInfo                     # 量化检测结果（is_quantized=False 时其余字段缺省）
    sub_configs: dict[str, SubConfig] | None = None  # 多模态子配置

    # === 架构树 ===
    tree: TreeNode                            # 嵌套架构树（根节点）
    tree_text: str                            # ASCII 文本树
    is_synthetic: bool = False                # True 表示合成树（非真实模型加载）

    # === 参数统计 ===
    params: ParamStats                        # 参数统计（结构化，见下方两种 variant）

    # === 推理数据流 ===
    data_flow: list[FlowStep]                # 端到端推理步骤列表

    # === 元信息 ===
    warning: str | None = None                # 加载警告信息
    config_transformers_version: str | None = None  # config 中记录的 transformers 版本

    model_config = ConfigDict(extra="ignore")
```

#### 5.2.3 TreeNode（嵌套递归树）

```python
class TreeNode(BaseModel):
    """架构树节点 — 递归嵌套结构"""

    name: str                                 # 模块名称，如 "self_attn", "layers.0"
    type: str                                 # 模块类型/类名，如 "DeepseekV3Attention"
    own_params: int                           # 该模块直接拥有的参数量（不含子模块）
    total_params: int                         # 该模块及所有后代的总参数量
    children: list["TreeNode"] = []           # 子节点列表

    # === 前端可视化辅助字段（后端填充） ===
    module_category: str | None = None        # 语义分类: "attention", "mlp", "moe", "norm",
                                              #   "embedding", "linear", "vision", "projector", "other"
    params_str: str | None = None             # 格式化参数量，如 "101.12M"
    is_repeated: bool = False                 # 是否为重复结构（如 layer 1-60 相同）
    repeat_count: int | None = None           # 重复次数

    model_config = ConfigDict(extra="ignore")

# 说明：
# - 真实模型加载(meta device)时: name=模块名, type=类名, own_params/total_params 精确
# - 合成树(config估算)时: name 和 type 可能相同, 参数量为估算值
# - module_category 由后端根据 type/name 自动推断，前端用于颜色编码
```

#### 5.2.4 参数统计（双形态）

```python
class TopModule(BaseModel):
    """参数量 Top 模块"""
    name: str
    params: int
    percentage: float                         # 占总参数量的百分比，如 42.35
    params_str: str                           # "101.12M"

class MoEActiveParams(BaseModel):
    """MoE 模型的激活参数估算"""
    total: int
    total_str: str
    experts_per_token: int

class ParamBreakdown(BaseModel):
    """config 估算时的参数分解"""
    embedding: int | None = None
    lm_head: int | None = None
    per_layer: int | None = None              # Dense 模型: 每层参数
    attention_per_layer: int | None = None     # MoE 模型: 每层 Attention 参数
    dense_layers: DenseLayerBreakdown | None = None  # MoE 模型前 K 层
    moe_layers: MoELayerBreakdown | None = None      # MoE 层
    vision_encoder: VisionBreakdown | None = None     # 多模态视觉编码器
    projector: int | None = None              # 多模态投影器
    active_params: MoEActiveParams | None = None  # MoE 激活参数

class DenseLayerBreakdown(BaseModel):
    count: int
    params_per_layer: int
    total: int

class MoELayerBreakdown(BaseModel):
    count: int
    params_per_layer: int
    total: int
    routed_experts: int
    shared_experts: int
    expert_mlp: int
    moe_intermediate_size: int

class VisionBreakdown(BaseModel):
    hidden_size: int
    num_layers: int
    intermediate_size: int
    patch_embed: int
    per_layer: int
    total: int

class ParamStats(BaseModel):
    """参数统计 — 根据加载方式有两种形态，统一为一个模型"""

    # === 精确统计（meta 加载成功时填充） ===
    total: int | None = None                  # 精确总参数量
    total_str: str | None = None              # "1026.85B"
    trainable: int | None = None
    trainable_str: str | None = None
    non_trainable: int | None = None
    size_mb: float | None = None              # 模型大小 MB
    size_str: str | None = None               # "14.25 GB"
    dtype_distribution: dict[str, str] | None = None  # {"torch.bfloat16": "1026.85B"}
    top_modules: list[TopModule] | None = None
    moe_note: str | None = None               # MoE 特殊说明

    # === 估算统计（config-only 或 meta 加载失败时填充） ===
    estimated_total: int | None = None
    estimated_total_str: str | None = None    # "1026.85B"
    breakdown: ParamBreakdown | None = None
    estimate_note: str | None = None          # "估算值，实际参数量可能略有差异"

    # === 前端辅助 ===
    is_estimated: bool = False                # True 表示使用估算值

    @property
    def display_total_str(self) -> str:
        """前端统一获取展示用的总参数量字符串"""
        return self.total_str or self.estimated_total_str or "N/A"
```

#### 5.2.5 MoE / MLA / 量化 检测结果

```python
class MoEExpertCounts(BaseModel):
    total_routed: int
    per_token: int | None = None
    shared: int = 0

class MoERouting(BaseModel):
    topk_method: str | None = None            # "noaux_tc", "greedy"
    topk_group: int | None = None
    n_group: int | None = None
    scoring_func: str | None = None           # "sigmoid", "softmax"
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None

class MoEInfo(BaseModel):
    is_moe: bool
    expert_counts: MoEExpertCounts | None = None  # None when is_moe=False
    architecture: dict | None = None          # moe_intermediate_size, first_k_dense_replace 等
    routing: MoERouting | None = None

class MLAInfo(BaseModel):
    is_mla: bool
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None

    @property
    def kv_cache_dim(self) -> int | None:
        """KV cache 压缩维度 = kv_lora_rank + qk_rope_head_dim"""
        if self.kv_lora_rank is not None and self.qk_rope_head_dim is not None:
            return self.kv_lora_rank + self.qk_rope_head_dim
        return None

class QuantInfo(BaseModel):
    is_quantized: bool
    method: str | None = None                 # "compressed-tensors", "fp8", "gptq", "awq", "bitsandbytes"
    bits: int | None = None                   # 4, 8, 16
    group_size: int | None = None
    strategy: str | None = None               # "group" etc.
    symmetric: bool | None = None
    format: str | None = None                 # "pack-quantized" etc.
    ignore_patterns: list[str] | None = None  # 不量化的模块正则
    activation_scheme: str | None = None      # FP8 专用
    quant_type: str | None = None             # BnB 专用
    skip_modules: list[str] | None = None     # BnB 专用
```

#### 5.2.6 ModelCard

```python
class ModelCard(BaseModel):
    model_name: str
    model_id: str | None = None
    author: str | None = None
    tags: list[str] = []
    pipeline_tag: str | None = None
    library_name: str | None = None
    license: str | None = None
    downloads: int | None = None
    likes: int | None = None
    introduction: str | None = None           # README 摘要（前端展示用）
    required_transformers_version: str | None = None
```

#### 5.2.7 KeyConfig（结构化关键配置）

```python
class KeyConfig(BaseModel):
    """从 extract_key_config() 提取的关键配置，全部字段可选"""
    architectures: list[str] | None = None
    model_type: str | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    torch_dtype: str | None = None
    tie_word_embeddings: bool | None = None
    rope_type: str | None = None
    rope_theta: float | None = None
    sliding_window: int | None = None
    head_dim: int | None = None

    # MLA 相关
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None

    # MoE 相关
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    num_local_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    first_k_dense_replace: int | None = None
    moe_layer_freq: int | None = None

    # RoPE Scaling
    rope_scaling: dict | None = None

    model_config = ConfigDict(extra="allow")  # 允许未列出的字段通过
```

#### 5.2.8 SubConfig（多模态子配置）

```python
class SubConfig(BaseModel):
    """多模态子编码器配置"""
    model_type: str | None = None
    architectures: list[str] | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    vocab_size: int | None = None

    model_config = ConfigDict(extra="allow")  # 子配置字段不可预知
```

#### 5.2.9 FlowStep（推理数据流步骤）

```python
class TensorShape(BaseModel):
    """结构化的张量形状"""
    dims: list[int | str]                     # 如 ["B", "S", 7168] 或 ["B", 3, "H", "W"]
    dtype: str | None = None                  # "bfloat16"
    description: str | None = None            # "hidden states after embedding"

class FlowStep(BaseModel):
    """端到端推理的单个步骤"""
    step: int                                 # 步骤序号
    name: str                                 # 步骤名称，如 "Embedding Lookup"
    module_path: str | None = None            # 对应 TreeNode 路径，如 "model.embed_tokens"
    module_type: str                          # "embedding" | "attention" | "mlp" | "moe" | "norm" |
                                              #   "lm_head" | "vision" | "projector" | "tokenizer" | "sampling"
    input_shapes: list[TensorShape]           # 输入张量形状列表
    output_shapes: list[TensorShape]          # 输出张量形状列表
    computation: str | None = None            # 计算描述，如 "Q×K^T/√d → Softmax → ×V"
    branch: str | None = None                 # "main" | "vision" | "audio" | None

    # === 数据流可视化辅助 ===
    is_repeated: bool = False                 # 是否为重复步骤（如 60 个 MoE 层）
    repeat_count: int | None = None           # 重复次数
    sub_steps: list["FlowStep"] | None = None # 层内子步骤（展开时显示）
    details: dict | None = None               # 附加信息（如专家数、head 数等）

    # === 准确性标注 ===
    confidence: str = "high"                  # "verified" | "high" | "inferred"
```

#### 5.2.10 模型对比响应

```python
class CompareRequest(BaseModel):
    model_a: str
    model_b: str
    config_only: bool = False
    trust_remote_code: bool = False
    token: str | None = None

class ConfigDiff(BaseModel):
    key: str
    model_a: str                              # 值的字符串表示
    model_b: str

class ConfigSame(BaseModel):
    key: str
    value: str

class ModuleParamCompare(BaseModel):
    module: str
    model_a_params: int
    model_a_str: str
    model_b_params: int
    model_b_str: str

class CompareResponse(BaseModel):
    model_a: ModelVisualization
    model_b: ModelVisualization
    config_diff: list[ConfigDiff]
    config_same: list[ConfigSame]

    # 以下仅在非 config-only 模式下返回
    param_comparison: ParamComparison | None = None
    structural_diff: StructuralDiff | None = None

class ParamComparison(BaseModel):
    model_a_total: int | None = None
    model_a_total_str: str
    model_b_total: int | None = None
    model_b_total_str: str
    ratio: float | None = None                # model_b / model_a
    per_module: list[ModuleParamCompare] = []
    note: str | None = None                   # "一个或两个模型使用估算值"

class StructuralDiff(BaseModel):
    only_in_a: list[str]                      # 仅在 model_a 中的模块路径（最多 30）
    only_in_b: list[str]
    type_changes: list[dict]                  # {"module": str, "model_a_type": str, "model_b_type": str}
    total_modules_a: int
    total_modules_b: int
```

#### 5.2.11 设计决策说明

| 决策 | 理由 |
|---|---|
| TreeNode 使用嵌套结构 | 前端需要层级展开/折叠，嵌套树是最自然的数据结构 |
| ParamStats 统一双形态 | meta 加载和 config 估算返回不同字段，统一为一个模型 + `is_estimated` 标志 |
| TensorShape 使用结构化 dims | 前端需要按维度渲染 3D 块体大小，字符串格式 "(B,S,7168)" 需要额外解析 |
| FlowStep 支持 sub_steps | 层内子步骤（Attention → Q/K/V 分解 → Softmax → Output）需要嵌套 |
| 所有检测结果非 None | MoEInfo/MLAInfo/QuantInfo 始终返回，通过 `is_moe`/`is_mla`/`is_quantized` 标志区分 |
| KeyConfig 使用 `extra="allow"` | HF config.json 字段不可穷举，允许透传新增字段 |
| ErrorResponse 统一格式 | 前端可统一处理所有错误类型 |
```

### 5.3 缓存策略

```
三层缓存架构:

L0: 进程内 LRU (maxsize=100, ttl=300s)
    — 零延迟热点缓存，最近请求的模型直接命中
    — 重启后失效

L1: 文件系统 JSON (/data/cache/{model_id_hash}.json)
    — 持久化跨重启，24h TTL
    — cache_key = f"{model_id}:{model_info.sha}" (用 HF commit SHA 做版本键)

L2: HF Hub API + 内置缓存 (~/.cache/huggingface/hub/)
    — 200-500ms 延迟，带 ETag 条件请求
    — 最终数据源
```

#### 5.3.1 L1 缓存并发安全

多 worker 部署时，L1 文件缓存需要处理并发写入问题：

```python
import tempfile, os

def safe_write_cache(cache_path: str, data: bytes):
    """原子写入：写临时文件 + rename，避免读到半写数据"""
    dir_name = os.path.dirname(cache_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        os.write(fd, data)
        os.close(fd)
        os.replace(tmp_path, cache_path)  # 原子操作
    except:
        os.close(fd)
        os.unlink(tmp_path)
        raise
```

替代方案：使用 `filelock` 库实现文件级锁（适用于高并发场景）。

#### HF Hub 降级策略

100% 数据来自 HF Hub，是单点故障，必须有降级方案：

| 状态 | 处理方式 |
|---|---|
| HF 正常 | 正常流程，写入 L0 + L1 |
| HF 限流 (429) | 返回 L1 缓存 + 响应标记 `"stale": true, "note": "数据可能非最新"` |
| HF 宕机 (5xx) | 返回 L1 缓存 + 响应标记 `"stale": true, "note": "HF Hub 暂时不可用"` |
| L1 也无缓存 | 返回 503 + `ErrorResponse(code="hub_unavailable", fix="请稍后重试")` |

启动时异步预热 Top-100 热门模型到 L0 + L1。

> **config-only 模式字段差异**: 当模型以 config-only 模式加载时（meta-device 加载失败的降级），以下字段为 `null` 或估算值：
> - `tree.own_params` / `tree.total_params`: 估算值（`is_estimated=true`）
> - `params.dtype_distribution`: `null`
> - `params.is_estimated`: `true`
> - `flow_steps`: `null`（无法从 config 生成数据流）

### 5.4 安全规则

1. **`trust_remote_code` 安全边界**:
   - 默认 `False`，仅支持标准 transformers 架构
   - 自定义架构模型（如 Kimi-K2.6）需用户主动开启
   - 用户开启时**前端显示警告弹窗**: "此模型使用自定义代码，将在服务端执行远程代码"
   - v1.1: Docker 容器内沙箱隔离执行 trust_remote_code 请求
2. **只下载 config.json、README.md 和 model.safetensors.index.json** — 使用 `torch.device("meta")` 零权重加载，永不下载实际模型权重文件（.safetensors / .bin）
3. **model_id 注入防护**:
   - 正则验证: `^[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)?$`
   - 长度限制: 最大 200 字符
   - 规范化: `strip()` + 统一 URL 编码
4. **Rate limit**: 30 req/min (匿名), 60 req/min (认证)
5. **缓存 404 结果** (1h) — 防止恶意重复查询

### 5.5 大模型加载进度推送

对于超大模型（如 1T 参数的 Kimi-K2.6），meta-device 加载可能需要 10-30 秒。为避免用户看到白屏，提供加载进度推送：

#### 方案：SSE (Server-Sent Events)

```
GET /api/model/{model_id}/progress
Accept: text/event-stream

event: progress
data: {"stage": "downloading_config", "percent": 20, "message": "下载 config.json..."}

event: progress
data: {"stage": "loading_model", "percent": 50, "message": "加载模型结构 (meta device)..."}

event: progress
data: {"stage": "detecting_features", "percent": 80, "message": "检测 MoE/MLA/量化特征..."}

event: complete
data: {"model_id": "moonshotai/Kimi-K2.6", "redirect": "/api/model/moonshotai%2FKimi-K2.6"}
```

#### 前端集成

```typescript
const evtSource = new EventSource(`/api/model/${encodeURIComponent(modelId)}/progress`);
evtSource.addEventListener('progress', (e) => {
  const { stage, percent, message } = JSON.parse(e.data);
  setLoadingState({ stage, percent, message });
});
evtSource.addEventListener('complete', (e) => {
  evtSource.close();
  fetchModelData(modelId);
});
```

---

[← 返回目录](README.md)