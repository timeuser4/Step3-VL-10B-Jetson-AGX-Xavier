# 路线 A: Safetensors 补丁说明

## 自动应用

```bash
bash apply_patches.sh
python3 deploy.py --test
```

## 补丁列表

| 补丁 | 修改文件 | 问题 | 解决 |
|------|----------|------|------|
| 创建 Qwen3 模块 | `transformers/models/qwen3/` | transformers 4.46.3 没有 Qwen3 | 基于 Qwen2 创建完整 Qwen3 模块 |
| 注册 auto classes | `transformers/__init__.py` 等 4 个文件 | Qwen3 未注册到 auto mapping | 添加 Config/Model/Tokenizer 映射 |
| 修复 metadata | `transformers/modeling_utils.py` | `metadata.get()` 在 None 时崩溃 | 改为 `(metadata or {}).get()` |
| 替换 Step3-VL 代码 | `modeling_step_vl.py` 等 3 个文件 | import 路径和类型注解不兼容 | 修正 import、修复 `tuple[int]` 语法 |

## 补丁详情

### 1. 创建 Qwen3 模块

**目录**: `~/.local/lib/python3.8/site-packages/transformers/models/qwen3/`

Step3-VL 基于 Qwen3 架构，但 transformers 4.46.3 只有 Qwen2。Qwen3 与 Qwen2 的关键差异：

| 差异 | Qwen3 | Qwen2 |
|------|-------|-------|
| `q_norm`/`k_norm` (RMSNorm on query/key) | ✅ 有 | ❌ 无 |
| `q_proj.bias` | ❌ 无 bias | ✅ 有 bias |
| `k_proj.bias` | ❌ 无 bias | ✅ 有 bias |
| `v_proj.bias` | ❌ 无 bias | ✅ 有 bias |

核心代码 (`modeling_qwen3.py`):
```python
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        self.q_proj = nn.Linear(..., bias=False)  # 无 bias
        self.k_proj = nn.Linear(..., bias=False)
        self.v_proj = nn.Linear(..., bias=False)
        self.q_norm = RMSNorm(self.head_dim)  # Qwen3 特有
        self.k_norm = RMSNorm(self.head_dim)  # Qwen3 特有
```

### 2. 注册到 transformers auto classes

需要修改 4 个文件：

- `transformers/__init__.py` — 添加 Qwen3 导入
- `transformers/models/auto/configuration_auto.py` — `("qwen3", "Qwen3Config")`
- `transformers/models/auto/modeling_auto.py` — `(Qwen3Config, Qwen3ForCausalLM)`
- `transformers/models/auto/tokenization_auto.py` — `("qwen3", ("Qwen2Tokenizer", ...))`

### 3. 修复 modeling_utils.py

**问题**: safetensors 元数据可能为 None，`metadata.get()` 会崩溃。

**修复**: `metadata.get(...)` → `(metadata or {}).get(...)`

### 4. 替换 Step3-VL 模型代码

**modeling_step_vl.py**:
- `from transformers import Qwen2Model as Qwen3Model` → `from transformers.models.qwen3 import Qwen3Model`
- forward 签名添加 `**kwargs`

**vision_encoder.py**:
- Python 3.9+ 类型注解修复: `tuple[int]` → `Tuple[int]`

## 源文件说明

| 文件 | 来源 | 说明 |
|------|------|------|
| `qwen3_modeling.py` | 新建 | Qwen3 模型实现 (基于 Qwen2 扩展) |
| `qwen3_config.py` | 新建 | Qwen3 配置类 |
| `qwen3_init.py` | 新建 | Qwen3 模块导出 |
| `modeling_step_vl.py` | 修改 | Step3-VL 模型代码 (修正 import) |
| `configuration_step_vl.py` | 原样 | Step3-VL 配置 |
| `vision_encoder.py` | 修改 | 视觉编码器 (修复类型注解) |
