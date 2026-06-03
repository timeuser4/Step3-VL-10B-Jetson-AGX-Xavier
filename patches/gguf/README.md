# 路线 B: llama.cpp 补丁说明

## 补丁列表

| 文件 | 问题 | 解决 |
|------|------|------|
| `01_float8_compat.patch` | PyTorch 2.1.0a0 缺少 float8 类型 | 添加 shim 映射到 uint8 |
| `02_removesuffix_compat.patch` | Python 3.8 不支持 removesuffix | 替换为切片操作 |

## 自动应用

```bash
bash apply_patches.sh
```

## 补丁详情

### 补丁 1: float8 兼容性

**文件**: `llama.cpp/conversion/base.py`

**问题**: `convert_hf_to_gguf.py` 在加载时会访问 `torch.float8_e4m3fn` 和 `torch.float8_e5m2`，但 Jetson 的 PyTorch 2.1.0a0 (Jetson 专用版) 没有这两个类型。

**报错**:
```
AttributeError: module 'torch' has no attribute 'float8_e4m3fn'
```

**修复**: 在 `conversion/base.py` 开头添加 shim，将 float8 类型映射到 uint8（转换时不会真正用到 float8 精度，只是类型检查通过即可）。

```python
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.uint8
if not hasattr(torch, "float8_e5m2"):
    torch.float8_e5m2 = torch.uint8
```

### 补丁 2: removesuffix 兼容性

**文件**: `llama.cpp/conversion/step3.py`

**问题**: `str.removesuffix()` 是 Python 3.9+ 的方法，Jetson 的 Python 3.8 不支持。

**报错**:
```
AttributeError: 'str' object has no attribute 'removesuffix'
```

**修复**: 替换为等效的切片操作。

```python
# 修改前
name = name.removesuffix(".gamma") + ".weight"

# 修改后
name = name[:-len(".gamma")] + ".weight" if name.endswith(".gamma") else name
```
