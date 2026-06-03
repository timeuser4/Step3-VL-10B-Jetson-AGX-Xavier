#!/bin/bash
# 路线 A: 自动应用 Safetensors 原生部署的所有补丁
# 在 Jetson 上执行: bash apply_patches.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES_DIR="$SCRIPT_DIR/files"

echo "=========================================="
echo "应用 Safetensors 部署补丁"
echo "=========================================="

# === 1. 创建 Qwen3 模块 ===
echo "[1/5] 创建 Qwen3 模块..."
QWEN3_DIR=~/.local/lib/python3.8/site-packages/transformers/models/qwen3
mkdir -p "$QWEN3_DIR"
cp "$FILES_DIR/qwen3_init.py" "$QWEN3_DIR/__init__.py"
cp "$FILES_DIR/qwen3_config.py" "$QWEN3_DIR/configuration_qwen3.py"
cp "$FILES_DIR/qwen3_modeling.py" "$QWEN3_DIR/modeling_qwen3.py"
echo "  ✓ Qwen3 模块已创建"

# === 2. 注册到 transformers auto classes ===
echo "[2/5] 注册 Qwen3 到 transformers auto classes..."
TF_DIR=~/.local/lib/python3.8/site-packages/transformers

# __init__.py
python3 -c "
f = '$TF_DIR/__init__.py'
c = open(f).read()
if 'Qwen3Config' not in c:
    c = c.replace(
        'from .models.qwen2 import',
        'from .models.qwen3 import Qwen3Config, Qwen3Model, Qwen3ForCausalLM\nfrom .models.qwen2 import'
    )
    open(f, 'w').write(c)
    print('  ✓ __init__.py')
else:
    print('  - __init__.py (已存在)')
"

# configuration_auto.py
python3 -c "
f = '$TF_DIR/models/auto/configuration_auto.py'
c = open(f).read()
if 'qwen3' not in c:
    c = c.replace(
        '(\"qwen2\", \"Qwen2Config\"),',
        '(\"qwen2\", \"Qwen2Config\"),\n        (\"qwen3\", \"Qwen3Config\"),'
    )
    c = c.replace(
        '(\"qwen2\", \"Qwen2\"),',
        '(\"qwen2\", \"Qwen2\"),\n        (\"qwen3\", \"Qwen3\"),'
    )
    open(f, 'w').write(c)
    print('  ✓ configuration_auto.py')
else:
    print('  - configuration_auto.py (已存在)')
"

# modeling_auto.py
python3 -c "
f = '$TF_DIR/models/auto/modeling_auto.py'
c = open(f).read()
if 'Qwen3ForCausalLM' not in c:
    c = c.replace(
        'from ..qwen2.modeling_qwen2 import',
        'from ..qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model\nfrom ..qwen2.modeling_qwen2 import'
    )
    c = c.replace(
        '(Qwen2Config, Qwen2ForCausalLM),',
        '(Qwen2Config, Qwen2ForCausalLM),\n        (Qwen3Config, Qwen3ForCausalLM),'
    )
    open(f, 'w').write(c)
    print('  ✓ modeling_auto.py')
else:
    print('  - modeling_auto.py (已存在)')
"

# tokenization_auto.py
python3 -c "
f = '$TF_DIR/models/auto/tokenization_auto.py'
c = open(f).read()
if 'qwen3' not in c:
    c = c.replace(
        '(\"qwen2\", (\"Qwen2Tokenizer\",',
        '(\"qwen3\", (\"Qwen2Tokenizer\", \"Qwen2TokenizerFast\")),\n        (\"qwen2\", (\"Qwen2Tokenizer\",'
    )
    open(f, 'w').write(c)
    print('  ✓ tokenization_auto.py')
else:
    print('  - tokenization_auto.py (已存在)')
"

# === 3. 修复 modeling_utils.py ===
echo "[3/5] 修复 modeling_utils.py 兼容性..."
python3 -c "
f = '$TF_DIR/modeling_utils.py'
c = open(f).read()
if 'metadata.get(' in c and '(metadata or {})' not in c:
    c = c.replace('metadata.get(', '(metadata or {}).get(')
    open(f, 'w').write(c)
    print('  ✓ metadata None 检查已修复')
else:
    print('  - 已修复或不需要')
"

# === 4. 替换 Step3-VL 模型代码 ===
echo "[4/5] 替换 Step3-VL 模型代码..."
STEP3VL_DIR=$(find ~/.cache/huggingface/modules/transformers_modules/stepfun-ai/Step3-VL-10B -name "modeling_step_vl.py" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
if [ -n "$STEP3VL_DIR" ]; then
    cp "$FILES_DIR/modeling_step_vl.py" "$STEP3VL_DIR/modeling_step_vl.py"
    cp "$FILES_DIR/configuration_step_vl.py" "$STEP3VL_DIR/configuration_step_vl.py"
    cp "$FILES_DIR/vision_encoder.py" "$STEP3VL_DIR/vision_encoder.py"
    echo "  ✓ Step3-VL 模型代码已替换"
else
    echo "  ✗ 未找到 Step3-VL 模型目录，请先下载模型"
    exit 1
fi

# === 5. 验证 ===
echo "[5/5] 验证..."
python3 -c "
from transformers.models.qwen3 import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
print('  ✓ Qwen3 模块导入成功')
"

echo ""
echo "=========================================="
echo "所有补丁已应用！"
echo "运行推理: python3 $SCRIPT_DIR/deploy.py --test"
echo "=========================================="
