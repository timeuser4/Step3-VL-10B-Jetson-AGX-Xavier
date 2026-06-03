#!/bin/bash
# PyTorch/transformers 方案安装脚本
# 在 Jetson 上执行
set -e

echo "=========================================="
echo "Step3-VL-10B PyTorch/transformers 方案安装"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES_DIR="$SCRIPT_DIR/files"

# === 1. 创建 Qwen3 模块 ===
echo "[1/6] 创建 Qwen3 模块..."
QWEN3_DIR=~/.local/lib/python3.8/site-packages/transformers/models/qwen3
mkdir -p "$QWEN3_DIR"
cp "$FILES_DIR/qwen3_init.py" "$QWEN3_DIR/__init__.py"
cp "$FILES_DIR/qwen3_config.py" "$QWEN3_DIR/configuration_qwen3.py"
cp "$FILES_DIR/qwen3_modeling.py" "$QWEN3_DIR/modeling_qwen3.py"
echo "  Qwen3 模块已创建: $QWEN3_DIR"

# === 2. 注册到 transformers auto classes ===
echo "[2/6] 注册 Qwen3 到 transformers auto classes..."

TRANSFORMERS_DIR=~/.local/lib/python3.8/site-packages/transformers

# __init__.py 添加导入
python3 -c "
f = '$TRANSFORMERS_DIR/__init__.py'
content = open(f).read()
if 'Qwen3Config' not in content:
    # 在 Qwen2 导入附近添加
    content = content.replace(
        'from .models.qwen2 import',
        'from .models.qwen3 import Qwen3Config, Qwen3Model, Qwen3ForCausalLM\nfrom .models.qwen2 import'
    )
    open(f, 'w').write(content)
    print('  __init__.py: 已添加 Qwen3 导入')
else:
    print('  __init__.py: Qwen3 导入已存在')
"

# configuration_auto.py
python3 -c "
f = '$TRANSFORMERS_DIR/models/auto/configuration_auto.py'
content = open(f).read()
if 'qwen3' not in content:
    content = content.replace(
        '(\"qwen2\", \"Qwen2Config\"),',
        '(\"qwen2\", \"Qwen2Config\"),\n        (\"qwen3\", \"Qwen3Config\"),'
    )
    content = content.replace(
        '(\"qwen2\", \"Qwen2\"),',
        '(\"qwen2\", \"Qwen2\"),\n        (\"qwen3\", \"Qwen3\"),'
    )
    open(f, 'w').write(content)
    print('  configuration_auto.py: 已注册 Qwen3Config')
else:
    print('  configuration_auto.py: 已存在')
"

# modeling_auto.py
python3 -c "
f = '$TRANSFORMERS_DIR/models/auto/modeling_auto.py'
content = open(f).read()
if 'Qwen3ForCausalLM' not in content:
    content = content.replace(
        'from ..qwen2.modeling_qwen2 import',
        'from ..qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model\nfrom ..qwen2.modeling_qwen2 import'
    )
    # 添加到 MODEL_MAPPING
    content = content.replace(
        '(Qwen2Config, Qwen2ForCausalLM),',
        '(Qwen2Config, Qwen2ForCausalLM),\n        (Qwen3Config, Qwen3ForCausalLM),'
    )
    open(f, 'w').write(content)
    print('  modeling_auto.py: 已注册 Qwen3ForCausalLM')
else:
    print('  modeling_auto.py: 已存在')
"

# === 3. 修复兼容性 ===
echo "[3/6] 修复兼容性问题..."

# modeling_utils.py - metadata None 检查
python3 -c "
f = '$TRANSFORMERS_DIR/modeling_utils.py'
content = open(f).read()
if 'metadata.get()' in content and 'is not None' not in content[:content.find('metadata.get')+200]:
    content = content.replace(
        'metadata.get(',
        '(metadata or {}).get('
    )
    open(f, 'w').write(content)
    print('  modeling_utils.py: 已修复 metadata None 检查')
else:
    print('  modeling_utils.py: 已修复或不需要')
"

# === 4. 修改 Step3-VL 模型代码 ===
echo "[4/6] 修改 Step3-VL 模型代码..."
STEP3VL_DIR=$(find ~/.cache/huggingface/modules/transformers_modules/stepfun-ai/Step3-VL-10B -name "modeling_step_vl.py" 2>/dev/null | head -1)
if [ -n "$STEP3VL_DIR" ]; then
    cp "$FILES_DIR/modeling_step_vl.py" "$STEP3VL_DIR"
    echo "  modeling_step_vl.py: 已替换"
else
    echo "  警告: 未找到 Step3-VL 模型代码目录"
fi

# === 5. 修复 vision_encoder.py ===
echo "[5/6] 修复 vision_encoder.py 类型注解..."
VISION_ENC=$(find ~/.cache/huggingface/modules/transformers_modules/stepfun-ai/Step3-VL-10B -name "vision_encoder.py" 2>/dev/null | head -1)
if [ -n "$VISION_ENC" ]; then
    cp "$FILES_DIR/vision_encoder.py" "$VISION_ENC"
    echo "  vision_encoder.py: 已替换"
else
    echo "  警告: 未找到 vision_encoder.py"
fi

# === 6. 验证 ===
echo "[6/6] 验证安装..."
python3 -c "
from transformers.models.qwen3 import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
print('Qwen3 模块导入成功')
print(f'  Qwen3Config: {Qwen3Config}')
print(f'  Qwen3Model: {Qwen3Model}')
print(f'  Qwen3ForCausalLM: {Qwen3ForCausalLM}')
"

echo ""
echo "=========================================="
echo "安装完成！"
echo ""
echo "运行推理:"
echo "  python3 $SCRIPT_DIR/deploy.py --test"
echo "=========================================="
