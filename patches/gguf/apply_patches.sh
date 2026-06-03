#!/bin/bash
# 路线 B: 自动应用 llama.cpp 的兼容性补丁
# 在 Jetson 上执行: bash apply_patches.sh
# 前提: 已克隆 llama.cpp 到 ~/llama.cpp
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "应用 llama.cpp 兼容性补丁"
echo "=========================================="

LLAMA_DIR=~/llama.cpp

if [ ! -d "$LLAMA_DIR" ]; then
    echo "错误: $LLAMA_DIR 不存在，请先克隆 llama.cpp"
    echo "  git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp"
    exit 1
fi

# === Patch 1: float8 兼容性 ===
echo "[1/2] 应用 float8 兼容性补丁..."
python3 -c "
f = '$LLAMA_DIR/conversion/base.py'
c = open(f).read()
if 'float8_e4m3fn' not in c:
    old = 'import torch'
    new = '''import torch

# float8 compatibility shim for PyTorch < 2.1
# Jetson Xavier with JetPack 5.x ships PyTorch 2.1.0a0 which lacks float8 types
if not hasattr(torch, \"float8_e4m3fn\"):
    torch.float8_e4m3fn = torch.uint8  # type: ignore
if not hasattr(torch, \"float8_e5m2\"):
    torch.float8_e5m2 = torch.uint8  # type: ignore
'''
    c = c.replace(old, new, 1)
    open(f, 'w').write(c)
    print('  ✓ float8 兼容性补丁已应用')
else:
    print('  - 已应用')
"

# === Patch 2: removesuffix 兼容性 ===
echo "[2/2] 应用 removesuffix 兼容性补丁..."
python3 -c "
f = '$LLAMA_DIR/conversion/step3.py'
c = open(f).read()
if 'removesuffix' in c:
    c = c.replace(
        'name = name.removesuffix(\".gamma\") + \".weight\"',
        'name = name[:-len(\".gamma\")] + \".weight\" if name.endswith(\".gamma\") else name'
    )
    open(f, 'w').write(c)
    print('  ✓ removesuffix 兼容性补丁已应用')
else:
    print('  - 已应用')
"

echo ""
echo "=========================================="
echo "所有补丁已应用！"
echo "下一步: cd ~/llama.cpp/build && cmake .. -DGGML_CUDA=ON && make -j\$(nproc)"
echo "=========================================="
