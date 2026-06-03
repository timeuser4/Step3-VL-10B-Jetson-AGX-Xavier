#!/bin/bash
# Step3-VL-10B on Jetson Xavier - Step 1: Build llama.cpp
# 在 Jetson 上执行此脚本
set -e

echo "=========================================="
echo "Step 1: 编译 llama.cpp (CUDA)"
echo "=========================================="

# 安装依赖
echo "[1/5] 安装依赖..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget

# 克隆 llama.cpp
echo "[2/5] 克隆 llama.cpp..."
cd ~
if [ -d "llama.cpp" ]; then
    echo "llama.cpp 目录已存在，跳过克隆"
else
    git clone https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp

# 应用兼容性补丁
echo "[3/5] 应用兼容性补丁..."

# Patch 1: float8 兼容性 (PyTorch 2.1.0a0 缺少 float8 类型)
python3 -c "
content = open('conversion/base.py').read()
old = 'import torch'
new = '''import torch

# float8 compatibility shim for PyTorch < 2.1
# Jetson Xavier with JetPack 5.x ships PyTorch 2.1.0a0 which lacks float8 types
if not hasattr(torch, \"float8_e4m3fn\"):
    torch.float8_e4m3fn = torch.uint8  # type: ignore
if not hasattr(torch, \"float8_e5m2\"):
    torch.float8_e5m2 = torch.uint8  # type: ignore
'''
content = content.replace(old, new, 1)
open('conversion/base.py', 'w').write(content)
print('Patch 1: float8 兼容性 - 已应用')
"

# Patch 2: removesuffix 兼容性 (Python 3.8 不支持 removesuffix)
python3 -c "
content = open('conversion/step3.py').read()
content = content.replace(
    'name = name.removesuffix(\".gamma\") + \".weight\"',
    'name = name[:-len(\".gamma\")] + \".weight\" if name.endswith(\".gamma\") else name'
)
open('conversion/step3.py', 'w').write(content)
print('Patch 2: removesuffix 兼容性 - 已应用')
"

# 编译
echo "[4/5] 编译 llama.cpp (CUDA)..."
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "[5/5] 验证编译..."
./bin/llama-cli --version

echo ""
echo "=========================================="
echo "llama.cpp 编译完成！"
echo "可执行文件: ~/llama.cpp/build/bin/"
echo "=========================================="
