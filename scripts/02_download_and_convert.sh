#!/bin/bash
# Step3-VL-10B on Jetson Xavier - Step 2: Download models and convert mmproj
# 在 Jetson 上执行此脚本
set -e

echo "=========================================="
echo "Step 2: 下载模型并转换 mmproj"
echo "=========================================="

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 创建模型目录
mkdir -p ~/models/step3-vl-gguf

# === 1. 下载文本模型 (GGUF) ===
echo "[1/4] 下载文本模型 (Q4_K_M, ~4.7GB)..."
if [ -f ~/models/step3-vl-gguf/Step3-VL-10B-Q4_K_M.gguf ]; then
    echo "文本模型已存在，跳过下载"
else
    wget -c -O ~/models/step3-vl-gguf/Step3-VL-10B-Q4_K_M.gguf \
        "https://hf-mirror.com/seanbailey518/Step3-VL-10B-GGUF/resolve/main/Step3-VL-10B-Q4_K_M.gguf"
fi

# === 2. 下载原始 safetensors (用于转换 mmproj) ===
echo "[2/4] 下载原始模型 safetensors (用于转换 mmproj)..."
python3 -c "
from huggingface_hub import snapshot_download
import os

path = snapshot_download(
    repo_id='stepfun-ai/Step3-VL-10B',
    local_dir=os.path.expanduser('~/models/step3-vl-10b-vision'),
    local_dir_use_symlinks=False
)
print(f'下载完成: {path}')
"

# === 3. 转换 mmproj ===
echo "[3/4] 转换 mmproj (F16, ~4GB)..."
cd ~/llama.cpp
python3 convert_hf_to_gguf.py \
    --mmproj \
    --outtype f16 \
    --outfile ~/models/step3-vl-gguf/step3-vl-mmproj.gguf \
    ~/models/step3-vl-10b-vision/

# === 4. 验证 ===
echo "[4/4] 验证文件..."
echo ""
echo "模型文件:"
ls -lh ~/models/step3-vl-gguf/
echo ""

# 清理 safetensors (可选，释放约 19GB)
read -p "是否删除原始 safetensors 文件？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ~/models/step3-vl-10b-vision/
    echo "已删除 safetensors 文件"
fi

echo ""
echo "=========================================="
echo "模型下载和转换完成！"
echo "文本模型: ~/models/step3-vl-gguf/Step3-VL-10B-Q4_K_M.gguf"
echo "视觉投影: ~/models/step3-vl-gguf/step3-vl-mmproj.gguf"
echo "=========================================="
