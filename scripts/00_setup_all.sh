#!/bin/bash
# Step3-VL-10B on Jetson Xavier - 一键安装脚本
# 在 Jetson 上执行此脚本，自动完成所有步骤
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Step3-VL-10B Jetson Xavier 一键部署"
echo "=========================================="
echo ""
echo "此脚本将执行以下步骤:"
echo "  1. 编译 llama.cpp (CUDA)"
echo "  2. 下载模型文件"
echo "  3. 转换 mmproj 视觉投影"
echo ""
echo "预计需要:"
echo "  - 磁盘空间: ~30GB"
echo "  - 时间: 1-2 小时 (取决于网速)"
echo ""
read -p "是否继续？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 执行各步骤
echo ""
echo "========== Step 1: 编译 llama.cpp =========="
bash "$SCRIPT_DIR/01_build_llamacpp.sh"

echo ""
echo "========== Step 2: 下载模型并转换 =========="
bash "$SCRIPT_DIR/02_download_and_convert.sh"

echo ""
echo "=========================================="
echo "安装完成！"
echo ""
echo "启动推理服务器:"
echo "  bash $SCRIPT_DIR/03_deploy_server.sh"
echo ""
echo "测试推理:"
echo "  bash $SCRIPT_DIR/04_test_inference.sh /path/to/image.png \"描述这张图片\""
echo "  bash $SCRIPT_DIR/04_test_inference.sh \"\" \"1+1等于几\"  # 纯文本"
echo "=========================================="
