#!/bin/bash
# Step3-VL-10B on Jetson Xavier - Step 3: Deploy llama-server
# 在 Jetson 上执行此脚本

echo "=========================================="
echo "Step 3: 启动 llama-server"
echo "=========================================="

# 杀掉旧进程
pkill -f llama-server 2>/dev/null
sleep 1

# 启动 server
~/llama.cpp/build/bin/llama-server \
    -m ~/models/step3-vl-gguf/Step3-VL-10B-Q4_K_M.gguf \
    --mmproj ~/models/step3-vl-gguf/step3-vl-mmproj.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    -t 4

echo "Server 已启动: http://0.0.0.0:8080"
echo "API 端点: http://<jetson-ip>:8080/v1/chat/completions"
