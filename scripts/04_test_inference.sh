#!/bin/bash
# Step3-VL-10B on Jetson Xavier - Step 4: Test inference with timing
# 在 Jetson 上执行此脚本
# 用法: bash 04_test_inference.sh [图片路径] [提示词]

IMAGE_PATH="${1:-}"
PROMPT="${2:-这是什么}"

echo "=========================================="
echo "Step3-VL-10B 视觉推理测试"
echo "=========================================="

# 检查 server 是否运行
if ! wget -q -O - http://localhost:8080/health 2>/dev/null | grep -q "ok"; then
    echo "错误: llama-server 未运行，请先执行 03_deploy_server.sh"
    exit 1
fi

# 总开始时间
TOTAL_START=$(date +%s%N)

# === 图片编码 ===
if [ -n "$IMAGE_PATH" ]; then
    echo "[1/2] 编码图片: $IMAGE_PATH"
    ENCODE_START=$(date +%s%N)
    IMG_B64=$(base64 -i "$IMAGE_PATH" | tr -d '\n')
    ENCODE_END=$(date +%s%N)
    ENCODE_MS=$(( (ENCODE_END - ENCODE_START) / 1000000 ))
    echo "[1/2] 图片编码完成: ${ENCODE_MS} ms (base64 长度: ${#IMG_B64})"

    # 构造带图片的请求
    REQUEST="{\"model\":\"step3-vl\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG_B64\"}},{\"type\":\"text\",\"text\":\"$PROMPT\"}]}],\"max_tokens\":200,\"temperature\":0.7}"
else
    echo "[1/2] 纯文本推理模式"
    REQUEST="{\"model\":\"step3-vl\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":200,\"temperature\":0.7}"
fi

# === 推理 ===
echo "[2/2] 发送推理请求..."
INFER_START=$(date +%s%N)

RESPONSE=$(wget -q -O - --header='Content-Type: application/json' \
    --post-data="$REQUEST" \
    http://localhost:8080/v1/chat/completions 2>&1)

INFER_END=$(date +%s%N)
INFER_MS=$(( (INFER_END - INFER_START) / 1000000 ))
echo "[2/2] 推理完成: ${INFER_MS} ms"

# 总时间
TOTAL_END=$(date +%s%N)
TOTAL_MS=$(( (TOTAL_END - TOTAL_START) / 1000000 ))

# === 输出结果 ===
echo ""
echo "=========================================="
echo "推理结果"
echo "=========================================="

# 提取 reasoning_content 和 content
REASONING=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('reasoning_content',''))" 2>/dev/null)
CONTENT=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('content',''))" 2>/dev/null)

if [ -n "$REASONING" ]; then
    echo "[思考过程]"
    echo "$REASONING"
    echo ""
fi
if [ -n "$CONTENT" ]; then
    echo "[回答]"
    echo "$CONTENT"
    echo ""
fi

# 提取 timing
echo "=========================================="
echo "时间统计"
echo "=========================================="
echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
t = d.get('timings', {})
u = d.get('usage', {})
print(f'输入 tokens:      {u.get(\"prompt_tokens\", \"N/A\")}')
print(f'输出 tokens:      {u.get(\"completion_tokens\", \"N/A\")}')
print(f'Prompt 处理速度: {t.get(\"prompt_per_second\", 0):.1f} tokens/s')
print(f'生成速度:        {t.get(\"predicted_per_second\", 0):.1f} tokens/s')
print(f'图片编码:        ${ENCODE_MS:-0} ms')
print(f'推理耗时:        $INFER_MS ms')
print(f'总耗时:          $TOTAL_MS ms')
" 2>/dev/null

echo "=========================================="
