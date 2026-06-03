# Step3-VL-10B Jetson AGX Xavier 部署指南

在 NVIDIA Jetson AGX Xavier (Volta/Sm_70) 上部署 `stepfun-ai/Step3-VL-10B` 多模态大语言模型。

**GGUF 模型下载**: https://huggingface.co/sq6er4/Step3-VL-10B-Q4_K_M

## 环境

| 项目 | 值 |
|------|-----|
| 设备 | Jetson AGX Xavier 32GB, Volta (Sm_70) |
| JetPack | 5.x (R35.6.1) |
| Python | 3.8.10 |
| PyTorch | 2.1.0a0 (Jetson 专用版) |
| Transformers | 4.46.3 |
| CUDA | 11.4 |

## 两条路线对比

| | 路线 A: Safetensors | 路线 B: llama.cpp |
|---|---|---|
| 模型格式 | safetensors FP16 (19GB) | GGUF Q4_K_M (4.7GB) + mmproj (4GB) |
| 显存占用 | ~24GB | ~8GB |
| 加载时间 | ~18 分钟 | ~14 秒 |
| 推理速度 | ~1.7 tokens/s | ~14.8 tokens/s |
| 视觉推理 | ⚠️ 未充分测试 | ✅ 支持 |
| 适用场景 | AGX 64GB / fine-tuning | 通用部署 |

---

## 路线 A: Safetensors 原生部署

```bash


# 1. 下载模型
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('stepfun-ai/Step3-VL-10B')"

# 2. 安装依赖
pip3 install safetensors einops

# 3. 打补丁 (自动)
cd patches/safetensors
bash apply_patches.sh

# 4. 测试推理
python3 deploy.py --test
```

补丁做了什么 → `patches/safetensors/README.md`

---

## 路线 B: llama.cpp GGUF 部署

```bash
# 1. 克隆并编译 llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd patches/gguf && bash apply_patches.sh  # 打补丁 (自动)
cd ~/llama.cpp/build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 2. 下载模型 (文本 + 视觉投影)
mkdir -p ~/models/step3-vl-gguf
wget -c -O ~/models/step3-vl-gguf/Step3-VL-10B-Q4_K_M.gguf \
    "https://huggingface.co/sq6er4/Step3-VL-10B-Q4_K_M/resolve/main/Step3-VL-10B-Q4_K_M.gguf"
wget -c -O ~/models/step3-vl-gguf/mmproj-Step3-VL-10B-f16.gguf \
    "https://huggingface.co/sq6er4/Step3-VL-10B-Q4_K_M/resolve/main/mmproj-Step3-VL-10B-f16.gguf"

# 4. 启动服务器
~/llama.cpp/build/bin/llama-server \
    -m ~/models/step3-vl-gguf/Step3-VL-10B-Q4_K_M.gguf \
    --mmproj ~/models/step3-vl-gguf/mmproj-Step3-VL-10B-f16.gguf \
    --host 0.0.0.0 --port 8080 -ngl 99 -t 4

# 5. 测试推理
# 文本
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"step3-vl","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":50}'

# 视觉 (base64)
IMG_B64=$(base64 -i image.png | tr -d '\n')
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"step3-vl\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG_B64\"}},{\"type\":\"text\",\"text\":\"这是什么\"}]}],\"max_tokens\":200}"
```

补丁做了什么 → `patches/gguf/README.md`

---

## 模型结构

```
Step3VL10BForCausalLM
  └── StepRoboticsModel (model.model)
        ├── Qwen3Model (model.model.model) ← 语言模型 (36层, 4096维)
        ├── StepRoboticsVisionEncoder (model.model.vision_model) ← 视觉编码器 (47层, 1536维)
        └── vit_large_projector (model.model.vit_large_projector) ← 视觉投影器
```

## 常见问题

### Q: mmproj 加载失败 "unable to find tensor mm.1.weight"

预构建的 mmproj 不兼容当前 llama.cpp。必须用 `convert_hf_to_gguf.py --mmproj` 从原始 safetensors 转换。

### Q: 输出在 reasoning_content 而非 content

模型默认启用 thinking 模式，思考过程在 `reasoning_content` 字段。这是正常行为。

### Q: 图片推理报错 "HTTPS is not supported"

llama-server 编译时未启用 SSL。使用 HTTP 图片 URL 或 base64 编码内嵌图片。

### Q: PyTorch 方案输出乱码

权重映射路径错误。确保 `model.layers.X` 映射到 `model.model.layers.X`。

## 文件结构

```
step3-vl-jetson-patches/
├── README.md                              # 本文档
├── patches/
│   ├── safetensors/                       # 路线 A
│   │   ├── README.md                      # 补丁说明
│   │   ├── apply_patches.sh               # 自动打补丁
│   │   ├── deploy.py                      # 推理脚本
│   │   └── files/                         # 源文件
│   └── gguf/                              # 路线 B
│       ├── README.md                      # 补丁说明
│       └── apply_patches.sh               # 自动打补丁
└── scripts/                               # 路线 B 一键脚本
    ├── 01_build_llamacpp.sh
    ├── 02_download_and_convert.sh
    ├── 03_deploy_server.sh
    └── 04_test_inference.sh
```

## 许可证

Apache 2.0（与原始模型一致）。
