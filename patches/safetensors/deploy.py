#!/usr/bin/env python3
"""
Step3-VL-10B PyTorch/transformers 推理脚本
在 Jetson Xavier 上运行
"""

import os
import sys
import time
import types
import argparse
import torch
from safetensors.torch import safe_open

# 设置 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_model(model_path=None, device="cuda"):
    """加载 Step3-VL-10B 模型"""
    from transformers import AutoConfig

    if model_path is None:
        model_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--stepfun-ai--Step3-VL-10B/snapshots/5026053b0c2f5dfaa08fc2d149384162c3c8bca1"
        )

    print(f"[1/5] 加载配置: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    print("[2/5] 创建模型...")
    t0 = time.time()

    # 动态加载 Step3VL 模型代码
    sys.path.insert(0, model_path)
    import importlib

    # 处理相对导入
    spec = importlib.util.spec_from_file_location(
        "step3vl_model", os.path.join(model_path, "modeling_step_vl.py")
    )
    module = importlib.util.module_from_spec(spec)

    # 修补导入
    source = open(os.path.join(model_path, "modeling_step_vl.py")).read()
    source = source.replace("from .configuration_step_vl import", "from configuration_step_vl import")
    source = source.replace("from .vision_encoder import", "from vision_encoder import")
    source = source.replace(
        "from transformers import Qwen3Model",
        "from transformers.models.qwen3 import Qwen3Model"
    )

    # 创建模块
    code = compile(source, os.path.join(model_path, "modeling_step_vl.py"), "exec")
    exec(code, module.__dict__)

    Step3VL10BForCausalLM = module.Step3VL10BForCausalLM

    # 创建模型实例
    model = Step3VL10BForCausalLM._from_config(config, torch_dtype=torch.float16)
    print(f"  模型创建耗时: {time.time() - t0:.1f} 秒")

    print("[3/5] 加载权重...")
    t0 = time.time()

    # 查找 safetensors 文件
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    import json
    with open(index_file) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    safetensors_files = set(weight_map.values())

    # 逐分片加载权重
    loaded = 0
    skipped = 0
    for fname in sorted(safetensors_files):
        fpath = os.path.join(model_path, fname)
        if not os.path.exists(fpath):
            print(f"  跳过缺失文件: {fname}")
            continue

        with safe_open(fpath, framework="pt") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key).to(torch.float16)

                # 映射 key: model.layers.X → model.model.layers.X
                if key.startswith("model."):
                    new_key = "model.model." + key[len("model."):]
                elif key.startswith("vision_model."):
                    new_key = "model.model." + key
                elif key.startswith("vit_large_projector."):
                    new_key = "model.model." + key
                elif key == "lm_head.weight":
                    new_key = key
                else:
                    skipped += 1
                    continue

                # 设置权重
                try:
                    param = model
                    for part in new_key.split(".")[:-1]:
                        param = getattr(param, part)
                    param_name = new_key.split(".")[-1]
                    param_dict = dict(param.named_parameters())
                    if param_name in param_dict:
                        param_dict[param_name].data.copy_(tensor)
                        loaded += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1

    print(f"  加载权重: {loaded} 个, 跳过: {skipped} 个")
    print(f"  权重加载耗时: {time.time() - t0:.1f} 秒")

    print("[4/5] 移到 GPU...")
    t0 = time.time()
    model = model.to(device)
    print(f"  GPU 移动耗时: {time.time() - t0:.1f} 秒")

    print("[5/5] 加载 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer


def inference(model, tokenizer, prompt, max_tokens=100, temperature=0.7):
    """推理"""
    print(f"\n输入: {prompt}")
    print("-" * 40)

    t0 = time.time()

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 手动自回归推理（避免 model.generate 的 torch.distributed 问题）
    language_model = model.model.model  # 直接调用 Qwen3Model

    generated = input_ids
    mask = attention_mask
    tokens = []

    for i in range(max_tokens):
        with torch.no_grad():
            outputs = language_model(input_ids=generated, attention_mask=mask)
            logits = model.lm_head(outputs.last_hidden_state)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        token_id = next_token.item()
        tokens.append(token_id)

        # 检查 EOS
        if token_id == tokenizer.eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=-1)
        mask = torch.cat([mask, torch.ones_like(next_token)], dim=-1)

    output_text = tokenizer.decode(tokens, skip_special_tokens=True)
    elapsed = time.time() - t0
    speed = len(tokens) / elapsed

    print(f"输出: {output_text}")
    print(f"\n耗时: {elapsed:.1f} 秒, 速度: {speed:.1f} tokens/s")

    return output_text


def main():
    parser = argparse.ArgumentParser(description="Step3-VL-10B PyTorch 推理")
    parser.add_argument("--test", action="store_true", help="运行测试推理")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="推理提示词")
    parser.add_argument("--max-tokens", type=int, default=100, help="最大生成 token 数")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()

    print("=" * 50)
    print("Step3-VL-10B PyTorch/transformers 推理")
    print("=" * 50)

    # 系统信息
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    model, tokenizer = load_model(device=args.device)

    if args.test:
        inference(model, tokenizer, "Hello, how are you?", max_tokens=50)
        print()
        inference(model, tokenizer, "1+1等于几？", max_tokens=30)

    # 交互模式
    print("\n进入交互模式 (输入 'quit' 退出)")
    while True:
        try:
            prompt = input("\n> ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if prompt:
                inference(model, tokenizer, prompt, max_tokens=args.max_tokens)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
