"""
Minimal Qwen3 implementation for transformers 4.46.3
Based on Qwen2 with Qwen3-specific changes:
- Added q_norm and k_norm (RMSNorm on query and key)
- No bias on q/k/v projections
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2MLP,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import Optional, Tuple, Union
import math


class Qwen3Config(Qwen2Config):
    """Qwen3 configuration - extends Qwen2 with q_norm/k_norm support"""
    model_type = "qwen3"

    def __init__(self, **kwargs):
        # Qwen3 defaults
        kwargs.setdefault("attention_bias", False)
        super().__init__(**kwargs)


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with q_norm/k_norm"""

    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # Qwen3 specific: q_norm and k_norm
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sliding_window = config.sliding_window if config.use_sliding_window else None
        self.is_causal = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Apply q_norm and k_norm (Qwen3 specific)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply rotary position embedding
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for caching
        if past_key_value is not None:
            # Handle DynamicCache from transformers
            if hasattr(past_key_value, 'key_cache') and hasattr(past_key_value, 'value_cache'):
                if len(past_key_value.key_cache) > self.layer_idx:
                    key_states = torch.cat([past_key_value.key_cache[self.layer_idx], key_states], dim=2)
                    value_states = torch.cat([past_key_value.value_cache[self.layer_idx], value_states], dim=2)
                if use_cache:
                    past_key_value.update(key_states, value_states, self.layer_idx)
            elif isinstance(past_key_value, tuple) and len(past_key_value) >= 2:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value_out = (key_states, value_states) if use_cache and not hasattr(past_key_value, 'key_cache') else past_key_value

        # Repeat k/v heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Handle attention mask
        if attention_mask is not None:
            # Resize mask if needed
            if attn_weights.shape != attention_mask.shape:
                # Create causal mask
                causal_mask = torch.full(
                    (attn_weights.shape[2], attn_weights.shape[3]),
                    float("-inf"),
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                )
                causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                attn_weights = attn_weights + causal_mask
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value_out


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen3Model(Qwen2Model):
    """Qwen3Model - extends Qwen2Model with Qwen3 attention layers"""
    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config):
        # Skip Qwen2Model.__init__ and call PreTrainedModel.__init__ directly
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2Model.__init__.__code__  # placeholder

        # Initialize rotary embedding from Qwen2
        # We need to properly initialize it
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    """Qwen3ForCausalLM - uses Qwen3Model"""
    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()


# Export
__all__ = ["Qwen3Config", "Qwen3Model", "Qwen3ForCausalLM"]
