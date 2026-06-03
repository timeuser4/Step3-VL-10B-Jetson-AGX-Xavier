# Copyright 2025 The STEPFUN and HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List, Dict, Any
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3 import Qwen3Model
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
# Unpack not available in this transformers version
from transformers.utils import logging

from typing import Any, Literal, Optional, TypedDict, Union

from .configuration_step_vl import StepRoboticsConfig
from .vision_encoder import StepRoboticsVisionEncoder
logger = logging.get_logger(__name__)

class StepVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    patch_pixel_values: Optional[torch.Tensor]
    num_patches: List[int]


class StepVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor


StepVLImageInputs = Union[StepVLImagePixelInputs,
                           StepVLImageEmbeddingInputs]


@dataclass
class StepVLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

def _flatten_embeddings(embeddings) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))

def _embedding_count_expression(embeddings) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)

def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    is_multimodal: torch.Tensor,
    multimodal_embeddings,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
    Note:
        This updates ``inputs_embeds`` in place.
    """
    num_expected_tokens = is_multimodal.sum().item()
    assert isinstance(num_expected_tokens, int)

    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders")

    is_multimodal = is_multimodal.to(inputs_embeds.device)
    flattened = flattened.to(inputs_embeds.device)
    inputs_embeds[is_multimodal] = flattened
    return inputs_embeds

def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings,
    placeholder_token_id: Union[int, List[int]],
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
    
    ``placeholder_token_id`` can be a list of token ids (e.g, token ids 
    of img_start, img_break, and img_end tokens) when needed: This means 
    the order of these tokens in the ``input_ids`` MUST MATCH the order of 
    their embeddings in ``multimodal_embeddings`` since we need to 
    slice-merge instead of individually scattering.
    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.
    
    Then the image embeddings (that correspond to I's) from vision encoder 
    must be padded with embeddings of S, B, and E in the same order of 
    input_ids for a correct embedding merge.
    Note:
        This updates ``inputs_embeds`` in place.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = torch.tensor(placeholder_token_id,
                                            device=input_ids.device)
        return _merge_multimodal_embeddings(
            inputs_embeds,
            torch.isin(input_ids, placeholder_token_id),
            multimodal_embeddings,
        )

    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )

class StepRoboticsPreTrainedModel(PreTrainedModel):
    # Link this model family to its configuration class so PreTrainedModel.from_pretrained
    # can load the config instead of failing with a NoneType error.
    config_class = StepRoboticsConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_static_cache = True
    _supports_attention_backend = True


class StepRoboticsModel(StepRoboticsPreTrainedModel, GenerationMixin):
    config: StepRoboticsConfig
    base_model_prefix = ""
    def __init__(self, config: StepRoboticsConfig):
        super().__init__(config)
        self._weight_map = {
            "model.layers.": "model.language_model.layers.",
            "model.embed_tokens.": "model.language_model.embed_tokens.",
            "model.norm.": "model.language_model.norm.",
            "vision_model.": "model.vision_model.",
            "vit_large_projector.": "model.vit_large_projector.",
        }
        self.vision_model = StepRoboticsVisionEncoder(config.vision_config)
        self.model = Qwen3Model(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.vit_large_projector = nn.Linear(
                config.vision_config.width * 4,
                config.text_config.hidden_size,                
                bias=config.projector_bias) 
        self.image_placeholder_token_id = config.image_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings  = None,
    ) -> torch.Tensor:
        input_ids = input_ids.squeeze(0)
        if multimodal_embeddings is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            is_text = input_ids != self.config.image_token_id
            text_ids = input_ids[is_text]
            text_embeds = self.model.embed_tokens(text_ids)
                     
            inputs_embeds = torch.empty(input_ids.shape[0],
                                        text_embeds.shape[-1],
                                        dtype=text_embeds.dtype,
                                        device=text_embeds.device)
            inputs_embeds[is_text] = text_embeds
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_id)
        inputs_embeds = inputs_embeds.unsqueeze(0)
        return inputs_embeds
       

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[StepVLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        patch_pixel_values = kwargs.pop("patch_pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            # pixel_values = flatten_bn(pixel_values, concat=True)
            if pixel_values.dim() >= 3:
                pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:])
            if patch_pixel_values is not None:
                # patch_pixel_values = flatten_bn(patch_pixel_values,
                #                                 concat=True)
                patch_pixel_values = patch_pixel_values.view(
                    -1, *patch_pixel_values.shape[-3:])
                # Handle empty patch_pixel_values by setting to None
                if patch_pixel_values.shape[0] == 0:
                    patch_pixel_values = None

            return StepVLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values.to(self.dtype).to(self.device),
                patch_pixel_values=patch_pixel_values.to(self.dtype).to(
                    self.device) if patch_pixel_values is not None else None,
                num_patches=num_patches,
            )

        if image_embeds is not None:
            if image_embeds.dim() == 2 or image_embeds.dim() >= 3:
                image_embeds = image_embeds.view(-1, image_embeds.shape[-1])
            else:
                raise ValueError(
                    f"Unexpected shape for image_embeds: {image_embeds.shape}")

            return StepVLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds.to(self.dtype).to(self.device),
            )
        return None
    
    def _process_image_features(self,
                                image_features: torch.Tensor) -> torch.Tensor:
        B, P = image_features.shape[:2]
        HW = int(P ** 0.5)
        image_features = image_features.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vision_model.vit_downsampler1(image_features)
        image_features = self.vision_model.vit_downsampler2(image_features)

        B, C, HW, HW = image_features.shape
        image_features = image_features.view(B, -1, HW * HW).permute(0, 2, 1)
        image_features = self.vit_large_projector(image_features)
        return image_features

    def _get_vision_model_output(self,
                                 input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vision_model(input_tensor)

    def _process_image_input(
            self, image_input: StepVLImageInputs) -> Tuple[torch.Tensor, ...]:

        if image_input["type"] == "image_embeds":
            image_features = image_input["image_embeds"]
        else:
            image_features = self._get_vision_model_output(
                image_input["pixel_values"])
            patch_image_features = self._get_vision_model_output(
                image_input["patch_pixel_values"]
            ) if image_input["patch_pixel_values"] is not None else None
            num_patches = image_input["num_patches"]

        image_features = self._process_image_features(image_features)
        patch_image_features = self._process_image_features(
            patch_image_features) if patch_image_features is not None else None

        merged_image_features = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[
                    cur_patch_idx:cur_patch_idx + num_patch]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(image_features[i].view(
                -1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged_image_features.append(
                torch.cat(cur_feature) if len(cur_feature) >
                1 else cur_feature[0])
    
        return merged_image_features
    
    def get_multimodal_embeddings(self, **kwargs):
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        images: Optional[List[Image.Image]] = None,
    ) -> Union[tuple, StepVLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Example:
        ```python
        >>> from transformers import AutoTokenizer, Llama4ForCausalLM
        >>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            input_ids = input_ids
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = StepVLCausalLMOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
            
        )
        return output if return_dict else output.to_tuple()



class Step3VL10BForCausalLM(StepRoboticsPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        "^vit_large_projector": "model.vit_large_projector"
        }
    _tied_weights_keys = ["lm_head.weight"]
    config: StepRoboticsConfig

    def __init__(self, config: StepRoboticsConfig):
        super().__init__(config)
        self._weight_map = {
            "model.layers.": "model.language_model.layers.",
            "model.embed_tokens.": "model.language_model.embed_tokens.",
            "model.norm.": "model.language_model.norm.",
            "vision_model.": "model.vision_model.",
            "vit_large_projector.": "model.vit_large_projector.",
        }
        self.model = StepRoboticsModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()
    
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        num_patches = None,
        patch_pixel_values = None,
        patch_newline_mask = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, StepVLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration
        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            num_patches = num_patches,
            patch_pixel_values = patch_pixel_values,
            patch_newline_mask=patch_newline_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        los = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return StepVLCausalLMOutputWithPast(
            logits=logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs
    
    def _fix_state_dict_key_on_load(self, key: str) -> Tuple[str, bool]:
        if key.startswith("language_model."):
            return key[len("language_model."):], True
        
        return key, False

