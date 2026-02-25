import torch
import torch.nn as nn
from .configuration import KimiLinearConfig
from transformers.modeling_utils import PreTrainedModel
from .moe import KimiBlockSparseMLP
from .layer import KimiDecoderLayer
from .norm import KimiRMSNorm
from .attention import KimiMLAAttention
from .cache import KimiDynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache
from transformers.utils import logging, auto_docstring, TransformersKwargs, can_return_tuple
from transformers.utils.generic import OutputRecorder, check_model_inputs
from transformers.processing_utils import Unpack
from transformers.masking_utils import create_causal_mask

logger = logging.get_logger(__name__)


class KimiPreTrainedModel(PreTrainedModel):
    config_class = KimiLinearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KimiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(KimiBlockSparseMLP, index=1),
        "hidden_states": KimiDecoderLayer,
        "attentions": KimiMLAAttention,
    }
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
                
class KimiLinearModel(KimiPreTrainedModel):
    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([KimiDecoderLayer(
            config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        if getattr(config, "_attn_implementation", None) is not None:
            if config._attn_implementation != "flash_attention_2":
                logger.warning_once(
                    f"Ignoring the provided attention implementation {config._attn_implementation}")
                logger.warning_once("Using flash_attention_2 backend instead.")
                config._attn_implementation = "flash_attention_2"
        else:
            config._attn_implementation = "flash_attention_2"

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        # Get inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = KimiDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(
            attention_mask, cache_position)

        hidden_states = inputs_embeds
        if past_key_values is not None:
            assert isinstance(past_key_values, KimiDynamicCache)

        for decoder_layer in self.layers:
            layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        

class KimiLinearTimeModel(KimiPreTrainedModel):
    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        
        self.embed_tokens = nn.Linear(config.input_size, config.hidden_size)
        self.layers = nn.ModuleList([KimiDecoderLayer(
            config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.output_proj = nn.Linear(config.hidden_size, config.input_size)

        if getattr(config, "_attn_implementation", None) is not None:
            if config._attn_implementation != "flash_attention_2":
                logger.warning_once(
                    f"Ignoring the provided attention implementation {config._attn_implementation}")
                logger.warning_once("Using flash_attention_2 backend instead.")
                config._attn_implementation = "flash_attention_2"
        else:
            config._attn_implementation = "flash_attention_2"

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        # Get inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = KimiDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(
            attention_mask, cache_position)

        hidden_states = inputs_embeds
        if past_key_values is not None:
            assert isinstance(past_key_values, KimiDynamicCache)

        for decoder_layer in self.layers:
            layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)

        return hidden_states, past_key_values


class KimiLinearForCausalLM(KimiPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = KimiLinearModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        generation_mode: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, KimiLinearForCausalLM

        >>> model = KimiLinearForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

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

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = outputs[0]
        if generation_mode:
            logits = logits[:, -1:]
        logits = self.lm_head(logits)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )