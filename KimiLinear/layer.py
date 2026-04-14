import torch
import torch.nn as nn

from .configuration import KimiLinearConfig
from transformers.processing_utils import Unpack
from .attention import KimiDeltaAttention, KimiMLAAttention
from .mlp import KimiMLP
from .norm import KimiRMSNorm
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from .moe import KimiSparseMoeBlock


class KimiDecoderLayer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        
        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            if config.attn_type == "cyclekda":
                self.self_attn = KimiDeltaAttention(config=config, layer_idx=layer_idx)
            elif config.attn_type == "kda":
                assert config.linear_attn_config["T_cycle"] == 1, "KimiLinearTimeModel only supports T_cycle=1 for baseline kda."
                self.self_attn = KimiDeltaAttention(config=config, layer_idx=layer_idx)
            elif config.attn_type == "rwkv7":
                from .attn.rwkv7_attn import RWKV7Attention
                self.self_attn = RWKV7Attention(config=config, layer_idx=layer_idx)
            else:
                raise ValueError(f"Unsupported attn_type: {config.attn_type}")
        elif config.is_mla:
            self.is_linear_attn = False
            self.self_attn = KimiMLAAttention(
                config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError
        
        # if (
        #     config.num_experts is not None
        #     and layer_idx >= config.first_k_dense_replace
        #     and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        # ):
        #     self.block_sparse_moe = KimiSparseMoeBlock(config)
        # else:
        self.mlp = KimiMLP(config)
        
        self.input_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states