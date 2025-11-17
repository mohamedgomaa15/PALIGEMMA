import torch
import math
import torch.nn as nn
from typing import Tuple, Optional, List

class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].size(-2)
    
    def update(
            self,
            key_states: torch.tensor,
            value_states: torch.tensor,
            layer_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_index:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_index] = torch.cat([self.key_cache[layer_index], key_states], dim=-2)
            self.value_cache[layer_index] = torch.cat([self.value_cache[layer_index], value_states], dim=-2)

        return self.key_cache[layer_index], self.value_cache[layer_index]

class GemmaRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=100000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_lens=None):
        #inv_freq -->  [head_dim / 2]
        #position_ids --> [B, seq_len] or [seq_len]
        self.inv_freq.to(x.device)
        # [head_dim / 2]  -> [B, head_dim / 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # [B, seq_len]  -> [B, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.dtype
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
           # [B, seq_len, head_dim/2]
            freq = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  
            # [B, seq_len, head_dim]
            emb = torch.cat([freq, freq], dim=-1)
            cos_emb = emb.cos().to(x.dtype)
            sin_emb = emb.sin().to(x.dtype)

        return cos_emb, sin_emb

def apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1):
    # [B, seq_len, head_dim] --> [B, 1, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query_states_rotated = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states_rotated = (key_states * cos) + (rotate_half(key_states) * sin)
    return query_states_rotated, key_states_rotated

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)

class GemmaConfig():

    def __init__(
            self, 
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=100000,
            attention_bias=False,
            pad_token_id=None,
            attention_dropout=0.0,
            **kwargs
    ):
        super().__init__(**kwargs)  
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class GemmaRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()) * (self.weight.float() + 1.0).type_as(x)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_heads, seq_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_heads, n_rep, seq_length, head_dim)
    return hidden_states.reshape(batch_size, num_heads * n_rep, seq_length, head_dim)
    

class GemmaAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_index: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0, "Hidden size must be divisible by number of heads"

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self, 
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        positional_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, q_length, _ = hidden_states.size()
        # [B, seq_len, num_heads * head_dim]
        query_states = self.q_proj(hidden_states)
        # [B, seq_len, num_key_value_heads * head_dim]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # [B, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, q_length, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_key_value_heads, seq_len, head_dim]
        key_states = key_states.view(batch_size, q_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, positional_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
         
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_index)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)  

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_length, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

         

class GemmaMLP(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(nn.functional.gelu(self.gate_proj(hidden_states), approximate="tanh") * self.up_proj(hidden_states))

class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, index_layer: int):
        super().__init__()
        self.config = config
        self.rms_norm1 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm2 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)
        self.attn = GemmaAttention(config, index_layer)

    def forward(
           self, 
            input_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            positional_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None):
        resudual = input_embeds
        hidden_states  = self.rms_norm1(input_embeds)
        hidden_states, _ = self.attn(hidden_states, attention_mask, positional_ids, kv_cache)
        hidden_states = resudual + hidden_states
        resudual = hidden_states
        hidden_states = self.rms_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return resudual + hidden_states

class GemmaDecoder(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
           [GemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.rms_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
           self, 
            input_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            positional_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None):
        hidden_states = input_embeds
        for layer in self.decoder_layers:
           hidden_states = layer(hidden_states, attention_mask, positional_ids, kv_cache)

        hidden_states = self.rms_norm(hidden_states)
        return hidden_states

class GemmaModel(nn.Module):
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.decoder = GemmaDecoder(config)

    def forward(
            self, 
            input_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            positional_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None) -> torch.FloatTensor:
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=input_embeds.dtype)
        hidden_states = input_embeds * normalizer
        
        hidden_states = self.decoder(hidden_states, attention_mask, positional_ids, kv_cache)
       
        return hidden_states

class GemmaModelForCausalLLM(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.gemma_decoder = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

    def tie_weight(self):
        self.lm_head.weight = self.gemma_decoder.embeddings.weight

    def get_input_embeddings(self):
        return self.gemma_decoder.embeddings

    def forward(
            self, 
            input_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            positional_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None) -> Tuple:
        hidden_states = self.gemma_decoder(input_embeds, attention_mask, positional_ids, kv_cache)
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data
    

