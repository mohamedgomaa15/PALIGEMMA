import torch 
import torch.nn as nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from gemma import GemmaModelForCausalLLM, GemmaConfig, KVCache
from siglip import SiglipVisionConfig, SiglipVisionModel

class PaliGemmaConfig():
    
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            ignore_index=-100,
            image_token_index=256000,
            vocab_size=257152,
            projection_dim=2048,
            hidden_size=2048,
            pad_token_id=None,
            **kwargs
    ):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = text_config.vocab_size
        self.vision_config.projection_dim = projection_dim
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2


class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        return self.linear(image_features)

class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaModelForCausalLLM(config.text_config)
        self.vocab_size = config.vocab_size

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_inputs_ids_with_image_features(
            self,
            image_embeds: torch.Tensor,
            input_embeds: torch.Tensor,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_embeds.shape
        batch_size, seq_length, _ = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        scaled_image_embeds = image_embeds / (self.config.hidden_size**0.5)

        final_embeds = torch.zeros(batch_size, seq_length, embed_dim, dtype=dtype, device=device)

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        text_mask_expand = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expand = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)  
        pad_mask_expand = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embeds = torch.where(text_mask_expand, input_embeds, final_embeds)
        final_embeds =  final_embeds.masked_scatter(image_mask_expand, scaled_image_embeds)
        final_embeds = torch.where(pad_mask_expand, torch.zeros_like(final_embeds), final_embeds)

        ##### Create the ATTENTION MASK #####

        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:

            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device    
            )
        else :
            assert q_len == 1, "When kv_cache is used, the input length should be 1"
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device    
            )

        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1] 
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embeds, causal_mask, position_ids

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_value: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        input_embeds = self.language_model.get_input_embeddings()(input_ids) 
        image_embeds = self.vision_encoder(pixel_value.to(input_embeds.dtype))
        image_embeds = self.multi_modal_projector(image_embeds)

        input_embeds, attention_mask, positional_ids = self._merge_inputs_ids_with_image_features(
            image_embeds, input_embeds, 
            input_ids, attention_mask,
            kv_cache
            )

        outputs = self.language_model(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            positional_ids=positional_ids,
            kv_cache=kv_cache
        )

        return outputs

        