from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import Sam3Model, Sam3Processor

from .medical_lexicon_adapter import MedicalLexiconAdapter


class OVSAMTextBackbone(nn.Module):
    """
    Wrap:
        SAM3 text_encoder + optional MedicalLexiconAdapter

    Input:
        prompts: List[str]
    Output:
        {
            "input_ids": ...,
            "attention_mask": ...,
            "raw_last_hidden_state": [B, T, 1024],
            "raw_text_embeds": [B, 512],
            "token_feats": [B, T, 1024],
            "phrase_feat": [B, 512],
            "aux": ...
        }
    """

    def __init__(
        self,
        sam3_ckpt: str,
        freeze_text_encoder: bool = True,
        use_medical_lexicon_adapter: bool = True,
        adapter_bottleneck_dim: int = 256,
        adapter_phrase_out_dim: int = 512,
        adapter_dropout: float = 0.1,
        adapter_gate_init: float = 0.1,
        max_length: int = 32,
    ):
        super().__init__()
        self.sam3_ckpt = sam3_ckpt
        self.freeze_text_encoder = freeze_text_encoder
        self.use_medical_lexicon_adapter = bool(use_medical_lexicon_adapter)

        # Load SAM3 once, then only keep text encoder
        sam3_model = Sam3Model.from_pretrained(sam3_ckpt)
        self.text_encoder = sam3_model.text_encoder
        del sam3_model

        encoder_max_length = int(
            getattr(getattr(self.text_encoder, "config", None), "max_position_embeddings", max_length)
        )
        self.max_length = min(int(max_length), encoder_max_length)

        self.processor = Sam3Processor.from_pretrained(sam3_ckpt)

        self.adapter = None
        if self.use_medical_lexicon_adapter:
            self.adapter = MedicalLexiconAdapter(
                token_dim=1024,
                bottleneck_dim=adapter_bottleneck_dim,
                phrase_out_dim=adapter_phrase_out_dim,
                dropout=adapter_dropout,
                gate_init=adapter_gate_init,
            )

        if self.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()

    @property
    def device(self):
        if self.adapter is not None:
            return next(self.adapter.parameters()).device
        return next(self.text_encoder.parameters()).device

    def tokenize(self, prompts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenized = tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
        else:
            tokenized = self.processor(
                text=prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        special_tokens_mask = None
        input_ids = tokenized.get("input_ids", None)
        if tokenizer is not None and input_ids is not None:
            special_masks = []
            for ids in input_ids.tolist():
                special_masks.append(
                    tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
                )
            special_tokens_mask = torch.tensor(special_masks, dtype=torch.long)

        tokenized = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in tokenized.items()
        }
        if special_tokens_mask is not None:
            tokenized["special_tokens_mask"] = special_tokens_mask.to(self.device)
        return tokenized

    def encode_text(
        self,
        prompts: Union[str, List[str]],
        return_raw: bool = True,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenize(prompts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask", None)

        if self.freeze_text_encoder:
            with torch.no_grad():
                text_outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden_state = text_outputs.last_hidden_state  # [B,T,1024]
        raw_text_embeds = text_outputs.text_embeds          # [B,512]

        if self.use_medical_lexicon_adapter:
            if return_aux:
                token_feats, phrase_feat, aux = self.adapter(
                    last_hidden_state=last_hidden_state,
                    attention_mask=attention_mask,
                    return_aux=True,
                )
            else:
                token_feats, phrase_feat = self.adapter(
                    last_hidden_state=last_hidden_state,
                    attention_mask=attention_mask,
                    return_aux=False,
                )
                aux = None
        else:
            # Adapter-off ablation: directly expose the frozen text encoder
            # token features and disable phrase-level query conditioning.
            token_feats = last_hidden_state
            phrase_feat = None
            aux = {"adapter_disabled": True} if return_aux else None

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": tokenized.get("special_tokens_mask", None),
            "offset_mapping": tokenized.get("offset_mapping", None),
            "token_feats": token_feats,
            "phrase_feat": phrase_feat,
        }

        if return_raw:
            out["raw_last_hidden_state"] = last_hidden_state
            out["raw_text_embeds"] = raw_text_embeds

        if return_aux:
            out["aux"] = aux

        return out

    def forward(
        self,
        prompts: Union[str, List[str]],
        return_raw: bool = True,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self.encode_text(prompts, return_raw=return_raw, return_aux=return_aux)
