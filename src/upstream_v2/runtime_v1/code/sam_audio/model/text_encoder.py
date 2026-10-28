# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from typing import Tuple
import hashlib
import json

import torch
import transformers

from sam_audio.model.config import T5EncoderConfig


class T5TextEncoder(torch.nn.Module):
    def __init__(self, cfg: T5EncoderConfig):
        super().__init__()
        self.model = transformers.T5EncoderModel.from_pretrained(cfg.name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.name)
        # T5 is loaded separately and is absent from the SAM checkpoint. Its
        # actual weights/tokenizer must participate in cache and delta identity.
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        tokenizer = getattr(self.tokenizer, "backend_tokenizer", None)
        tokenizer_state = tokenizer.to_str() if tokenizer is not None else json.dumps(
            {"vocab": self.tokenizer.get_vocab(), "special_tokens": self.tokenizer.special_tokens_map},
            sort_keys=True,
        )
        self.pretrained_identity = {
            "weights_sha256": digest.hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_state.encode()).hexdigest(),
        }
        self.pad_mode = cfg.pad_mode
        self.max_length = cfg.max_length

    def forward(self, texts: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=self.pad_mode,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )["last_hidden_state"]

        return res, attention_mask.bool()
