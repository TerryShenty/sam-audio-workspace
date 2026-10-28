# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import json
import os
import hashlib
from typing import Callable, Dict, Optional, Union

import torch
from huggingface_hub import ModelHubMixin, snapshot_download


class BaseModel(torch.nn.Module, ModelHubMixin):
    config_cls: Callable

    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        cache_dir: str,
        force_download: bool,
        # huggingface_hub 1.x stopped forwarding these two deprecated
        # arguments to ModelHubMixin._from_pretrained.  Keep them optional so
        # both the older SAM Audio environment and current Hub releases work.
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = True,
        revision: Optional[str] = None,
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            cached_model_dir = model_id
        else:
            cached_model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision if revision is not None else cls.revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        with open(os.path.join(cached_model_dir, "config.json")) as fin:
            config = json.load(fin)

        for key, value in model_kwargs.items():
            if key in config:
                config[key] = value

        config = cls.config_cls(**config)
        model = cls(config)
        state_dict = torch.load(
            os.path.join(cached_model_dir, "checkpoint.pt"),
            weights_only=True,
            map_location=map_location,
            mmap=True,
        )
        result = model.load_state_dict(state_dict, strict=strict)
        if result is None:
            raise RuntimeError("load_state_dict returned no result; refusing an unverified model")
        # Verify the actual destination tensors, not merely that a file exists
        # or that fresh and cached features agree with the same broken encoder.
        destination = model.state_dict()
        checked_tensors = checked_numel = 0
        for name, value in state_dict.items():
            if name not in destination:
                continue
            actual = destination[name]
            if not torch.equal(actual, value.to(device=actual.device, dtype=actual.dtype)):
                raise RuntimeError(f"Checkpoint tensor was not loaded correctly: {name}")
            checked_tensors += 1
            checked_numel += actual.numel()
        if not checked_tensors:
            raise RuntimeError("Checkpoint loaded zero tensors")
        def file_digest(name):
            digest = hashlib.sha256()
            with open(os.path.join(cached_model_dir, name), "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        model._pretrained_load_report = {
            "version": 2,
            "verified": True,
            "strict": strict,
            "checkpoint_sha256": file_digest("checkpoint.pt"),
            "config_sha256": file_digest("config.json"),
            "checked_tensors": checked_tensors,
            "checked_numel": checked_numel,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "text_encoder_identity": getattr(getattr(model, "text_encoder", None), "pretrained_identity", {}),
        }
        model._pretrained_model_path = os.path.realpath(cached_model_dir)
        return model
