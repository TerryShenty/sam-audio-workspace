"""Trainable video-language AUDSEG bridge following LISA-style supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

IGNORE_INDEX = -100


@dataclass
class AudsegEncoderOutput:
    """Contextual target representation and its language-model supervision."""

    hidden_state: torch.Tensor
    ce_loss: torch.Tensor
    token_accuracy: torch.Tensor


def select_audseg_h_before(
    last_hidden_state: torch.Tensor,
    labels: torch.Tensor,
    audseg_token_id: int,
) -> torch.Tensor:
    """Select the final-layer state that predicts ``<AUDSEG>``.

    Causal language-model logits at position ``i`` predict label ``i + 1``.
    LISA, MAOAM-GLaMM, and SimToken therefore shift the special-token mask by
    one position before indexing the hidden states.  This helper implements the
    same operation without hard-coded image/video-token offsets.
    """

    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state must have shape [B, L, D], got "
            f"{last_hidden_state.shape}"
        )
    if labels.ndim != 2 or labels.shape != last_hidden_state.shape[:2]:
        raise ValueError(
            "labels must have shape [B, L] matching last_hidden_state; got "
            f"labels={labels.shape}, hidden={last_hidden_state.shape}"
        )

    predicts_audseg = labels[:, 1:].eq(audseg_token_id)
    counts = predicts_audseg.sum(dim=1)
    if not torch.all(counts == 1):
        raise RuntimeError(
            "Expected exactly one supervised <AUDSEG> token per sample; found "
            f"counts={counts.detach().cpu().tolist()}"
        )
    selected = last_hidden_state[:, :-1][predicts_audseg]
    return selected.reshape(last_hidden_state.size(0), last_hidden_state.size(-1))


def build_audseg_response(target_description: str | None = None) -> str:
    """Keep semantic GT strictly AFTER the audio-conditioning state.

    The causal h_before used for separation cannot attend to this suffix.
    A missing description retains marker-only supervision; never guess labels.
    """
    response = "The target sound is <AUDSEG>."
    if target_description is not None:
        description = target_description.strip()
        if not description:
            raise ValueError("target_description must be nonempty when supplied")
        if "<" in description or ">" in description:
            raise ValueError("target_description must be plain text, without special tokens")
        response += " Target identity: " + description
    return response


class QwenAudsegEncoder(nn.Module):
    """Train Qwen-VL to emit one contextual target-sound marker.

    The assistant target starts with ``The target sound is <AUDSEG>.``.
    Optional verified target descriptions follow the marker as auxiliary CE labels.  Standard
    causal CE trains the short response, while downstream audio losses use the
    last-layer state immediately before ``<AUDSEG>``.  Only language LoRA plus
    ``lm_head``/``embed_tokens`` are trainable; the original VLM weights and
    visual encoder remain frozen.
    """

    token = "<AUDSEG>"
    response = "The target sound is <AUDSEG>."

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        fps: float = 1.0,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        attn_implementation: str | None = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
        gradient_checkpointing: bool = True,
        backend: str = "qwen2_5_vl",
    ):
        super().__init__()
        if lora_rank <= 0:
            raise ValueError("lora_rank must be positive for AUDSEG fine-tuning")
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Trainable Qwen2.5-VL support requires transformers>=4.54.0 and peft"
            ) from exc

        if backend not in {"qwen2_5_vl", "qwen3_omni"}:
            raise ValueError(f"Unsupported VLM backend: {backend}")
        self.backend = backend
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

        model_kwargs = {"torch_dtype": dtype}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if backend == "qwen3_omni":
            from transformers import Qwen3OmniMoeForConditionalGeneration

            omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_name, **model_kwargs
            )
            # Train the Thinker directly: its hidden states/logits and embeddings
            # are the language bridge. No speech generation/Talker is needed.
            base_model = omni.thinker
            del omni
        else:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, **model_kwargs
            )
        added = self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.token]}
        )
        if added:
            base_model.resize_token_embeddings(len(self.processor.tokenizer))
        self.audseg_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.token
        )
        tokenized_marker = self.processor.tokenizer(
            self.token, add_special_tokens=False
        )["input_ids"]
        if tokenized_marker != [self.audseg_token_id]:
            raise RuntimeError(
                f"{self.token} must tokenize to exactly one ID, got {tokenized_marker}"
            )
        if added:
            self._initialize_audseg_embedding(base_model)

        # Freeze first, then let PEFT re-enable only language LoRA and the two
        # modules required to learn/generate the newly added vocabulary item.
        base_model.requires_grad_(False)
        exact_lora_targets = self._language_lora_targets(
            base_model, lora_target_modules
        )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=exact_lora_targets,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"],
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.config.use_cache = False
        trainable_visual = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and any(tag in name.lower() for tag in ("visual", "vision", "audio_tower"))
        ]
        if trainable_visual:
            raise RuntimeError(
                "Language-only LoRA unexpectedly enabled visual parameters: "
                f"{trainable_visual[:5]}"
            )
        if gradient_checkpointing:
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()

        self.fps = fps
        self._device = torch.device(device)
        self.model.to(self._device)

        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.hidden_size = int(text_config.hidden_size)

    def _initialize_audseg_embedding(self, model: nn.Module) -> None:
        """Initialize the new row from semantically related existing tokens."""

        source_ids = self.processor.tokenizer(
            "audio target", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        with torch.no_grad():
            input_embeddings = model.get_input_embeddings().weight
            source_ids = source_ids.to(input_embeddings.device)
            initialized = input_embeddings[source_ids].mean(dim=0)
            input_embeddings[self.audseg_token_id].copy_(initialized)
            output_layer = model.get_output_embeddings()
            if (
                output_layer is not None
                and output_layer.weight.data_ptr() != input_embeddings.data_ptr()
            ):
                output_layer.weight[self.audseg_token_id].copy_(initialized)

    @staticmethod
    def _language_lora_targets(
        model: nn.Module, requested_suffixes: Sequence[str]
    ) -> list[str]:
        """Resolve exact language-module names and exclude the visual encoder."""

        targets = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(suffix) for suffix in requested_suffixes):
                continue
            lowered = name.lower()
            if any(tag in lowered for tag in ("visual", "vision", "audio_tower")):
                continue
            targets.append(name)
        if not targets:
            raise RuntimeError(
                f"No language LoRA modules matched suffixes {list(requested_suffixes)}"
            )
        return targets

    def train(self, mode: bool = True):
        # LoRA dropout and modules_to_save must follow the requested mode.  Base
        # parameters remain frozen through requires_grad=False.
        super().train(mode)
        return self

    def _messages(
        self, video_path: str, prompt: str, *, include_response: bool,
        target_description: str | None = None
    ) -> list[dict[str, Any]]:
        video_uri = Path(video_path).resolve().as_uri()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Given a video and a referring expression, identify the exact "
                    "visible sound source whose audio should be extracted from the "
                    "mixture. Return the required short target-sound answer."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_uri, "fps": self.fps},
                    {
                        "type": "text",
                        "text": f"Referring expression: {prompt}",
                    },
                ],
            },
        ]
        if include_response:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": build_audseg_response(target_description)}],
                }
            )
        return messages

    def _template_kwargs(self) -> dict[str, Any]:
        return {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "fps": self.fps,
        }

    @staticmethod
    def _pop_assistant_mask(inputs: dict[str, Any]) -> torch.Tensor | None:
        for key in ("assistant_masks", "assistant_mask", "assistant_tokens_mask"):
            if key in inputs:
                value = inputs.pop(key)
                return value if torch.is_tensor(value) else torch.as_tensor(value)
        return None

    def _tokenize_messages(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        if self.backend == "qwen2_5_vl":
            return self.processor.apply_chat_template(messages, **kwargs)
        from qwen_omni_utils import process_mm_info

        # This backend uses the video container's synchronized MIXTURE audio.
        # Never insert isolated target/residual audio into these messages.
        text = self.processor.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=kwargs.get("add_generation_prompt", False),
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=True
        )
        return dict(self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        ), use_audio_in_video=True)

    def _prepare_one(self, video_path: str, prompt: str, target_description: str | None = None) -> dict[str, Any]:
        """Tokenize once when the chat template exposes an assistant mask.

        Older Transformers/Qwen templates may not expose that mask.  The fallback
        constructs the prompt separately and verifies it is an exact prefix before
        masking it.  It is slower, but never guesses a dynamic video-token offset.
        """

        messages = self._messages(video_path, prompt, include_response=True,
                                  target_description=target_description)
        kwargs = self._template_kwargs()
        try:
            inputs = self._tokenize_messages(
                messages,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                **kwargs,
            )
        except (TypeError, ValueError):
            inputs = self._tokenize_messages(
                messages, add_generation_prompt=False, **kwargs
            )
        inputs = dict(inputs)
        assistant_mask = self._pop_assistant_mask(inputs)

        if assistant_mask is None or not assistant_mask.bool().any():
            prompt_inputs = self._tokenize_messages(
                self._messages(video_path, prompt, include_response=False),
                add_generation_prompt=True,
                **kwargs,
            )
            prompt_ids = prompt_inputs["input_ids"]
            full_ids = inputs["input_ids"]
            prompt_length = prompt_ids.size(1)
            if not torch.equal(full_ids[:, :prompt_length], prompt_ids):
                raise RuntimeError(
                    "The Qwen chat template did not preserve the prompt as an exact "
                    "prefix; cannot build leakage-free assistant labels safely"
                )
            assistant_mask = torch.zeros_like(full_ids, dtype=torch.bool)
            assistant_mask[:, prompt_length:] = True
        else:
            assistant_mask = assistant_mask.to(dtype=torch.bool)
            if assistant_mask.ndim == 1:
                assistant_mask = assistant_mask.unsqueeze(0)
            if assistant_mask.shape != inputs["input_ids"].shape:
                raise RuntimeError(
                    "assistant token mask does not match input_ids: "
                    f"{assistant_mask.shape} vs {inputs['input_ids'].shape}"
                )

        labels = inputs["input_ids"].clone()
        labels[~assistant_mask] = IGNORE_INDEX
        if "attention_mask" in inputs:
            labels[~inputs["attention_mask"].bool()] = IGNORE_INDEX
        supervised_markers = labels.eq(self.audseg_token_id).sum().item()
        if supervised_markers != 1:
            raise RuntimeError(
                f"Expected one assistant <AUDSEG> label, found {supervised_markers}"
            )
        inputs["labels"] = labels
        return {
            key: value.to(self._device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    def _encode_one(
        self, video_path: str, prompt: str, target_description: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._prepare_one(video_path, prompt, target_description)
        output = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last_hidden = output.hidden_states[-1]
        h_before = select_audseg_h_before(
            last_hidden, inputs["labels"], self.audseg_token_id
        )[0]

        predicts_audseg = inputs["labels"][:, 1:].eq(self.audseg_token_id)
        audseg_logits = output.logits[:, :-1][predicts_audseg]
        token_accuracy = (
            audseg_logits.argmax(dim=-1).eq(self.audseg_token_id).float().mean()
        )
        return h_before, output.loss, token_accuracy

    def forward(
        self, video_paths: list[str], prompts: list[str],
        target_descriptions: list[str | None] | None = None,
    ) -> AudsegEncoderOutput:
        if len(video_paths) != len(prompts):
            raise ValueError("video_paths and prompts must have the same length")
        if not video_paths:
            raise ValueError("video_paths and prompts must not be empty")

        if target_descriptions is None:
            target_descriptions = [None] * len(prompts)
        if len(target_descriptions) != len(prompts):
            raise ValueError("target_descriptions and prompts must have the same length")
        if not self.training and any(x is not None for x in target_descriptions):
            raise ValueError("Ground-truth descriptions are training labels, not inference inputs")

        # Sequential video processing avoids padding fragile variable-size visual
        # token grids.  Gradients still accumulate across every sample in the stack.
        encoded = [
            self._encode_one(video_path, prompt, description)
            for video_path, prompt, description in zip(
                video_paths, prompts, target_descriptions, strict=True
            )
        ]
        hidden, losses, accuracies = zip(*encoded, strict=True)
        return AudsegEncoderOutput(
            hidden_state=torch.stack(hidden),
            ce_loss=torch.stack(losses).mean(),
            token_accuracy=torch.stack(accuracies).mean(),
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return LoRA plus the saved token embedding/LM-head modules."""

        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.model)

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state)
