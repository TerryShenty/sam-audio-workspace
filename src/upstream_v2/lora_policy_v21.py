"""Reversible Qwen q/v LoRA leaf caching composed with the NF4 memo policy.

Register after Retention. No tensor, parameter or optimizer-state replacement.
An ineligible input or zero budget restores the actual original leaf forward.
"""
from functools import partial
import torch
from torch.utils.checkpoint import checkpoint
from lora_recompute_candidate_v2.scoped_linear_cache import ScopedLinearCache


class JointCheckpointPolicy:
    def __init__(self, model, memo):
        self.model, self.memo = model, memo
        self.linear = ScopedLinearCache(0)
        self.first = self.count = 0
        self.originals = [layer._gradient_checkpointing_func for layer in model.layers]
        self.entries = []
        self.active = set()
        self.last_eligible = False
        self.last_cached_layers = []
        # Construct wrappers once and retain their identities across configure
        # transitions. Merely constructing this policy leaves the model plain.
        leaves = torch.nn.ModuleList([proj for layer in model.layers
            for proj in (layer.self_attn.q_proj, layer.self_attn.v_proj)])
        paths = self.linear.install(leaves)
        assert len(paths) == 4 * len(model.layers), paths
        for index, (name, module, original) in enumerate(self.linear.installed):
            projection_index = int(name.split('.')[0])
            assert name.split('.')[-1] in ('lora_A', 'lora_B'), name
            self.entries.append((projection_index // 2, name, module, original, module.forward))
        self.linear.remove()
        self.handle = model.register_forward_pre_hook(self.select, with_kwargs=True)

    def configure(self, max_bytes):
        self.linear.configure(max_bytes)
        # No stale wrapper may survive into a zero-cache reference, even before
        # the model's next input-selection hook executes.
        if not max_bytes:
            self._activate(set())

    def _activate(self, selected):
        if selected == self.active:
            return
        assert self.linear.live_bytes == 0, 'LoRA policy changed with live graph'
        installed = []
        for layer, name, module, original, wrapped in self.entries:
            module.forward = wrapped if layer in selected else original
            if layer in selected:
                installed.append((name, module, original))
        self.linear.installed = installed
        self.active = set(selected)

    def select(self, model, args, kwargs):
        x = kwargs.get('inputs_embeds')
        if x is None:
            x = kwargs.get('input_ids')
        if x is None and args:
            x = args[0]
        assert x is not None
        eligible = x.shape[1] <= 1024 and x.shape[0] * x.shape[1] <= 2048
        cached = {i for i, layer in enumerate(model.layers)
                  if eligible and layer.gradient_checkpointing and self.linear.max_bytes > 0}
        self._activate(cached)
        self.last_eligible = bool(eligible)
        self.last_cached_layers = sorted(cached)
        for i, (layer, original) in enumerate(zip(model.layers, self.originals)):
            use_memo = (eligible and layer.gradient_checkpointing
                        and self.first <= i < self.first + self.count)
            use_linear = i in cached
            layer.self_attn._qwen_lora_cache_variant = 'cached' if use_linear else 'plain'
            if use_memo and use_linear:
                context_fn = self.combined_contexts
            elif use_memo:
                context_fn = self.memo.contexts
            elif use_linear:
                context_fn = self.linear.contexts
            else:
                layer._gradient_checkpointing_func = original
                continue
            layer._gradient_checkpointing_func = partial(checkpoint,
                use_reentrant=False, preserve_rng_state=True, context_fn=context_fn)

    def combined_contexts(self):
        return self.linear.combined_contexts(self.memo.contexts)

    def statistics(self):
        return dict(self.linear.statistics(), eligible=self.last_eligible,
                    cached_layers=list(self.last_cached_layers),
                    original_leaf_forwards_when_disabled=True)

    def remove(self):
        self._activate(set())
        self.handle.remove()
        for layer, original in zip(self.model.layers, self.originals):
            layer._gradient_checkpointing_func = original
            layer.self_attn._qwen_lora_cache_variant = 'plain'
