"""Isolated candidate for SAM's ONE-token cross memory; never auto-installed.

Softmax over one key is one. Preserve head permutation, LoRA dropout call
order, and material zero gradients (AdamW distinguishes zero from None).
Only the frozen output projection is collapsed across query positions.
"""
import types
import torch


def _zero_projection(module, value):
    # Keep the stochastic call in its original position. Its result need not
    # be retained because the corresponding Q/K derivatives are zero.
    if hasattr(module, 'lora_A'):
        module.dropout(value)
    zero = value[..., :1] * 0
    for p in module.parameters():
        if p.requires_grad:
            zero = zero + p.reshape(-1)[0] * 0
    return zero


def singleton_forward(self, x, cross_x=None, key_padding_mask=None, rope=None):
    if (not self.singleton_enabled or cross_x is None or cross_x.shape[1] != 1 or rope is not None
            or self.n_heads != self.n_kv_heads
            or (key_padding_mask is not None and
                (key_padding_mask.dtype != torch.bool or key_padding_mask.shape != cross_x.shape[:2]))):
        return self._singleton_original(x, cross_x, key_padding_mask, rope)
    qzero = _zero_projection(self.wq, x)
    kzero = _zero_projection(self.wk, cross_x)
    if self.use_qk_norm:
        for mod in (self.q_norm, self.k_norm):
            for p in mod.parameters():
                if p.requires_grad:
                    kzero = kzero + p.reshape(-1)[0] * 0
    value = self.wv(cross_x)
    # SAM stores features as [head_dim, heads], but joins as [heads, head_dim].
    value = self.reshape_heads(value, self.n_kv_heads).transpose(1, 2).reshape_as(value)
    if key_padding_mask is not None:
        value = value * key_padding_mask.unsqueeze(-1)
    value = value + kzero
    n = x.shape[1]
    if hasattr(self.wo, 'lora_A'):
        assert not any(p.requires_grad for p in self.wo.base_layer.parameters())
        base = self.wo.base_layer(value).expand(-1, n, -1)
        expanded = value.expand(-1, n, -1).contiguous()
        update = self.wo.lora_B(self.wo.lora_A(self.wo.dropout(expanded)))
        output = base + update * self.wo.scaling
    else:
        output = self.wo(value).expand(-1, n, -1)
    return output + qzero


def install(module):
    assert not hasattr(module, '_singleton_original')
    module._singleton_original = module.forward
    module.singleton_enabled = True
    module.forward = types.MethodType(singleton_forward, module)


def uninstall(module):
    module.forward = module._singleton_original
    del module._singleton_original
    del module.singleton_enabled
