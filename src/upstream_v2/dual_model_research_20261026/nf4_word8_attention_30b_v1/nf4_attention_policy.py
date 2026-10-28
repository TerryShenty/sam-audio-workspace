"""Frozen NF4 attention bases: derived forward INT8, original NF4 input gradient.

Never self-installs. Cache preparation and policy changes require an idle Trainer.
Original packed parameters and all q/v LoRA work remain authoritative.
"""
import types
from dataclasses import dataclass
import torch
import bitsandbytes as bnb
import int8_fused_rowquant_linear as ref

FLAG = 'qwen_nf4_attention_forward_int8_30b_v1'

def stamp(t):
    if t is None or type(t) in (int, float, bool, str): return t
    assert torch.is_tensor(t)
    return (id(t), t.data_ptr(), t._version, tuple(t.shape), tuple(t.stride()),
            t.dtype, t.device, t.requires_grad)

def quant_stamp(q, depth=0):
    assert q is not None and depth <= 1
    return (id(q), stamp(q.absmax), stamp(q.code), q.blocksize, q.dtype,
            None if q.shape is None else tuple(q.shape), q.quant_type,
            bool(q.nested), stamp(q.offset),
            None if q.state2 is None else quant_stamp(q.state2, depth+1))

def signature(base):
    w = base.weight
    return (stamp(w), stamp(base.bias), quant_stamp(w.quant_state),
            quant_stamp(base.quant_state), base.in_features, base.out_features,
            base.compute_dtype, w.quant_type, w.blocksize, w.compress_statistics)

def decode_weight(weight, state):
    return bnb.functional.dequantize_4bit(weight, state).to(torch.bfloat16)

class NF4ForwardInt8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qf, sf, weight, state, bias):
        ctx.save_for_backward(weight)
        ctx.state, ctx.shape, ctx.dtype = state, x.shape, x.dtype
        y = ref.multiply(x.reshape(-1, x.shape[-1]).to(torch.bfloat16), qf, sf)
        if bias is not None: y = y + bias
        return y.reshape(*x.shape[:-1], qf.shape[1]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        # This is the existing MatMul4Bit input-gradient arithmetic, without a
        # persistent decoded weight. Input was BF16 at the frozen base boundary.
        dense = decode_weight(weight, ctx.state)
        dx = torch.matmul(grad.reshape(-1, grad.shape[-1]).to(torch.bfloat16), dense)
        return dx.reshape(ctx.shape).to(ctx.dtype), None, None, None, None, None

@dataclass
class Entry:
    layer: object
    attention: object
    role: str
    projection: object
    base: object
    original: object
    source: tuple
    had_forward: bool

class NF4AttentionPolicy:
    def __init__(self, layers, busy, *, expected_layers=48, dim=2048, qdim=4096, kv=512):
        assert len(layers) == expected_layers
        self.layers, self.busy = layers, busy
        self.entries, self.cache, self.cache_ids = [], {}, {}
        self.cache_parent = self.cache
        self.enabled, self.count = False, None
        for layer in layers:
            attention = layer.self_attn
            for role, n, k in [('q_proj', qdim, dim), ('k_proj', kv, dim),
                               ('v_proj', kv, dim), ('o_proj', dim, qdim)]:
                projection = getattr(attention, role)
                base = projection.base_layer if role in ('q_proj', 'v_proj') else projection
                assert isinstance(base, bnb.nn.Linear4bit)
                w, q = base.weight, base.weight.quant_state
                assert isinstance(w, bnb.nn.Params4bit) and not w.requires_grad
                assert base.quant_state is q and tuple(q.shape) == (n, k)
                assert q.quant_type == 'nf4' and q.blocksize == 64 and q.nested
                assert q.dtype == torch.bfloat16 and base.compute_dtype == torch.bfloat16
                assert base.bias is None or not base.bias.requires_grad
                if role in ('q_proj', 'v_proj'):
                    assert projection.lora_A.weight.requires_grad and projection.lora_B.weight.requires_grad
                self.entries.append(Entry(layer, attention, role, projection, base,
                    base.forward, signature(base), 'forward' in base.__dict__))
        assert len({id(e.base) for e in self.entries}) == 4*expected_layers

    def boundary(self):
        assert self.cache is self.cache_parent, 'NF4_ATTN_CACHE_PARENT'
        for e in self.entries:
            assert e.layer.self_attn is e.attention, 'NF4_ATTN_LAYER_PARENT'
            assert getattr(e.attention, e.role) is e.projection, 'NF4_ATTN_PROJECTION_PARENT'
            assert (e.projection.base_layer if e.role in ('q_proj','v_proj') else e.projection) is e.base, 'NF4_ATTN_BASE_PARENT'
            assert signature(e.base) == e.source, 'NF4_ATTN_SOURCE_CHANGED'
            if self.cache:
                item = self.cache.get(id(e.base))
                assert item is not None and (id(item), tuple(stamp(t) for t in item)) == self.cache_ids[id(e.base)], 'NF4_ATTN_CACHE_CHANGED'

    def logical_bytes(self):
        return sum(e.base.out_features*e.base.in_features+4*e.base.out_features for e in self.entries)

    def select(self, enabled):
        assert type(enabled) is bool and not self.busy(), 'NF4_ATTN_BUSY'
        self.boundary()
        if enabled and not self.cache:
            first = self.entries[0].base.weight
            logical = self.logical_bytes()
            scratch = max(2*e.base.out_features*e.base.in_features for e in self.entries)
            if first.is_cuda:
                free, total = torch.cuda.mem_get_info(first.device)
                assert free-logical-scratch >= 1.5*2**30, 'NF4_ATTN_PHYSICAL_FLOOR'
                assert torch.cuda.memory_allocated(first.device)+logical+scratch+.5*2**30 <= .92*total, 'NF4_ATTN_CAP'
            fresh = {}
            with torch.no_grad():
                for e in self.entries:
                    dense = decode_weight(e.base.weight, e.base.weight.quant_state)
                    assert dense.shape == (e.base.out_features, e.base.in_features)
                    qf, sf = ref.quant_rows(dense)
                    fresh[id(e.base)] = (qf.t(), sf.t())
                    del dense
            self.boundary()
            count = torch.zeros((), dtype=torch.int64, device=first.device)
            self.cache, self.cache_parent, self.count = fresh, fresh, count
            self.cache_ids = {k:(id(v), tuple(stamp(t) for t in v)) for k,v in fresh.items()}
        for e in self.entries:
            if enabled:
                qf, sf = self.cache[id(e.base)]
                counter = self.count
                def forward(base, x, qf=qf, sf=sf, counter=counter):
                    counter.add_(1)
                    return NF4ForwardInt8.apply(x, qf, sf, base.weight, base.weight.quant_state, base.bias)
                e.base.forward = types.MethodType(forward, e.base)
            elif e.had_forward: e.base.forward = e.original
            else: e.base.__dict__.pop('forward', None)
        self.enabled = enabled
        if not enabled:
            self.cache.clear(); self.cache_ids.clear(); self.count = None

    def calls(self): return 0 if self.count is None else int(self.count.item())

    def statistics(self):
        return dict(enabled=self.enabled, layers=len(self.layers), bases=len(self.entries),
                    calls=self.calls(), derived_cache_bytes=sum(t.numel()*t.element_size() for v in self.cache.values() for t in v),
                    original_packed_parameters_retained=True, backward='original_nf4_dequant_bf16_matmul')

    def close(self): self.select(False)
