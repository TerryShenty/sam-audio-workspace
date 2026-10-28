"""Bounded study: lossless immutable NF4 storage and nested decode fusion.

This module never installs itself. No language hidden states are cached.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(DATA, ABS, ABS2, CODE2, OFFSETS, CODE4, OUT,
            N: tl.constexpr, PER_EXPERT: tl.constexpr,
            BLOCK: tl.constexpr, SCALE_BLOCK: tl.constexpr,
            TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = i < N
    q = tl.load(DATA + i // 2, valid, 0).to(tl.int32)
    nibble = tl.where(i % 2 == 0, q >> 4, q & 15)
    scale_i = i // BLOCK
    scale_code = tl.load(ABS + scale_i, valid, 0).to(tl.int32)
    scale = tl.load(CODE2 + scale_code) * tl.load(ABS2 + scale_i // SCALE_BLOCK, valid, 0)
    scale = scale + tl.load(OFFSETS + i // PER_EXPERT, valid, 0)
    value = tl.load(CODE4 + nibble) * scale
    tl.store(OUT + i, value.to(tl.bfloat16), valid)


def alias_storage(block):
    """Replace each frozen expert's bytes by views of one packed allocation."""
    records = []
    for name, packed in block._packed_nf4.items():
        sources = packed.sources
        assert sources and all(not w.requires_grad and w.dtype == torch.uint8 for w in sources)
        data = torch.cat([w.detach().reshape(-1) for w in sources]).view(-1, 1)
        offset = 0
        for w in sources:
            n = w.numel()
            view = data.flatten().narrow(0, offset, n).view(w.shape)
            assert torch.equal(w, view), 'Frozen bytes changed during packing'
            w.data = view
            assert w.untyped_storage().data_ptr() == data.untyped_storage().data_ptr()
            offset += n
        assert offset == data.numel()
        packed._round7_views = sources
        packed._round7_data = data
        packed.sources = None
        packed.data = data
        records.append({'projection':name, 'packed_bytes':data.numel(), 'storage_shared':True})
    return records


def select_storage(block, use_alias):
    for p in block._packed_nf4.values():
        p.sources = None if use_alias else p._round7_views
        p.data = p._round7_data if use_alias else None


def fused_decode(p):
    assert p.sources is None and p.data.dtype == torch.uint8
    assert p.shape[0]*p.shape[1] % (p.blocksize*p.scale_state.blocksize) == 0
    n = p.experts*p.shape[0]*p.shape[1]
    output = torch.empty((p.experts,*p.shape),device=p.data.device,dtype=torch.bfloat16)
    _decode[(triton.cdiv(n,1024),)](p.data,p.absmax,p.scale_state.absmax,
        p.scale_state.code,p.offsets,p.code,output,n,p.shape[0]*p.shape[1],
        p.blocksize,p.scale_state.blocksize,1024,enable_fp_fusion=False)
    return output


MODE = 'baseline'
_cache = None


def clear_cache():
    global _cache
    _cache = None


def decode(packed):
    global _cache
    reuse = MODE.endswith('_reuse')
    if reuse and _cache is not None and _cache[0] is packed:
        return _cache[1]
    _cache = None
    values = {n:(fused_decode(p) if MODE.startswith('fused') else p.decode()) for n,p in packed.items()}
    if reuse:
        _cache = (packed,values)
    return values
