"""Losslessly alias frozen NF4 metadata into its already packed copies.

No dequantization, requantization, changed kernels, or cached activations.
Releases thousands of obsolete small allocator islands after expert packing.
"""
import torch

def alias_projection(packed):
    weights=packed._round7_views
    assert len(weights)==packed.experts and all(not w.requires_grad for w in weights)
    states=[w.quant_state for w in weights]
    assert all(s.nested and not s.state2.nested for s in states)
    assert torch.equal(torch.cat([s.absmax.reshape(-1) for s in states]),packed.absmax.reshape(-1))
    assert torch.equal(torch.cat([s.state2.absmax.reshape(-1) for s in states]),packed.scale_state.absmax.reshape(-1))
    assert torch.equal(torch.stack([s.offset for s in states]).reshape(-1),packed.offsets.reshape(-1))
    assert torch.equal(torch.stack([s.code for s in states]),packed.code.expand(len(states),-1))
    assert torch.equal(torch.stack([s.state2.code for s in states]),packed.scale_state.code.expand(len(states),-1))
    a=b=0
    for i,s in enumerate(states):
        assert s.blocksize==packed.blocksize and tuple(s.shape)==tuple(packed.shape)
        assert s.state2.blocksize==packed.scale_state.blocksize
        n=s.absmax.numel();m=s.state2.absmax.numel()
        s.absmax=packed.absmax.reshape(-1).narrow(0,a,n).view_as(s.absmax)
        s.state2.absmax=packed.scale_state.absmax.reshape(-1).narrow(0,b,m).view_as(s.state2.absmax)
        s.offset=packed.offsets[i].view_as(s.offset)
        s.code=packed.code;s.state2.code=packed.scale_state.code
        a+=n;b+=m
    assert a==packed.absmax.numel() and b==packed.scale_state.absmax.numel()
    return len(states)

def install(qwen):
    blocks=[m for m in qwen.modules() if hasattr(m,'_packed_nf4')]
    assert len(blocks)==48
    count=0
    for block in blocks:
        for p in block._packed_nf4.values():count+=alias_projection(p)
    assert count==18432
    return dict(status='PASS_EXACT_PACKED_METADATA_ALIAS',experts=18432,fields_per_expert=5,
                requantized=False,language_hidden_cached=False)
