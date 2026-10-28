"""Add an explicit approximate INT8 input-gradient cache to accepted NF4 attention forward.
Original packed weights, forward INT8 cache and q/v LoRA remain authoritative.
This is not quality-equivalent to the original NF4/BF16 input gradient.
"""
import types
import torch
import nf4_attention_policy as forward_policy
import int8_fused_rowquant_linear as ref
FLAG='qwen_nf4_attention_backward_int8_30b_v1'

class NF4AttentionBackwardPolicy(forward_policy.NF4AttentionPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.backward_cache={};self.backward_cache_parent=self.backward_cache
        self.backward_ids={};self.backward_enabled=False;self.backward_forward_count=None

    def boundary(self):
        super().boundary()
        if not hasattr(self,'backward_cache'):return
        assert self.backward_cache is self.backward_cache_parent,'NF4_ATTN_BACK_CACHE_PARENT'
        if self.backward_cache:
            assert self.enabled and self.backward_enabled,'NF4_ATTN_BACK_PARENT_DISABLED'
            assert len(self.backward_cache)==len(self.entries)
            for e in self.entries:
                item=self.backward_cache.get(id(e.base))
                assert item is not None and (id(item),tuple(forward_policy.stamp(t) for t in item))==self.backward_ids[id(e.base)],'NF4_ATTN_BACK_CACHE_CHANGED'

    def backward_logical_bytes(self):
        return sum(e.base.out_features*e.base.in_features+4*e.base.in_features for e in self.entries)

    def select(self,enabled,backward=False):
        assert type(enabled) is bool and type(backward) is bool and not self.busy(),'NF4_ATTN_BACK_BUSY'
        assert enabled or not backward,'NF4_ATTN_BACK_REQUIRES_FORWARD'
        self.boundary()
        fresh=None;counter=None
        if backward and not self.backward_cache:
            # The accepted forward cache must already be prepared; its preparation
            # is reused and never charged as a second candidate cache.
            assert self.enabled and self.cache,'PREPARE_ACCEPTED_FORWARD_FIRST'
            first=self.entries[0].base.weight;logical=self.backward_logical_bytes()
            # Dense decode plus contiguous transpose may coexist during preparation.
            scratch=max(4*e.base.out_features*e.base.in_features for e in self.entries)
            if first.is_cuda:
                free,total=torch.cuda.mem_get_info(first.device)
                assert free-logical-scratch>=1.5*2**30,'NF4_ATTN_BACK_PHYSICAL_FLOOR'
                assert torch.cuda.memory_allocated(first.device)+logical+scratch+.5*2**30<=.92*total,'NF4_ATTN_BACK_CAP'
            fresh={}
            with torch.no_grad():
                for e in self.entries:
                    dense=forward_policy.decode_weight(e.base.weight,e.base.weight.quant_state)
                    assert dense.shape==(e.base.out_features,e.base.in_features)
                    qb,sb=ref.quant_rows(dense.t().contiguous())
                    fresh[id(e.base)]=(qb.t(),sb.t())
                    del dense
            self.boundary()
            counter=torch.zeros((),dtype=torch.int64,device=first.device)
        # All preparation is complete before changing any live forward.
        super().select(enabled)
        if backward:
            if fresh is not None:
                self.backward_cache,self.backward_cache_parent=fresh,fresh
                self.backward_ids={k:(id(v),tuple(forward_policy.stamp(t) for t in v)) for k,v in fresh.items()}
                self.backward_forward_count=counter
            for e in self.entries:
                qf,sf=self.cache[id(e.base)];qb,sb=self.backward_cache[id(e.base)]
                accepted_count=self.count;candidate_count=self.backward_forward_count
                def forward(base,x,qf=qf,sf=sf,qb=qb,sb=sb,accepted_count=accepted_count,candidate_count=candidate_count):
                    accepted_count.add_(1);candidate_count.add_(1)
                    return ref.FrozenInt8.apply(x,qf,sf,qb,sb,base.bias)
                e.base.forward=types.MethodType(forward,e.base)
        else:
            self.backward_cache.clear();self.backward_ids.clear();self.backward_forward_count=None
        self.backward_enabled=backward

    def backward_dispatch_calls(self):
        return 0 if self.backward_forward_count is None else int(self.backward_forward_count.item())

    def statistics(self):
        r=super().statistics()
        r.update(backward_int8_enabled=self.backward_enabled,
                 backward_dispatch_forward_calls=self.backward_dispatch_calls(),
                 backward_cache_bytes=sum(t.numel()*t.element_size() for item in self.backward_cache.values() for t in item),
                 backward='approximate_dynamic_row_int8_input_gradient' if self.backward_enabled else 'original_nf4_dequant_bf16_matmul',
                 forward_cache_unchanged=True,quality_equivalence_claimed=False)
        return r

    def close(self):self.select(False,False)
