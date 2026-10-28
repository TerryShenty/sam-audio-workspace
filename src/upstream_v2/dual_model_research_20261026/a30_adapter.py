from nf4_attention_policy import NF4AttentionPolicy,FLAG,signature,stamp
from nf4_attention_backward_policy import NF4AttentionBackwardPolicy,FLAG as BACKFLAG
class AttentionRuntime:
    def __init__(self,layers,lifecycle,*,dim=2048,qdim=4096,kv=512):
        self.impl=NF4AttentionBackwardPolicy(layers,lambda:lifecycle.in_step,dim=dim,qdim=qdim,kv=kv)
        self.lifecycle=lifecycle
        self.markers=[(l.self_attn,hasattr(l.self_attn,'_qwen_nf4_forward_int8_variant'),getattr(l.self_attn,'_qwen_nf4_forward_int8_variant','plain')) for l in layers]
        assert all(v=='plain' for _,_,v in self.markers)
        self.back_markers=[(l.self_attn,hasattr(l.self_attn,'_qwen_nf4_backward_int8_variant'),getattr(l.self_attn,'_qwen_nf4_backward_int8_variant',None)) for l in layers]
        assert all(not had for _,had,_ in self.back_markers)
        lifecycle.registry.extra_boundaries=tuple(lifecycle.registry.extra_boundaries)+(self.boundary,)
        lifecycle.compiler.attention_nf4_runtime=self
    def configure(self,enabled,backward=False):
        if backward and not self.impl.enabled:self.impl.select(True,False)
        self.impl.select(enabled,backward)
        for m,_,_ in self.markers:
            m._qwen_nf4_forward_int8_variant=FLAG if enabled else 'plain'
            if backward:m._qwen_nf4_backward_int8_variant=BACKFLAG
            else:m.__dict__.pop('_qwen_nf4_backward_int8_variant',None)
    def layer_signature(self,index):
        p=self.impl;assert p.cache is p.cache_parent,'ATTN_CACHE_PARENT'
        values=[]
        for e in p.entries[4*index:4*index+4]:
            assert e.layer.self_attn is e.attention and getattr(e.attention,e.role) is e.projection,'ATTN_PARENT'
            assert (e.projection.base_layer if e.role in ('q_proj','v_proj') else e.projection) is e.base,'ATTN_BASE'
            assert signature(e.base)==e.source,'ATTN_SOURCE'
            assert getattr(e.attention,'_qwen_lora_cache_variant','plain')=='plain'
            marker=getattr(e.attention,'_qwen_nf4_forward_int8_variant','plain')
            assert marker==(FLAG if p.enabled else 'plain'),'ATTN_FAMILY'
            if p.enabled:
                item=p.cache[id(e.base)];item_stamp=(id(item),tuple(stamp(t) for t in item))
                assert item_stamp==p.cache_ids[id(e.base)],'ATTN_CACHE_TENSOR'
                forward=e.base.__dict__['forward'];assert forward.__self__ is e.base
                values.append((e.source,item_stamp,id(forward),id(forward.__func__.__code__),marker))
            else:
                assert ('forward' in e.base.__dict__)==e.had_forward
                assert e.base.forward==e.original,'ATTN_OFF_FORWARD'
                values.append((e.source,None,marker))
        back=[]
        assert p.backward_cache is p.backward_cache_parent,'BACK_CACHE_PARENT'
        for e in p.entries[4*index:4*index+4]:
            marker=getattr(e.attention,'_qwen_nf4_backward_int8_variant',None)
            assert marker==(BACKFLAG if p.backward_enabled else None),'BACK_FAMILY'
            if p.backward_enabled:
                item=p.backward_cache[id(e.base)];sig=(id(item),tuple(stamp(t) for t in item))
                assert sig==p.backward_ids[id(e.base)],'BACK_CACHE_TENSOR'
                back.append((sig,marker))
            else:
                assert not p.backward_cache
                back.append((None,marker))
        return (id(self),id(p),p.enabled,p.backward_enabled,tuple(values),tuple(back))
    def boundary(self):
        self.impl.boundary()
        for i in range(len(self.impl.layers)):self.layer_signature(i)
    def calls(self):return self.impl.calls()
    def statistics(self):return self.impl.statistics()
    def close(self):
        self.configure(False)
        for m,had,v in self.markers:
            if had:m._qwen_nf4_forward_int8_variant=v
            else:m.__dict__.pop('_qwen_nf4_forward_int8_variant',None)
        for m,had,v in self.back_markers:
            if had:m._qwen_nf4_backward_int8_variant=v
            else:m.__dict__.pop('_qwen_nf4_backward_int8_variant',None)
