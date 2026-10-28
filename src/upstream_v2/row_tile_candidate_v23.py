"""Opt-in static NF4 row tiling; CPU validation is not GPU acceptance.

Install only in an isolated worker, after directional/stage wrappers and before
checkpoint memo. Routing padding and every forward/backward tile must agree.
The default 32-row path calls the exact original functions. Dynamic/long-input
fallback is never wrapped. No packed weights or training parameters change.
"""
from pipeline_launch_candidate_v12 import launch_config

def row_tile(cfg):
    value=cfg.get('nf4_row_tile',32)
    if isinstance(value,bool) or value not in (16,32,64):
        raise ValueError('Unsupported NF4 row tile')
    if value!=32 and not cfg.get('static',False):
        raise ValueError('Alternative row tile requires static routing')
    return int(value)

def capacity(tokens,top_k,experts,tile):
    if min(tokens,top_k,experts)<=0 or top_k>experts or tile not in (16,32,64):
        raise ValueError('Invalid routing geometry')
    return ((tokens*top_k+experts*(tile-1)+tile-1)//tile)*tile

def install_backend(module,choose_config):
    original=module.grouped_mm
    def grouped_mm(a,p,ends,rows,transposed):
        cfg=choose_config();bm=row_tile(cfg)
        if bm==32:return original(a,p,ends,rows,transposed)
        v=launch_config(cfg,transposed)
        if v['stages']!=1:raise ValueError('Row tile candidate requires measured stage-one stack')
        assert a.is_contiguous() and a.dtype==module.torch.bfloat16 and p.sources is None
        n,k=p.shape if transposed else p.shape[::-1]
        assert a.shape[1]==k and rows>0 and rows%bm==0
        output=a.new_zeros(a.shape[0],n)
        ctas=min(rows//bm*module.triton.cdiv(n,v['bn']),
                 module.torch.cuda.get_device_properties(a.device).multi_processor_count*v['waves'])
        module._mm[(ctas,)](a,p.data,p.absmax,p.scale_state.absmax,p.scale_state.code,
            p.offsets,p.code,ends,output,n,k,p.experts,rows,p.shape[0]*p.shape[1],
            transposed,p.blocksize,p.scale_state.blocksize,bm,v['bn'],v['bk'],ctas,
            rows//bm,v['group'],num_warps=v['warps'],num_stages=1,enable_fp_fusion=False)
        return output
    module.grouped_mm=grouped_mm
    def undo():
        assert module.grouped_mm is grouped_mm
        module.grouped_mm=original
    return undo

def install_routing(module,choose_config):
    original=module.forward
    def forward(block,hidden_states):
        bm=row_tile(choose_config())
        if bm==32:return original(block,hidden_states)
        if hidden_states.ndim!=3 or hidden_states.shape[0]!=2 or hidden_states.shape[1]>1024:
            raise ValueError('Caller must route unsupported shapes to original dynamic path')
        shape=hidden_states.shape;x=hidden_states.reshape(-1,shape[-1])
        logits=block.gate(x)
        weights=module.F.softmax(logits,dim=1,dtype=module.torch.float32)
        weights,selected=module.torch.topk(weights,block.top_k,dim=-1)
        if block.norm_topk_prob:weights=weights/weights.sum(-1,keepdim=True)
        out=module.Experts.apply(x,weights.to(x.dtype),selected,block._packed_nf4,bm,module.FP32_OUTPUT)
        return out.view(shape),logits
    module.forward=forward
    def undo():
        assert module.forward is forward
        module.forward=original
    return undo
