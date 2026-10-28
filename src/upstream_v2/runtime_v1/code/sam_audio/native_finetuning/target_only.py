"""Strict post-load conversion of the released joint flow into a target-only flow."""
import types
import torch
from torch import nn

CONVERSION={'objective':'target_only','version':1,'state_channels':128,'input_columns':'0:128,256:768','output_rows':'0:128'}

def convert_target_only(sam):
    from .checkpoint import verified_base
    verified_base(sam)
    if getattr(sam,'_target_only_conversion',None):
        raise ValueError('Model already converted')
    p,o=sam.proj,sam.transformer.output
    if p.in_features!=768 or o.out_features!=256:
        raise ValueError(f'Unexpected pretrained layout: {p.weight.shape}, {o.weight.shape}')
    newp=nn.Linear(640,p.out_features,bias=p.bias is not None,device=p.weight.device,dtype=p.weight.dtype)
    newo=nn.Linear(o.in_features,128,bias=o.bias is not None,device=o.weight.device,dtype=o.weight.dtype)
    with torch.no_grad():
        newp.weight.copy_(torch.cat((p.weight[:,:128],p.weight[:,256:]),dim=1))
        if p.bias is not None:newp.bias.copy_(p.bias)
        newo.weight.copy_(o.weight[:128])
        if o.bias is not None:newo.bias.copy_(o.bias[:128])
    sam.proj=newp;sam.transformer.output=newo
    sam._target_only_conversion=dict(CONVERSION)
    # align_inputs already concatenates state, zeros_like(condition), condition;
    # 128+256+256 now matches the physically sliced 640-column projection.
    sam.separate=types.MethodType(_reject_joint_separate,sam)
    return sam

def _reject_joint_separate(self,*args,**kwargs):
    raise RuntimeError('Target-only model: use sample_target_only; joint separate() is invalid')

@torch.no_grad()
def sample_target_only(sam,forward_args,*,steps=32,seed=2026):
    if getattr(sam,'_target_only_conversion',None)!=CONVERSION:raise ValueError('Not a converted target-only model')
    shape=(*forward_args['audio_features'].shape[:2],128)
    generator=torch.Generator(device=forward_args['audio_features'].device).manual_seed(seed)
    state=torch.randn(shape,device=forward_args['audio_features'].device,dtype=forward_args['audio_features'].dtype,generator=generator)
    if steps<1:raise ValueError('steps must be positive')
    # Fixed-step midpoint ODE; only target state is ever initialized or integrated.
    for i in range(steps):
        t=torch.full((shape[0],),i/steps,device=state.device,dtype=state.dtype)
        v=sam(noisy_audio=state,time=t,**forward_args)
        if v.shape!=state.shape:raise RuntimeError('Non-target velocity shape')
        state=state+sam(noisy_audio=state+v/(2*steps),time=t+.5/steps,**forward_args)/steps
    return state
