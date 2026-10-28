"""Shape-compatible SAM B2, unchanged per-example loss/global batch.

Each flow state is still sampled independently. Batching changes the ordering
of flow and LoRA-dropout RNG draws; this is disclosed, not bitwise equivalence.
The full saved four-rank RNG states are restored before/after acceptance tests.
"""
import torch
from batch2_joint_round4 import Batch2Joint,condition_batch
SAM_BATCHED=True

def compatible(args,clean):
    if len(args)!=2 or set(args[0])!=set(args[1]):return False
    if clean[0].shape!=clean[1].shape:return False
    for name,a in args[0].items():
        b=args[1][name]
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.ndim==0 or a.shape[0]!=1 or a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device:return False
        elif type(a)!=type(b) or a!=b:return False
    return True

class SAMBatchJoint(Batch2Joint):
    def forward(self,item,args,clean,weight=1.):
        if not SAM_BATCHED or not isinstance(item,list) or not compatible(args,clean):
            self.sam_batch_used=False
            return super().forward(item,args,clean,weight)
        assert len(item)==len(args)==len(clean)==len(weight)==2
        from sam_audio.native_finetuning.objective import make_flow_matching_state
        from sam_audio.native_finetuning.benchmark_utils import per_target_mse
        cond,ce,acc,lengths,_=condition_batch(self,item)
        states=[make_flow_matching_state(x,sigma_min=1e-5) for x in clean]
        packed={k:torch.cat([a[k] for a in args],0) if torch.is_tensor(v) else v for k,v in args[0].items()}
        assert 'text_features' not in packed and 'text_mask' not in packed
        device=next(self.sam.parameters()).device
        pred=self.sam(noisy_audio=torch.cat([x.noisy_latent for x in states],0),time=torch.cat([x.time for x in states]),text_features=cond.to(device),text_mask=torch.ones((2,1),dtype=torch.bool,device=device),**packed)
        velocity=torch.cat([x.target_velocity for x in states],0)
        t=per_target_mse(pred[...,:128],velocity[...,:128],packed['audio_pad_mask']).to(ce.device)
        r=per_target_mse(pred[...,128:],velocity[...,128:],packed['audio_pad_mask']).to(ce.device)
        w=torch.as_tensor(weight,device=ce.device,dtype=t.dtype)
        self.sam_batch_used=True
        return ((.5*t+.5*r+.1*ce)*w).sum(),torch.stack([t.detach(),r.detach(),ce.detach(),acc.detach()],dim=1),lengths
