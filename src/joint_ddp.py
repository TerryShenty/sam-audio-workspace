"""Synchronize every trainable joint parameter; immutable base is identity-bound."""
import hashlib,json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def wrap(joint,*,device,immutable_identity,bucket_cap_mb=25):
    assert dist.is_initialized() and 1<=dist.get_world_size()<=8
    assert isinstance(immutable_identity,str) and len(immutable_identity)==64
    params=dict(joint.named_parameters())
    trained={n:p for n,p in params.items() if p.requires_grad}
    assert trained and bucket_cap_mb==25
    schema=[(n,list(p.shape),str(p.dtype)) for n,p in trained.items()]
    signature=hashlib.sha256(json.dumps([immutable_identity,schema]).encode()).hexdigest()
    signatures=[None]*dist.get_world_size();dist.all_gather_object(signatures,signature)
    assert len(set(signatures))==1,'Rank source/trainable schema mismatch'
    # Frozen model weights are loaded from the same sealed pretrained identity.
    # Buffers are rank-local, checkpointed and restored; no running BN is allowed.
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in joint.modules())
    ignored={n for n,p in params.items() if not p.requires_grad}|set(dict(joint.named_buffers()))
    assert not ignored.intersection(trained)
    if dist.get_world_size()==1:
        class SingleDevice(torch.nn.Module):
            def __init__(self,module):
                super().__init__();self.module=module
            def forward(self,*args,**kwargs):return self.module(*args,**kwargs)
        return SingleDevice(joint),dict(world_size=1,per_rank_batch=2,global_batch=2,trainable_tensors=len(trained),schema_sha256=signature,ddp_enabled=False,communication_overlap_measured=False)
    DDP._set_params_and_buffers_to_ignore_for_model(joint,ignored)
    wrapped=DDP(joint,device_ids=[device.index] if device.type=='cuda' else None,
        output_device=device.index if device.type=='cuda' else None,
        broadcast_buffers=False,find_unused_parameters=True,
        gradient_as_bucket_view=True,bucket_cap_mb=bucket_cap_mb)
    assert wrapped.parameters_to_ignore==ignored
    return wrapped,dict(world_size=dist.get_world_size(),per_rank_batch=2,global_batch=2*dist.get_world_size(),
        trainable_tensors=len(trained),ignored_frozen_tensors=len(ignored),
        schema_sha256=signature,find_unused_parameters=True,broadcast_buffers=False,
        gradient_as_bucket_view=True,bucket_cap_mb=bucket_cap_mb,
        communication_overlap_measured=False)
