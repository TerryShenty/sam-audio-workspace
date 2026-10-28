"""Opt-in saved-activation offload for retained Qwen MoE blocks only.

Synchronous copies prioritize correctness for the first experiment. No async
prefetch/overlap, GPU memory saving, or speed improvement is claimed by CPU tests.
Install AFTER a future isolated worker configures routing, never into live v12.
Weights/buffers sharing protected storage are never transferred by this policy.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import types
import weakref
import torch


def storage_key(t):
    s=t.untyped_storage()
    return (str(t.device),s.data_ptr(),s.nbytes())

@dataclass
class HostSaved:
    device: torch.device
    value: torch.Tensor
    nbytes: int

class SelectiveOffload:
    def __init__(self, *, minimum_bytes=4<<20, maximum_live_bytes=2<<30,
                 pin_memory=True, protected_tensors=(), simulate_cpu=False):
        assert minimum_bytes>=0 and maximum_live_bytes>=0
        self.minimum=minimum_bytes;self.cap=maximum_live_bytes
        self.pin=pin_memory;self.simulate_cpu=simulate_cpu
        self.protected={storage_key(t) for t in protected_tensors}
        self.stats=dict(packed=0,unpacked=0,live_host_bytes=0,peak_host_bytes=0,
                        copied_to_host_bytes=0,copied_to_device_bytes=0,budget_skips=0,
                        protected_skips=0)
        self.patches=[]

    def pack(self,t):
        n=t.numel()*t.element_size()
        if storage_key(t) in self.protected:
            self.stats['protected_skips']+=1
            return t.detach()
        if (not t.is_floating_point() or n<self.minimum or
                (t.device.type!='cuda' and not self.simulate_cpu)):
            return t.detach()
        if self.stats['live_host_bytes']+n>self.cap:
            self.stats['budget_skips']+=1
            return t.detach()
        host=torch.empty_like(t,device='cpu',pin_memory=self.pin and t.device.type=='cuda')
        # No outstanding transfer is allowed before autograd can unpack.
        host.copy_(t.detach(),non_blocking=False)
        saved=HostSaved(t.device,host,n)
        self.stats['packed']+=1;self.stats['live_host_bytes']+=n
        self.stats['copied_to_host_bytes']+=n
        self.stats['peak_host_bytes']=max(self.stats['peak_host_bytes'],self.stats['live_host_bytes'])
        stats=self.stats
        def release():stats['live_host_bytes']-=n
        weakref.finalize(saved,release)
        return saved

    def unpack(self,saved):
        if isinstance(saved,HostSaved):
            self.stats['unpacked']+=1;self.stats['copied_to_device_bytes']+=saved.nbytes
            return saved.value.to(saved.device,non_blocking=False)
        return saved

    @contextmanager
    def context(self):
        with torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack):
            yield

    def install(self,blocks):
        assert not self.patches,'Remove prior scoped policy before changing configuration'
        for block in blocks:
            self.protected.update(storage_key(t) for t in list(block.parameters())+list(block.buffers()))
            original=block.forward;had='forward' in block.__dict__;instance=block.__dict__.get('forward')
            policy=self
            def dispatch(module,*args,_original=original,**kwargs):
                if not torch.is_grad_enabled():return _original(*args,**kwargs)
                with policy.context():return _original(*args,**kwargs)
            block.forward=types.MethodType(dispatch,block)
            self.patches.append((block,had,instance))

    def remove(self):
        for block,had,instance in self.patches:
            if had:block.forward=instance
            else:del block.forward
        self.patches=[]
