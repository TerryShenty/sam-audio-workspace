"""Offload only real retained MoE blocks; checkpoint fallback is respected."""
import types
from selective_offload_v13 import SelectiveOffload

class RetainedOffloadPolicy:
    def __init__(self,layers,blocks,**test_options):
        self.mapping=[]
        for block in blocks:
            matches=[layer for layer in layers if any(m is block for m in layer.modules())]
            assert len(matches)==1,'Each MoE block must map to exactly one language layer'
            self.mapping.append((matches[0],block))
        self.options=test_options;self.patches=[];self.policy=None
    def configure(self,cap):
        assert not self.patches,'Remove policy before changing dispatch'
        assert 0<=cap<=2*2**30
        self.policy=SelectiveOffload(minimum_bytes=self.options.get('minimum_bytes',4<<20),
            maximum_live_bytes=cap,simulate_cpu=self.options.get('simulate_cpu',False))
        if not cap:return
        from pathlib import Path
        status=Path('/proc/self/status')
        if status.exists():
            rss=int(next(line.split()[1] for line in status.read_text().splitlines() if line.startswith('VmRSS:')))*1024
            assert rss+cap<=45*2**30,'Host process RSS plus offload budget needs3GiB reserve in48GiB allocation' 
        protected=[t for _,b in self.mapping for t in list(b.parameters())+list(b.buffers())]
        from selective_offload_v13 import storage_key
        self.policy.protected.update(storage_key(t) for t in protected)
        for layer,block in self.mapping:
            prior=block.forward;had='forward' in block.__dict__;instance=block.__dict__.get('forward')
            policy=self.policy
            def dispatch(module,*args,_prior=prior,_layer=layer,**kwargs):
                # The model's runtime Retention hook runs before any layer.
                # Long/bigger-batch fallback may switch requested retention off.
                import torch
                if not torch.is_grad_enabled() or getattr(_layer,'gradient_checkpointing',True):
                    return _prior(*args,**kwargs)
                with policy.context():return _prior(*args,**kwargs)
            block.forward=types.MethodType(dispatch,block)
            self.patches.append((block,had,instance))
    def statistics(self):
        return dict(self.policy.stats if self.policy else dict(packed=0,unpacked=0,
            live_host_bytes=0,peak_host_bytes=0,copied_to_host_bytes=0,
            copied_to_device_bytes=0,budget_skips=0,protected_skips=0),
            checkpointed_blocks_never_wrapped=True,asynchronous_overlap=False)
    def remove(self):
        assert self.statistics()['live_host_bytes']==0,'Cannot reconfigure offload with live graph'
        for block,had,instance in self.patches:
            if had:block.forward=instance
            else:del block.forward
        self.patches=[]
