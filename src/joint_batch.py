"""Real bound rows -> existing accepted SAMBatchJoint/DDP call, no input dropping."""
import contextlib
import torch
from production_data import tree

class JointBatch:
    def __init__(self,rows,store,device):
        self.items=[row for split,items in sorted(rows.items()) if split.endswith('_train') for row in items]
        assert len(self.items)==111068
        assert len({r['sample_id'] for r in self.items})==111068
        self.store=store;self.device=device;self.last=None

    def __call__(self,wrapped,indices):
        assert len(indices)==2
        items=[self.items[i] for i in indices]
        samples=[self.store.assemble(item) for item in items]
        args=[tree(a,lambda t:t.to(self.device,non_blocking=True)) for a,_ in samples]
        clean=[target.to(self.device,non_blocking=True) for _,target in samples]
        for item,a,target in zip(items,args,clean):
            assert target.shape[-1]==256 and 'text_features' not in a and 'text_mask' not in a
            assert item['split'] in ('music_train','speech_train')
        context=torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else contextlib.nullcontext()
        with context:
            # Average two local per-example objectives; DDP averages four ranks.
            # Original FM target/residual .5/.5 and CE .1 remain inside joint.
            loss,terms,lengths=wrapped(items,args,clean,[.5,.5])
        self.last=dict(sample_ids=[r['sample_id'] for r in items],
                       lengths=list(lengths),terms=terms.detach(),
                       sam_batch_used=bool(getattr(wrapped.module,'sam_batch_used',False)))
        return loss
