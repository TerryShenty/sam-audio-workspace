"""Joint trainer core; actual model/runtime/data entry remains separately gated."""
import copy, random
import numpy as np
import torch
import torch.distributed as dist
from joint_checkpoint import save_rank,publish,load_rank

def tree(value,fn):
    if torch.is_tensor(value):return fn(value)
    if isinstance(value,dict):return {k:tree(v,fn) for k,v in value.items()}
    if isinstance(value,list):return [tree(v,fn) for v in value]
    if isinstance(value,tuple):return tuple(tree(v,fn) for v in value)
    return copy.deepcopy(value)

class DistributedTrainer:
    def __init__(self,wrapped,optimizer,scheduler,sampler,*,identity,configuration):
        assert dist.is_initialized() and dist.get_world_size()==4
        self.rank=dist.get_rank();assert sampler.rank==self.rank
        self.wrapped=wrapped;self.model=wrapped.module
        self.params={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        self.groups={k:[] for k in ('qwen','bridge','sam_lora','sam_conditions')}
        for n in self.params:
            group='qwen' if n.startswith('vlm.') else 'bridge' if n.startswith('bridge.') else 'sam_lora' if '.lora_' in n else 'sam_conditions'
            self.groups[group].append(n)
        assert all(self.groups.values())
        self.opt=optimizer;self.sched=scheduler;self.sampler=sampler
        self.identity=identity;self.config=copy.deepcopy(configuration);self.step_number=0
        self.device=next(iter(self.params.values())).device
        self.ema={n:p.detach().float().clone() for n,p in self.params.items()}
        self.busy=False;self.restore_failed=False

    def require_all(self,valid,message):
        flag=torch.tensor(int(valid),device=self.device,dtype=torch.int32)
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        assert flag.item()==1,message

    def step(self,compute_loss):
        assert not self.busy and not self.restore_failed;self.busy=True
        try:
            indices=self.sampler.peek();self.opt.zero_grad(set_to_none=True)
            # Caller supplies actual joint forward/autocast; DDP must be used here.
            loss=compute_loss(self.wrapped,indices)
            self.require_all(bool(torch.isfinite(loss).all()),'Nonfinite loss on a rank')
            loss.backward()
            active=all(any(self.params[n].grad is not None for n in ns) for ns in self.groups.values())
            norm=torch.nn.utils.clip_grad_norm_(self.params.values(),1.,error_if_nonfinite=False)
            self.require_all(active and bool(torch.isfinite(norm)),'Missing group or nonfinite gradient')
            self.opt.step();self.sched.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(self.ema.values()),[self.params[n].detach().float() for n in self.ema],.001)
            self.sampler.commit();self.step_number+=1
            return loss.detach(),norm.detach()
        finally:self.busy=False

    def snapshot(self):
        assert not self.busy
        clone=lambda x:tree(x,lambda t:t.detach().to(device='cpu',copy=True))
        return dict(model=clone(self.params),buffers=clone(dict(self.model.named_buffers())),
                    optimizer=clone(self.opt.state_dict()),scheduler=copy.deepcopy(self.sched.state_dict()),
                    ema=clone(self.ema),rng=dict(python=random.getstate(),numpy=np.random.get_state(),
                    torch=torch.get_rng_state(),cuda=[torch.cuda.get_rng_state(self.device)] if self.device.type=='cuda' else []),
                    sampler=self.sampler.state_dict(),identity=self.identity,configuration=copy.deepcopy(self.config),
                    step=self.step_number,rank=self.rank,groups=copy.deepcopy(self.groups))

    def checkpoint(self,directory):
        # Only one rank materializes a CPU snapshot at a time. This bounds
        # transient host duplication without omitting any rank's RNG/data state.
        receipts=[]
        for writer in range(4):
            outcome=[None]
            if self.rank==writer:
                try:outcome[0]=dict(receipt=save_rank(directory,self.snapshot()))
                except Exception as error:outcome[0]=dict(error=repr(error))
            dist.broadcast_object_list(outcome,src=writer)
            if 'error' in outcome[0]:raise RuntimeError('Checkpoint writer failed: '+outcome[0]['error'])
            receipts.append(outcome[0]['receipt'])
        outcome=[None]
        if self.rank==0:
            try:publish(directory,receipts);outcome[0]=dict(published=True)
            except Exception as error:outcome[0]=dict(error=repr(error))
        dist.broadcast_object_list(outcome,src=0)
        if 'error' in outcome[0]:raise RuntimeError('Checkpoint publication failed: '+outcome[0]['error'])

    def prepare_restore(self,manifest):
        assert not self.busy
        state=load_rank(manifest,rank=self.rank,identity=self.identity,configuration=self.config)
        assert state['groups']==self.groups and set(state['buffers'])==set(dict(self.model.named_buffers()))
        assert set(state['model'])==set(self.params) and set(state['ema'])==set(self.ema)
        return state

    def apply_restore(self,state):
        assert not self.busy
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,p in self.params.items():p.copy_(state['model'][n])
            for n,p in self.model.named_buffers():p.copy_(state['buffers'][n])
            for n,p in self.ema.items():p.copy_(state['ema'][n])
        self.opt.load_state_dict(state['optimizer']);self.sched.load_state_dict(state['scheduler'])
        self.sampler.load_state_dict(state['sampler']);self.step_number=state['step']
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if self.device.type=='cuda':
            assert len(state['rng']['cuda'])==1
            torch.cuda.set_rng_state(state['rng']['cuda'][0],self.device)

    def serial_restore(self,operation):
        # One rank holds checkpoint CPU tensors at a time, including optimizer state.
        assert not self.busy
        for reader in range(4):
            outcome=[None]
            if self.rank==reader:
                try:
                    operation()
                    outcome[0]=dict(restored=True)
                except Exception as error:outcome[0]=dict(error=repr(error))
            dist.broadcast_object_list(outcome,src=reader)
            if 'error' in outcome[0]:
                self.restore_failed=True
                raise RuntimeError('Serialized restore failed: '+outcome[0]['error'])
        self.restore_failed=False

    def restore(self,manifest):
        self.serial_restore(lambda:self.apply_restore(self.prepare_restore(manifest)))
