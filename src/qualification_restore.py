"""Verify actual restored full state while its one-rank CPU checkpoint is resident."""
import random
import numpy as np
import torch

def equal(actual,expected,path='state'):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual) and actual.shape==expected.shape and actual.dtype==expected.dtype,path
        assert torch.equal(actual.detach().cpu(),expected.detach().cpu()),path
    elif isinstance(expected,np.ndarray):assert isinstance(actual,np.ndarray) and np.array_equal(actual,expected),path
    elif isinstance(expected,dict):
        assert isinstance(actual,dict) and set(actual)==set(expected),path
        for key in expected:equal(actual[key],expected[key],path+'.'+str(key))
    elif isinstance(expected,(tuple,list)):
        assert isinstance(actual,type(expected)) and len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):equal(a,b,path+'.'+str(i))
    else:assert actual==expected,path

def verify(trainer,state):
    # Compare one tensor at a time, without constructing another full CPU snapshot.
    equal(trainer.params,state['model'],'model')
    equal(dict(trainer.model.named_buffers()),state['buffers'],'buffers')
    equal(trainer.ema,state['ema'],'ema')
    equal(trainer.opt.state_dict(),state['optimizer'],'optimizer')
    equal(trainer.sched.state_dict(),state['scheduler'],'scheduler')
    equal(trainer.sampler.state_dict(),state['sampler'],'sampler')
    equal(trainer.config,state['configuration'],'configuration')
    equal(trainer.groups,state['groups'],'groups')
    equal(trainer.identity,state['identity'],'identity')
    equal(trainer.step_number,state['step'],'step')
    equal(trainer.rank,state['rank'],'rank')
    equal(random.getstate(),state['rng']['python'],'rng.python')
    equal(np.random.get_state(),state['rng']['numpy'],'rng.numpy')
    equal(torch.get_rng_state(),state['rng']['torch'],'rng.torch')
    cuda=[torch.cuda.get_rng_state(trainer.device)] if trainer.device.type=='cuda' else []
    equal(cuda,state['rng']['cuda'],'rng.cuda')

def restore_verified(session,manifest):
    trainer=session.trainer;original=trainer.apply_restore;verified=[]
    def checked(state):
        original(state);verify(trainer,state);verified.append(state['step'])
    trainer.apply_restore=checked
    try:session.restore(manifest)
    finally:trainer.apply_restore=original
    assert verified==[trainer.step_number],'LOCAL_RESTORE_NOT_VERIFIED'
    return dict(rank=trainer.rank,step=trainer.step_number,full_state_exact=True,
                continuation_verified=False)
