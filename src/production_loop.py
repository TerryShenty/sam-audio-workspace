"""Uncapped training loop; admission and measured time reserves are mandatory upstream."""
import math,time
from pathlib import Path

def run(session,output,*,deadline_epoch,checkpoint_reserve_seconds,
        step_reserve_seconds,checkpoint_every_seconds,decide,clock=time.time,on_step=lambda result:None,validation_due=lambda:False,on_validation=lambda path:None):
    # decide reduces all-rank proposals (stop > checkpoint > step); no independent exits.
    values=(deadline_epoch,checkpoint_reserve_seconds,step_reserve_seconds,checkpoint_every_seconds)
    assert all(math.isfinite(v) and v>0 for v in values)
    assert checkpoint_reserve_seconds>step_reserve_seconds
    assert deadline_epoch>clock()+checkpoint_reserve_seconds,'INSUFFICIENT_INITIAL_CHECKPOINT_WINDOW'
    output=Path(output);last_saved=None;last_save_time=clock();overrun=False
    while True:
        now=clock();remaining=deadline_epoch-now
        proposed='stop' if overrun or remaining<=checkpoint_reserve_seconds+step_reserve_seconds else 'validate' if validation_due() else 'checkpoint' if now-last_save_time>=checkpoint_every_seconds else 'step'
        action=decide(proposed)
        assert action in ('stop','checkpoint','validate','step')
        if action in ('checkpoint','stop','validate'):
            step=session.trainer.step_number
            if last_saved!=step:
                started=clock()
                assert deadline_epoch-started>0,'DEADLINE_ALREADY_EXPIRED'
                path=output/('step-%012d'%step)
                session.checkpoint(path)
                elapsed=clock()-started
                last_saved=step;last_save_time=clock()
                # If a measured reserve is exceeded, preserve this checkpoint and fail.
                assert elapsed<=checkpoint_reserve_seconds,'CHECKPOINT_RESERVE_EXCEEDED'
            if action=='validate':
                on_validation(output/('step-%012d'%step))
                last_save_time=clock()
            if action=='stop':
                return dict(step=step,checkpoint=str(output/('step-%012d'%step)),
                            reason='coordinated_stop_checkpoint_boundary',deadline_epoch=deadline_epoch)
        else:
            started=clock();result=session.step();on_step(result)
            # Cold compilation and longest-input qualification must precede this loop.
            if clock()-started>step_reserve_seconds:
                # All ranks enter a coordinated checkpoint on the next decision.
                overrun=True

def collective_decision(proposed):
    import torch
    import torch.distributed as dist
    levels={'step':0,'checkpoint':1,'validate':2,'stop':3}
    assert dist.is_initialized() and dist.get_world_size()==4
    device=torch.device('cuda',torch.cuda.current_device()) if dist.get_backend()=='nccl' else torch.device('cpu')
    value=torch.tensor(levels[proposed],device=device,dtype=torch.int32)
    dist.all_reduce(value,op=dist.ReduceOp.MAX)
    return ('step','checkpoint','validate','stop')[value.item()]
