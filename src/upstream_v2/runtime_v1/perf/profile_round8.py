"""Sparse diagnostic stage timings; not a steady-throughput benchmark."""
import contextlib
import json
import time
from pathlib import Path
import torch


class StepMeter:
    def __init__(self,joint,rank,out,step,relative):
        self.joint,self.rank,self.out,self.step=joint,rank,Path(out),step
        self.enabled=relative in [3,23,24,25]
        self.tracing=False
        self.timings={}
        self.prof=None
        if self.tracing and rank==0:
            self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False)
            self.prof.__enter__()
        if self.enabled:torch.cuda.synchronize()
        self.start=self.last=time.perf_counter()

    def range(self,name):
        return torch.profiler.record_function('round8/'+name) if self.tracing else contextlib.nullcontext()

    def mark(self,name):
        if self.enabled:
            torch.cuda.synchronize()
            now=time.perf_counter();self.timings[name]=now-self.last;self.last=now

    def finish(self,tokens):
        if self.enabled:
            details={}
            events=getattr(self.joint,'last_perf_events',[])
            for (left,a),(right,b) in zip(events,events[1:]):details[right+'_ms']=a.elapsed_time(b)
            record=dict(step=self.step,rank=self.rank,tokens=tokens,stage_seconds=self.timings,
                        total_seconds=self.last-self.start,forward_cuda_event_spans=details,
                        synchronized_diagnostic=True,steady_throughput=False)
            (self.out/f'round8-stages-step{self.step}-rank{self.rank}.json').write_text(json.dumps(record,indent=2)+'\n')
        if self.prof is not None:
            self.prof.__exit__(None,None,None)
            rows=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                       self_device_us=e.self_device_time_total,cpu_us=e.cpu_time_total,
                       device_us=e.device_time_total) for e in self.prof.key_averages()]
            record=dict(step=self.step,rank=self.rank,rows=rows,
                        warning='Profiler perturbs latency; kernel durations may overlap and NCCL duration can include waiting.')
            (self.out/f'round8-operators-rank{self.rank}.json').write_text(json.dumps(record,indent=2)+'\n')
            self.prof.export_chrome_trace(str(self.out/f'round8-rank{self.rank}.trace.json'))
            del self.prof


def timed_forward_backward(ddp,joint,opt,items,args,clean,rng,rng_set,no_sync=False):
    """No parameter/optimizer updates. Caller restores RNG and clears gradients."""
    opt.zero_grad(set_to_none=True);rng_set(rng);torch.cuda.synchronize()
    start=time.perf_counter()
    with ddp.no_sync() if no_sync else contextlib.nullcontext():
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms,tokens=ddp(items,args,clean,[.5,.5])
        assert torch.isfinite(loss);loss.backward()
    torch.cuda.synchronize()
    return time.perf_counter()-start
