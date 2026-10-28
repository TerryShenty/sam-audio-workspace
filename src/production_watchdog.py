"""Per-worker watchdog; terminates this worker, never the parent allocation."""
import json,math,os,signal,threading,time,traceback
from pathlib import Path

def cgroup_memory():
    base=Path('/sys/fs/cgroup');relative=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    path=base/relative.lstrip('/')
    while path!=base:
        cap=path/'memory.max'
        if cap.exists() and cap.read_text().strip()!='max':
            return dict(path=str(path),current=int((path/'memory.current').read_text()),limit=int(cap.read_text()),
                stat={k:int(v) for k,v in (line.split() for line in (path/'memory.stat').read_text().splitlines())},
                events={k:int(v) for k,v in (line.split() for line in (path/'memory.events').read_text().splitlines())})
        path=path.parent
    raise RuntimeError('No finite allocation cgroup memory limit')

def check(sample,*,now,deadline,last_progress,phase_timeout,emergency_free_bytes=512*2**20):
    assert all(math.isfinite(x) for x in [now,deadline,last_progress,phase_timeout]) and phase_timeout>0
    assert now<deadline,'FIXED_ALLOCATION_DEADLINE'
    assert now-last_progress<phase_timeout,'WORKER_PROGRESS_TIMEOUT'
    assert sample['physical_free']>=emergency_free_bytes,'PHYSICAL_GPU_EMERGENCY_RESERVE'
    host=sample['host'];assert host['limit']>0 and host['current']<.9*host['limit'],'AGGREGATE_HOST_RESERVE'
    # GPU utilization is recorded; actual update participation is qualified separately.

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

class WorkerWatchdog:
    def __init__(self,torch,device,*,rank,output,deadline,phase_timeouts):
        import pynvml as nv
        from gpu_identity_v3 import bind
        assert set(phase_timeouts)=={'construct','qualify','train','checkpoint','restore','validate'}
        assert all(math.isfinite(v) and v>0 for v in phase_timeouts.values())
        self.nv=nv;nv.nvmlInit();self.handle,binding=bind(torch,device,nv)
        self.root=Path(output);self.root.mkdir(parents=True,exist_ok=True)
        self.rank=rank;self.deadline=deadline;self.timeouts=dict(phase_timeouts);self.pid=os.getpid()
        self.progress=('construct',0,time.time());self.stop=threading.Event();self.samples=[]
        write(self.root/f'gpu-binding-rank{rank}.json',binding)
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def tick(self,phase,completed):
        assert phase in self.timeouts and type(completed) is int and completed>=0
        previous=self.progress
        # Repeated polling cannot conceal a stalled operation.
        if (phase,completed)!=previous[:2]:self.progress=(phase,completed,time.time())
    def watch(self):
        try:
            while not self.stop.wait(5):
                phase,count,last=self.progress;now=time.time()
                memory=self.nv.nvmlDeviceGetMemoryInfo(self.handle)
                sample=dict(time=now,rank=self.rank,phase=phase,completed=count,physical_used=memory.used,
                    physical_free=memory.free,physical_total=memory.total,
                    utilization=self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu,host=cgroup_memory())
                self.samples.append(sample);self.samples=self.samples[-120:]
                write(self.root/f'telemetry-rank{self.rank}.json',dict(samples=self.samples))
                check(sample,now=now,deadline=self.deadline,last_progress=last,phase_timeout=self.timeouts[phase])
        except BaseException:
            write(self.root/f'failure-rank{self.rank}.json',dict(time=time.time(),pid=self.pid,traceback=traceback.format_exc()))
            os.kill(self.pid,signal.SIGTERM)
    def close(self):self.stop.set();self.thread.join(timeout=6)
