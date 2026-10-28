"""Match physical UUIDs without confusing CUDA's bare UUID with NVML's prefix."""
import os,uuid

def canonical(value):
    if isinstance(value,bytes):
        if len(value)==16:return str(uuid.UUID(bytes=value))
        value=value.decode('ascii')
    return str(uuid.UUID(str(value).removeprefix('GPU-')))

def bind(torch,device,nv):
    props=torch.cuda.get_device_properties(device);raw=str(props.uuid)
    torch.ones(1,device=device).square_();torch.cuda.synchronize(device)
    matches=[]
    for i in range(nv.nvmlDeviceGetCount()):
        handle=nv.nvmlDeviceGetHandleByIndex(i);other=nv.nvmlDeviceGetUUID(handle)
        if isinstance(other,bytes):other=other.decode()
        if canonical(other)==canonical(raw):matches.append((i,handle,other))
    assert len(matches)==1,dict(cuda_uuid=raw,nvml_matches=len(matches))
    index,handle,other=matches[0]
    pids=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]
    assert os.getpid() in pids,dict(cuda_uuid=raw,nvml_uuid=other,pids=pids)
    return handle,dict(cuda_uuid=raw,nvml_uuid=other,canonical_uuid=canonical(raw),
        nvml_index=index,pid=os.getpid(),compute_pids=pids,name=props.name,
        total_bytes=props.total_memory,job=os.environ['SLURM_JOB_ID'],same_nvml_device=True)
