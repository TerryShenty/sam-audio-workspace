"""Joint assembly for target qualification; not a production launch command."""
import hashlib,json,random
from pathlib import Path

def inputs(root):
    def read(name):return json.loads((root/name).read_text())
    processor=read('PROCESSOR_COMPLETE.json');av=read('QWEN_AV_COMPLETE.json')
    assert processor['all_media'] and processor['status']=='PASS_CPU_PROCESSOR_ONLY_AV_PENDING'
    assert av['status']=='PASS_FROZEN_AV_ALL_MEDIA' and av['all_finite']
    assert av['media']==12161 and av['components']==24322
    assert not processor['language_hidden_cached'] and not av['language_hidden_cached']
    assert processor['identity']==av['identity']
    media=read('MEDIA.json');rows=read('BOUND_ROWS.json');spans=read('TRAIN_SPANS.json')
    assert len(media)==12161
    assert {r['media_key'] for r in processor['records']}==set(media)
    assert len(processor['records'])==12161
    assert processor['max_actual_tokens']==max(r['max_actual_tokens'] for r in processor['records'])
    train=[r for split,items in sorted(rows.items()) if split.endswith('_train') for r in items]
    validation=[r for split,items in sorted(rows.items()) if split.endswith('_validation') for r in items]
    assert len(train)==111068 and len(validation)==817
    assert len({r['sample_id'] for r in train+validation})==111885
    assert {r['sample_id'] for r in train if r['split']=='speech_train'}==set(spans)
    names=['PROCESSOR_COMPLETE.json','QWEN_AV_COMPLETE.json','MEDIA.json','BOUND_ROWS.json','TRAIN_SPANS.json']
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return dict(processor=processor,media=media,rows=rows,spans=spans,train=train,
                validation=validation,hashes=hashes,identity=identity)

def build(*,root,device,contract_path,immutable_identity,sam_namespace,configuration,input_admission_sha256):
    # All metadata gates precede model loading. Source seal, model stats,
    # physical memory/watchdog and job admission belong to the calling entry.
    from deployment_guard import configuration as check_configuration,GROUPS
    check_configuration(contract_path,configuration,GROUPS)
    world=configuration['world_size']
    assert 1<=world<=8 and configuration['per_rank_batch']==2 and configuration['global_batch']==2*world
    from production_input_admission import load
    data=load(root,input_admission_sha256)
    import numpy as np
    import torch
    import torch.distributed as dist
    assert dist.is_initialized() and dist.get_world_size()==world and dist.get_backend()=='nccl'
    from production_models import construct,optimizer
    from production_runtime import ProductionRuntime
    from joint_ddp import wrap
    from joint_sampler import JointSampler
    from production_data import ProductionStore
    from joint_batch import JointBatch
    from distributed_trainer import DistributedTrainer
    from production_session import ProductionSession
    joint,tuning,caches,evidence=construct(device,cache_root=root/'qwen_cache',
        media=data['media'],cache_identity=data['processor']['identity'],
        processor_bytes=128*2**20,av_bytes=512*2**20)
    check_configuration(contract_path,configuration,evidence['groups'])
    runtime=ProductionRuntime(joint,max_tokens=data['processor']['max_actual_tokens'],contract_path=contract_path)
    wrapped,ddp=wrap(joint,device=device,immutable_identity=immutable_identity)
    opt=optimizer(joint)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:min((n+1)/50,1.))
    rank=dist.get_rank();seed=20261027+rank
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    sampler=JointSampler([r['sample_id'] for r in data['train']],rank=rank,seed=20261027,data_identity=data['identity'],world_size=world)
    trainer=DistributedTrainer(wrapped,opt,scheduler,sampler,
        identity=hashlib.sha256((immutable_identity+data['identity']).encode()).hexdigest(),configuration=configuration)
    runtime.activate_after_training_state_initialized()
    store=ProductionStore(sam_namespace,data['spans'],limit=512*2**20,pinned=True)
    batch=JointBatch(data['rows'],store,device)
    session=ProductionSession(trainer,runtime,batch)
    return session,dict(model=evidence,ddp=ddp,input_hashes=data['hashes'],tuning=tuning,
        processor_cache=caches[0],av_cache=caches[1],sam_cache=store.cache,
        validation_rows=data['validation'],ready_for_production=False)
