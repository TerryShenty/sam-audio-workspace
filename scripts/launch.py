"""One torchrun, N persistent ranks in the existing allocation. No sbatch loop."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent.parent;V=P/'src'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    from verify_package import verify
    verify()
    a=argparse.ArgumentParser();a.add_argument('--gpus',type=int,default=4);a.add_argument('--mode',choices=['qualify','train'],default='train');a.add_argument('--output',type=Path,required=True);a.add_argument('--gate',type=Path);a.add_argument('--resume',type=Path);a=a.parse_args()
    assert 1<=a.gpus<=8,'GPUs must be 1-8 on one node'
    assert os.environ.get('SLURM_JOB_ID'),'Must run inside your own Slurm allocation'
    end=int(os.environ['SLURM_JOB_END_TIME']);assert end-time.time()>900
    paths=json.loads((P/'local_paths.json').read_text());ready=json.loads((P/'CPU_READY.json').read_text())
    assert ready['paths_sha256']==sha(P/'local_paths.json')
    assert ready['native_sha256']==sha(V/'native_extensions.json')
    assert ready['codec_admission_sha256']==sha(P/'codec_assets/CODEC_READY.json')
    root=Path(paths['data_root']);os.environ['QWEN_DATA_ROOT']=str(root)
    assert not a.output.exists(),'Use a new output directory, including for resume'
    a.output.mkdir(parents=True);out=a.output.resolve()
    files={str(p.resolve()):sha(p) for p in V.rglob('*.py')}
    source_id=sha(P/'SOURCE_MANIFEST.json')
    identity=hashlib.sha256((source_id+ready['paths_sha256']+ready['input_admission_sha256']+ready['native_sha256']+ready['codec_admission_sha256']).encode()).hexdigest()
    effective=json.loads((P/'configs/effective_config.json').read_text())
    effective.update(world_size=a.gpus,per_rank_batch=2,global_batch=2*a.gpus)
    plan={'root':str(root),'code_root':str(V),'output':str(out),'job_id':os.environ['SLURM_JOB_ID'],'end_epoch':end,'mode':a.mode,'qualification_only':a.mode=='qualify','world_size':a.gpus,'per_rank_batch':2,'global_batch':2*a.gpus,'effective':effective,'source_files':files,'deployment_identity':identity,'sam_namespace':paths['sam_namespace'],'input_admission_sha256':ready['input_admission_sha256'],'checkpoint_reserve_seconds':600,'minimum_physical_free_bytes':2*2**30,'collective_timeout_seconds':600,'phase_timeouts':{'construct':600,'qualify':900,'train':120,'checkpoint':600,'restore':600,'validate':300},'demanding_cases_sha256':sha(root/'TARGET_DEMANDING_CASES_CPU.json')}
    if a.mode=='train':
        # Direct training is explicitly supported; optional diagnostic receipts are not prerequisites.
        if a.gate:
            gate=json.loads(a.gate.read_text());assert gate['deployment_identity']==identity and gate['qualified'] is True
            assert gate['full817_validation_completed'] is True
        if a.resume:plan['resume']=str(a.resume.resolve())
    else:assert not a.resume,'Research/qualification checkpoints are not production initialization'
    # Resolve the mandatory runtime modules without importing Torch on the launch parent.
    sys.path.insert(0,str(V));from production_bootstrap import PATHS
    names=['bp_backend','bp_dispatch','bp_live','bp_registry','bp_lifecycle','bytepair_launch','bytepair_kernel','word8_launch','word8_decode_kernel','word4_launch','word4_decode_kernel','nf4_attention_policy','nf4_attention_backward_policy','a30_adapter','int8_fused_rowquant_linear','int8_fused_dynamic_kernel','int8_rowquant_kernel']
    modules={}
    for n in names:
        p=next(r/(n+'.py') for r in PATHS if (r/(n+'.py')).is_file());modules[n]={'path':str(p),'sha256':sha(p)}
    from deployment_guard import BASELINE_SHA
    entry=V/('qualification_entry.py' if a.mode=='qualify' else 'portable_train.py')
    plan['runtime_binding']={'modules':modules,'entry':str(entry),'entry_sha256':sha(entry),'contract_sha256':BASELINE_SHA}
    planpath=out/'plan.json';planpath.write_text(json.dumps(plan,indent=2)+'\n')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node='+str(a.gpus),str(entry),'--plan',str(planpath),'--plan-sha256',sha(planpath)]
    subprocess.run(cmd,check=True)
    if a.mode=='qualify':
        result=json.loads((out/'CHECK_COMPLETE.json').read_text());assert result['status']=='QUALIFICATION_FINISHED'
        (out/'PORTABLE_GATE.json').write_text(json.dumps({'qualified':True,'deployment_identity':identity,'full817_validation_completed':result['full817_validation_completed'],'qualification_output':str(out)},indent=2)+'\n')
if __name__=='__main__':main()
