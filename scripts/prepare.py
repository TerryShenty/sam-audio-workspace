"""CPU-only preparation. Never submits a job or downloads dataset/model assets."""
import argparse,hashlib,importlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent.parent;V=P/'src'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
    from verify_package import verify
    verify()
    a=argparse.ArgumentParser();a.add_argument('--paths',type=Path,required=True);a=a.parse_args()
    assert not os.environ.get('SLURM_JOB_GPUS'),'Run this on CPU resources'
    paths=json.loads(a.paths.read_text())
    for key in ('data_root','sam_namespace','qwen_model','sam_model'):
        paths[key]=str(Path(paths[key]).expanduser().resolve());assert Path(paths[key]).is_dir(),key
    data=Path(paths['data_root']);required=['PROCESSOR_COMPLETE.json','QWEN_AV_COMPLETE.json','MEDIA.json','BOUND_ROWS.json','TRAIN_SPANS.json','base_identity.json','TARGET_DEMANDING_CASES_CPU.json','VALIDATION_PROTOCOL_BINDING.json']
    for n in required:assert (data/n).is_file(),f'Missing HF dataset metadata: {n}'
    for kind,records in json.loads((P/'configs/MODEL_SHA256.json').read_text()).items():
        for name,digest in records.items():assert sha(Path(paths[kind+'_model'])/name)==digest,('Model mismatch',kind,name)
    from relocate_validation import bind_validation
    bind_validation(data,V)
    write(P/'local_paths.json',paths);os.environ['QWEN_DATA_ROOT']=str(data)
    sys.path.insert(0,str(V));from production_bootstrap import install
    install()
    from dependency_seal import verify
    verify(V)
    import torch
    from portable_codec import prepare as prepare_codec
    prepare_codec(paths['sam_model'],P/'codec_assets')
    assert not torch.cuda.is_initialized()
    natives=[]
    for name in ['unique_scatter_round3','bulk_gather_support_round6','pointwise_moe_round8','buffers_moe_round9']:
        m=importlib.import_module(name);e=m.extension();p=Path(e.__file__).resolve()
        assert p.is_relative_to(V)
        natives.append({'module':name,'path':str(p),'sha256':sha(p)})
    assert not torch.cuda.is_initialized()
    write(V/'native_extensions.json',natives)
    from production_input_admission import prepare,load
    admission=data/'PRODUCTION_INPUT_ADMISSION.json'
    digest=sha(admission) if admission.exists() else prepare(data)
    load(data,digest)
    from deployment_guard import configuration,GROUPS
    effective=json.loads((P/'configs/effective_config.json').read_text());configuration(V/'QWEN_SAM_FAST_BASELINE_V6.json',effective,GROUPS)
    import importlib.metadata as metadata
    env={n:metadata.version(n) for n in ('torch','transformers','bitsandbytes','triton','numpy')}
    write(P/'CPU_READY.json',{'status':'CPU_PREPARED_TARGET_GPU_UNTESTED','paths_sha256':sha(P/'local_paths.json'),'input_admission_sha256':digest,'native_sha256':sha(V/'native_extensions.json'),'environment':env,'codec_admission_sha256':sha(P/'codec_assets/CODEC_READY.json')})
    print('CPU preparation complete. Four-GPU execution is still unverified.')
if __name__=='__main__':main()
