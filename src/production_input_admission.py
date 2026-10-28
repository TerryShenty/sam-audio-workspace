"""CPU metadata admission; GPU entry only checks stamps and reads needed metadata."""
import hashlib,json,os
from pathlib import Path

NAMES=('PROCESSOR_COMPLETE.json','QWEN_AV_COMPLETE.json','MEDIA.json','BOUND_ROWS.json','TRAIN_SPANS.json')
def stamp(path):
    st=path.stat()
    return dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)

def prepare(root):
    assert not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS')
    from production_build import inputs
    before={n:stamp(root/n) for n in NAMES}
    data=inputs(root)
    assert before=={n:stamp(root/n) for n in NAMES},'Input changed during CPU admission'
    receipt=dict(status='PASS_CPU_METADATA_ADMISSION_NOT_GPU_READY',files=before,
        hashes=data['hashes'],identity=data['identity'],processor={k:data['processor'][k]
        for k in ('identity','max_actual_tokens')},train_rows=111068,validation_rows=817)
    target=root/'PRODUCTION_INPUT_ADMISSION.json'
    # Exclusive create prevents silently replacing a deployment's input identity.
    payload=json.dumps(receipt,sort_keys=True,indent=2)+'\n'
    temp=root/'PRODUCTION_INPUT_ADMISSION.partial'
    with temp.open('x') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    try:os.link(temp,target)
    finally:temp.unlink()
    return hashlib.sha256(payload.encode()).hexdigest()

def load(root,expected_sha256):
    raw=(root/'PRODUCTION_INPUT_ADMISSION.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected_sha256,'Input admission seal changed'
    receipt=json.loads(raw)
    assert receipt['status']=='PASS_CPU_METADATA_ADMISSION_NOT_GPU_READY'
    assert set(receipt['files'])==set(NAMES)==set(receipt['hashes'])
    assert hashlib.sha256(json.dumps(receipt['hashes'],sort_keys=True).encode()).hexdigest()==receipt['identity']
    def check():
        assert receipt['files']=={n:stamp(root/n) for n in NAMES},'Admitted metadata changed'
    check()
    def read(name):return json.loads((root/name).read_text())
    rows=read('BOUND_ROWS.json');media=read('MEDIA.json');spans=read('TRAIN_SPANS.json')
    train=[r for split,items in sorted(rows.items()) if split.endswith('_train') for r in items]
    validation=[r for split,items in sorted(rows.items()) if split.endswith('_validation') for r in items]
    check()
    return dict(rows=rows,media=media,spans=spans,train=train,validation=validation,
                processor=receipt['processor'],hashes=receipt['hashes'],identity=receipt['identity'])

if __name__=='__main__':
    import sys
    assert len(sys.argv)==2
    print(prepare(Path(sys.argv[1])))
