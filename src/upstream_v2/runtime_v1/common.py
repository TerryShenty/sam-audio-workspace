import hashlib, json, os, random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[1]
SOURCE = Path('/YOUR_SITE_PATH')

def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def key(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def epoch_batches(n,epoch,seed=2026,world=4,accum=2):
    indices=list(range(n));random.Random(seed+epoch).shuffle(indices)
    width=world*accum
    for start in range(0,n,width):
        valid=indices[start:start+width]
        yield [(valid[i] if i<len(valid) else valid[0],i<len(valid)) for i in range(width)]

def verify_identity(items, expected):
    a=sorted((int(x['row_index']),str(x['clip_id']),str(x['target_source_id']),x['prompt']) for x in items)
    b=sorted((int(x['row']['row_index']),str(x['row']['clip_id']),str(x['row']['target_source_id']),x['row']['prompt']) for x in expected)
    if a!=b:raise RuntimeError('Validation identity mismatch')
