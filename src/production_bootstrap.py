"""Portable roots; dataset identity remains opaque and unchanged."""
import hashlib,json,os,sys
from pathlib import Path
V=Path(__file__).resolve().parent;U=V/'upstream_v2'
R=Path(os.environ['QWEN_DATA_ROOT']).resolve()
H=V
PATHS=[V,U/'dual_model_research_20261026/nf4_word8_attention_30b_v1',U,U/'runtime_v1',U/'runtime_v1/code',U/'runtime_v1/perf']
def install():
    sys.path[:0]=[str(p) for p in PATHS]
    paths=json.loads((V.parent/'local_paths.json').read_text())
    import common
    original=common.read
    def read(p):
        p=Path(p)
        if p==R/'plan.json':return {'qwen_base':paths['qwen_model']}
        if p==R/'sam_tuning.json':
            cfg=original(V.parent/'configs/sam_tuning.json');cfg['model_local_path']=paths['sam_model'];return cfg
        if p==R/'base_identity.json':return original(R/'base_identity.json')
        if p==R/'media.json':p=R/'MEDIA.json'
        return original(p)
    common.HERE=R;common.ROOT=Path('/');common.read=read
    import bridge
    bridge.HERE=R;bridge.ROOT=Path('/');bridge.read=read
    return bridge

def resolved_sources():
    result={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file:
            p=Path(file).resolve()
            if p.suffix=='.py' and p.is_relative_to(V):
                result[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return result
