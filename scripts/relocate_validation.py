"""Bind local scorer sources/audio without modifying opaque cache identities."""
import hashlib,json
from pathlib import Path

def bind_validation(data,version):
    data=Path(data).resolve();version=Path(version).resolve()
    mapping=json.loads((data/'VALIDATION_AUDIO_MAP.json').read_text());paths={}
    for key,relative in mapping.items():
        p=(data.parent/relative).resolve()
        assert p.is_relative_to(data.parent) and p.is_file(),relative
        paths[key]=str(p)
    sources={}
    for domain in ('music','speech'):
        for name in ('scoring.py','gt_metric_protocol.py','media_utils.py'):
            p=version/'validation_reference'/domain/name
            sources[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    value={'status':'LOCAL_SCORER_AND_AUDIO_PATHS_BOUND_NOT_GPU_VALIDATED','sources':sources,'audio_paths':paths}
    out=data/'VALIDATION_LOCAL_BINDING.json'
    text=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if out.exists():assert out.read_text()==text,'Existing validation binding differs; use a separate dataset deployment'
    else:out.write_text(text)
    return value
