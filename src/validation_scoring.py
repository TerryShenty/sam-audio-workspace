"""Source-bound original scorers, without legacy model imports or waveform LRU."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
KEYS=['si_sdr_db','aligned_si_sdr_db','si_sdri_db','activity_primary_f1']

def load(root,domain):
    assert domain in ('music','speech')
    root=Path(root);folder=Path(__file__).resolve().parent/'validation_reference'/domain
    manifest=json.loads((root/'VALIDATION_LOCAL_BINDING.json').read_text())
    for name in ['scoring.py','gt_metric_protocol.py','media_utils.py']:
        p=folder/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['sources'][str(p)]['sha256']
    def module(name):
        identifier='_production_validation_'+domain+'_'+name
        spec=importlib.util.spec_from_file_location(identifier,folder/(name+'.py'))
        value=importlib.util.module_from_spec(spec);sys.modules[identifier]=value
        spec.loader.exec_module(value);return value
    metrics=module('gt_metric_protocol');media=module('media_utils')
    nodes=ast.parse((folder/'scoring.py').read_text()).body
    selected=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in ('mono','reference_inputs','score')]
    assert [n.name for n in selected]==['mono','reference_inputs','score']
    for n in selected:n.decorator_list=[]
    namespace=dict(np=np,sf=sf,Path=Path,metrics=metrics,decode_audio_tensor=media.decode_audio_tensor,KEYS=KEYS)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(folder/'scoring.py'),'exec'),namespace)
    # Function bodies are unchanged. Removing caching bounds lifetime to one score.
    namespace['_audio_map']=manifest['audio_paths']
    return namespace

def score_item(namespace,entry,prediction):
    row=dict(entry['item']['row']);row['row_index']=entry['protocol']['original_row_index']
    for key in ('gt_audio_path','mixture_audio_path'):
        row[key]=namespace['_audio_map'][row[key]]
    result=namespace['score'](row,prediction)
    result.update(sample_id=entry['item']['sample_id'],cohort_domain=entry['protocol']['domain'])
    return result
