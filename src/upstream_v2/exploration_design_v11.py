"""Spend the session on new hypotheses; confirm only competitive finalists."""
import json


def identity(cfg):
    return json.dumps({k:v for k,v in cfg.items() if k not in ('name','factors')},sort_keys=True)


def compiler_candidates(incumbent):
    result=[]
    for mode in ('qwen_attention_dynamic','sam_dynamic','both_dynamic','sam_autotune'):
        for retained in (16,24):
            cfg=dict(incumbent, name=f'compile_{mode}_r{retained}', retained=retained, compiler_mode=mode)
            cfg.pop('factors',None)
            result.append(cfg)
    return result


def interleave(*families):
    result=[]; seen=set()
    for i in range(max(map(len,families),default=0)):
        for family in families:
            if i<len(family) and identity(family[i]) not in seen:
                seen.add(identity(family[i]));result.append(family[i])
    return result


def finalists(rows, incumbent, minimum_gain=.015, limit=2):
    controls=[r['seconds_per_step'] for r in rows if r['phase'] in ('incumbent_recheck','time_control')]
    if not controls:return []
    reference=sum(controls[-3:])/len(controls[-3:]);selected=[];seen={identity(incumbent)}
    for row in sorted(rows,key=lambda r:r['seconds_per_step']):
        if row['phase'] not in ('factorial_screen','compiler_screen','refinement_screen','combination_screen'):continue
        cfg=row['config'];key=identity(cfg)
        if key in seen or row['seconds_per_step']>reference*(1-minimum_gain):continue
        seen.add(key);selected.append(cfg)
        if len(selected)==limit:break
    return selected


def paired_order(incumbent, candidate, repeat):
    return [incumbent,candidate] if repeat%2==0 else [candidate,incumbent]


def confirmation_due(new_candidates, last_control):
    return new_candidates-last_control>=8
