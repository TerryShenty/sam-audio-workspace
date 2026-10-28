"""Portable CSV reports; JSON receipts remain authoritative."""
import csv,json
from pathlib import Path

def cell(value):
    if isinstance(value,(dict,list,tuple)):return json.dumps(value,sort_keys=True)
    if isinstance(value,str) and value.startswith(('=','+','-','@')):return "'"+value
    return value

def save(path,rows):
    path=Path(path);rows=list(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:cell(v) for k,v in row.items()} for row in rows)
    temp.replace(path)

def training(path,*,rank,step,loss,grad_norm,elapsed_seconds):
    path=Path(path);exists=path.exists()
    with path.open('a',newline='') as f:
        writer=csv.writer(f)
        if not exists:writer.writerow(['rank','step','processed_epochs_approx','loss_local','grad_norm','elapsed_seconds'])
        writer.writerow([rank,step,step*8/111068,loss,grad_norm,elapsed_seconds])

def validation(path,result):
    path=Path(path)
    shared={k:result[k] for k in ('checkpoint_identity','validation_identity','training_step','validation_seconds','weight_choice','solver','steps')}
    rows=[dict(shared,dataset='ALL',examples=result['examples'],**result['mean'])]
    rows.extend(dict(shared,dataset=k,examples=result['counts'][k],**values) for k,values in result['by_dataset'].items())
    save(path.with_name(path.stem+'-summary.csv'),rows)
    save(path.with_name(path.stem+'-samples.csv'),[dict(shared,**row) for row in result['rows']])
