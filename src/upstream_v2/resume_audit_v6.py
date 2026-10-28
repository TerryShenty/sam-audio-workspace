"""Separate exact checkpoint restoration from floating-point GPU replay evidence."""
import hashlib, math
import numpy as np
import torch


def fingerprint(value):
    h=hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous();h.update(str((a.shape,a.dtype)).encode());h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.shape,x.dtype)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))):h.update(repr(k).encode());walk(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for y in x:walk(y)
        else:h.update(repr(x).encode())
    walk(value);return h.hexdigest()


def compare(a,b):
    groups={};changed=[];structural=[]
    def add(path,x,y):
        domain=path.split('/')[0];g=groups.setdefault(domain,dict(numel=0,changed=0,ref_sq=0.,diff_sq=0.,max_abs=0.))
        g['numel']+=x.numel()
        if torch.equal(x,y):return
        if not x.is_floating_point():structural.append(path+': exact integer/state mismatch');return
        xf=x.to(torch.float64);yf=y.to(torch.float64)
        if not bool(torch.isfinite(xf).all() and torch.isfinite(yf).all()):structural.append(path+': nonfinite');return
        d=xf-yf;ds=float(d.square().sum());rs=float(xf.square().sum());mx=float(d.abs().max()) if d.numel() else 0.
        g['changed']+=int((x!=y).sum());g['ref_sq']+=rs;g['diff_sq']+=ds;g['max_abs']=max(g['max_abs'],mx)
        changed.append(dict(path=path,relative_l2=math.sqrt(ds/max(rs,1e-300)),max_abs=mx,diff_sq=ds,ref_sq=rs))
    def walk(x,y,path):
        if torch.is_tensor(x):
            if not torch.is_tensor(y) or x.shape!=y.shape or x.dtype!=y.dtype:structural.append(path+': tensor structure mismatch');return
            add(path,x,y)
        elif isinstance(x,np.ndarray):
            if not isinstance(y,np.ndarray) or x.dtype!=y.dtype or not np.array_equal(x,y):structural.append(path+': numpy mismatch')
        elif isinstance(x,dict):
            if not isinstance(y,dict) or set(x)!=set(y):structural.append(path+': keys mismatch');return
            for k in x:walk(x[k],y[k],path+'/'+str(k) if path else str(k))
        elif isinstance(x,(tuple,list)):
            if type(x)!=type(y) or len(x)!=len(y):structural.append(path+': sequence mismatch');return
            for i,(xx,yy) in enumerate(zip(x,y)):walk(xx,yy,path+'/'+str(i))
        elif x!=y:structural.append(path+': scalar mismatch')
    walk(a,b,'')
    # Relative norm uses changed tensors only, making the bound conservative.
    for g in groups.values():g['relative_l2']=math.sqrt(g['diff_sq']/max(g['ref_sq'],1e-300))
    return dict(exact=not structural and not changed,structural=structural,groups=groups,
                changed_tensors=len(changed),largest_changes=sorted(changed,key=lambda x:x['diff_sq'],reverse=True)[:30])


def acceptable(report):
    return not report['structural'] and all(g['relative_l2'] <= (1e-3 if k=='optimizer' else 1e-6)
        for k,g in report['groups'].items())
