"""Read-only, byte-bounded production inputs; original per-row supervision."""
import collections, json, types
from pathlib import Path
import torch

def tree(x,fn):
    if torch.is_tensor(x):return fn(x)
    if isinstance(x,dict):return {k:tree(v,fn) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(tree(v,fn) for v in x)
    return x
def tensor_bytes(x):
    sizes=[];tree(x,lambda t:sizes.append(t.numel()*t.element_size()) or t);return sum(sizes)

class ByteCache:
    def __init__(self,limit,pinned=False):
        assert limit>0;self.limit=limit;self.pinned=pinned;self.used=0;self.values=collections.OrderedDict()
    def get(self,key,loader):
        if key in self.values:
            self.values.move_to_end(key);return self.values[key][0]
        value=loader();size=tensor_bytes(value)
        def check(t):
            assert t.device.type=='cpu' and not t.requires_grad and t.grad_fn is None
            if t.is_floating_point():assert bool(torch.isfinite(t).all()),key
            return t
        tree(value,check)
        while self.values and self.used+size>self.limit:
            _,(_,n)=self.values.popitem(last=False);self.used-=n
        # A large single example remains pageable instead of exceeding pinned budget.
        if size>self.limit:return value
        if self.pinned:value=tree(value,lambda t:t.pin_memory())
        self.values[key]=(value,size);self.used+=size;return value

def apply_spans(args,intervals):
    from sam_audio.processor import Batch
    b=Batch.__new__(Batch);b.audios=torch.empty((1,1,0),device=args['audio_features'].device)
    b.audio_pad_mask=args['audio_pad_mask'].cpu();b.hop_length=1920;b.audio_sampling_rate=48000
    Batch.process_anchors(b,[[['+',float(a),float(z)] for a,z in intervals]])
    out=dict(args)
    if (b.anchor_ids.gather(1,b.anchor_alignment)==1).any():
        out['anchor_ids']=b.anchor_ids;out['anchor_alignment']=b.anchor_alignment
    return out

class ProductionStore:
    def __init__(self,namespace,spans,limit=512*2**20,pinned=True):
        self.root=Path(namespace);self.spans=spans;self.cache=ByteCache(limit,pinned)
    def component(self,kind,key):
        assert kind in ('mixture','video','clean'), 'SAM text cache cannot supply joint conditioning'
        p=self.root/kind/key[:2]/(key+'.pt')
        return self.cache.get((kind,key),lambda:torch.load(p,map_location='cpu',weights_only=True))
    def assemble(self,item):
        refs=item['refs'];args=dict(self.component('mixture',refs['mixture'])['forward_args'])
        args.pop('text_features',None);args.pop('text_mask',None)
        args['masked_video_features']=self.component('video',refs['video'])
        clean=self.component('clean',refs['clean'])
        assert clean.shape==args['audio_features'].shape and clean.shape[-1]==256
        if item['split']=='speech_train':args=apply_spans(args,self.spans[item['sample_id']])
        return args,clean

def install_qwen_cache(vlm,root,media,identity,key_fn,processor_bytes=128*2**20,av_bytes=512*2**20,pinned=True):
    """Fail closed for missing prepared/frozen inputs; no online raw decoding."""
    root=Path(root);processor_cache=ByteCache(processor_bytes,False);av_cache=ByteCache(av_bytes,pinned)
    vlm.identity=identity;vlm.media=media;vlm.cache=root
    def prepared(self,mk):
        ident=key_fn([identity,mk,media[mk]]);path=root/'processor'/ident[:2]/(ident+'.pt')
        def load():
            rec=torch.load(path,map_location='cpu',weights_only=True);assert rec['identity']==ident;return rec
        return processor_cache.get(ident,load)
    def tower(self,kind,fn,args,kwargs):
        assert kind in ('audio','video')
        ident=key_fn([identity,self.active_key,kind]);path=root/'encoders'/ident[:2]/(ident+'.pt')
        def load():
            rec=torch.load(path,map_location='cpu',weights_only=True);assert rec['identity']==ident;return rec['value']
        value=av_cache.get(ident,load)
        return tree(value,lambda t:t.to(self.device_,non_blocking=pinned))
    vlm.prepared=types.MethodType(prepared,vlm);vlm.cached_tower=types.MethodType(tower,vlm)
    return processor_cache,av_cache
