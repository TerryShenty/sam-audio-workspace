"""Isolated Qwen NF4 + cached native SAM. T5 is structurally absent."""
import contextlib, fcntl, gc, math, os, types
from pathlib import Path
import torch
from torch import nn
from common import HERE,ROOT,SOURCE,read,sha,key,write

def tree(x,fn):
    if torch.is_tensor(x):return fn(x)
    if isinstance(x,dict):return {k:tree(v,fn) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(tree(v,fn) for v in x)
    return x
def finite(x):
    if torch.is_tensor(x):return not x.is_floating_point() or bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
    return True
def atomic_torch(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');torch.save(x,t);t.replace(p)

class SpecialEmbedding(nn.Module):
    def __init__(self,base,token_id,initial):
        super().__init__();self.base=base.requires_grad_(False);self.token_id=token_id
        self.row=nn.Parameter(initial.float().clone())
    @property
    def weight(self):return self.base.weight
    def forward(self,ids):
        return torch.where((ids==self.token_id).unsqueeze(-1),self.row.to(self.base.weight.dtype),self.base(ids))

class SparseHead(nn.Module):
    """Only assistant positions need vocabulary logits; capture h_before causally."""
    def __init__(self,base,token_id,initial):
        super().__init__();self.base=base.requires_grad_(False);self.token_id=token_id
        self.row=nn.Parameter(initial.float().clone());self.positions=None;self.marker_position=None;self.hidden=None
    @property
    def weight(self):return self.base.weight
    def forward(self,h):
        assert self.positions is not None and self.marker_position is not None
        self.hidden=h[:,self.marker_position,:]
        x=h[:,self.positions,:];y=self.base(x)
        special=torch.nn.functional.linear(x,self.row.to(x.dtype).unsqueeze(0))
        return torch.cat([y[...,:self.token_id],special,y[...,self.token_id+1:]],-1)

class FeatureStore:
    """Load audio/video/GT components only. No text cache reads in this store."""
    def __init__(self):
        from sam_audio.native_finetuning.feature_cache import ComponentStore
        self.mapping=read(SOURCE/'cache_mapping.json');u=read(SOURCE/'urmp_cache_source.json')
        self.urmp=ComponentStore(u['cache_root'],u['signature'],lru_entries=4)
    def get(self,kind,k):
        assert kind!='text'
        return torch.load(self.mapping[kind+'/'+k]['path'],map_location='cpu',weights_only=True)
    def assemble(self,refs):
        if refs.get('cache_source')=='urmp':
            get=self.urmp.get;m=get('mixture',refs['mixture']);args=dict(m['forward_args'])
            args['masked_video_features']=get('video',refs['video']);clean=get('clean',refs['clean'])
        else:
            m=self.get('mixture',refs['mixture']);args=dict(m['forward_args'])
            args['masked_video_features']=self.get('video',refs['video'])
            clean=torch.cat([self.get('target',refs['target']),self.get('residual',refs['residual'])],-1)
        args.pop('text_features',None);args.pop('text_mask',None)
        assert args['anchor_ids'].tolist()==[[0,3]],'Unexpected spans'
        assert clean.shape==args['audio_features'].shape
        return args,clean

def materialize_sam_buffers(sam,c):
    from sam_audio.model.model import SinusoidalEmbedding
    # inv_freq is nonpersistent and therefore absent from state_dict/meta loading.
    sam.timestep_emb=SinusoidalEmbedding(c.transformer.dim)
    half=sam.transformer.t_embedder.frequency_embedding_size//2
    sam.transformer.t_embedder.freqs=torch.exp(-math.log(10000)*torch.arange(half,dtype=torch.float32)/half)
    if sam.transformer.rope_embeddings is not None:sam.transformer.rope_embeddings.reset_parameters()
    assert not any(b.is_meta for b in sam.buffers()),'Unmaterialized nonpersistent buffer'

def load_sam(device):
    from sam_audio.model.model import SAMAudio
    from sam_audio.model.config import SAMAudioConfig
    from sam_audio.model.transformer import DiT
    from sam_audio.model.align import AlignModalities
    from sam_audio.model.model import EmbedAnchors,SinusoidalEmbedding
    from sam_audio.native_finetuning.model import configure_native_tuning,native_tuning_config_from_mapping
    cfg=read(HERE/'sam_tuning.json');folder=Path(cfg['model_local_path']);c=SAMAudioConfig(**read(folder/'config.json'))
    class CachedSAM(SAMAudio):
        def __init__(self,c):
            nn.Module.__init__(self)
            self.transformer=DiT(c.transformer);self.proj=nn.Linear(c.in_channels,c.transformer.dim)
            self.align_masked_video=AlignModalities(c.vision_encoder.dim,c.transformer.dim)
            self.embed_anchors=EmbedAnchors(c.num_anchors,c.anchor_embedding_dim,c.transformer.dim)
            self.memory_proj=nn.Linear(c.text_encoder.dim,c.transformer.dim)
            self.timestep_emb=SinusoidalEmbedding(c.transformer.dim)
    with torch.device('meta'):sam=CachedSAM(c)
    state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
    wanted=set(sam.state_dict());assert wanted<=set(state)
    excluded=set(state)-wanted
    assert all(n.startswith(('audio_codec.','vision_encoder.','text_encoder.','span_predictor.','visual_ranker.','text_ranker.')) for n in excluded),sorted(excluded)[:10]
    sam.load_state_dict({k:state[k] for k in wanted},strict=True,assign=True)
    materialize_sam_buffers(sam,c)
    sam.to(device=device,dtype=torch.bfloat16).requires_grad_(False)
    torch.manual_seed(314159)
    tuning=configure_native_tuning(sam,native_tuning_config_from_mapping(cfg))
    assert tuning['trainable_parameters']==74409985,tuning['trainable_parameters']
    assert not hasattr(sam,'text_encoder')
    del state;gc.collect()
    return sam,c,tuning

class QwenBridge(nn.Module):
    def __init__(self,device):
        super().__init__()
        from transformers import Qwen3OmniMoeForConditionalGeneration,Qwen3OmniMoeProcessor,BitsAndBytesConfig
        from sam_audio.native_finetuning.lora import LoRALinear
        self.device_=device;self.media=read(HERE/'media.json');self.cache=HERE/'qwen_cache'
        self.identity=key({'base':read(HERE/'base_identity.json'),'fps':1,'max_pixels':100352,'audio':'explicit_mixture','processor_fast':False,'version':2})
        p=ROOT/read(HERE/'plan.json')['qwen_base']
        full=Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(p),torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
            attn_implementation='sdpa',device_map={'':device.index},low_cpu_mem_usage=True,local_files_only=True)
        full.disable_talker();self.qwen=full.thinker;del full
        self.qwen.requires_grad_(False);self.processor=Qwen3OmniMoeProcessor.from_pretrained(str(p),local_files_only=True,use_fast=False)
        tok=self.processor.tokenizer;tok.add_special_tokens({'additional_special_tokens':['<AUDSEG>','<|SAM_REF_SLOT|>']})
        self.token_id=tok.convert_tokens_to_ids('<AUDSEG>');assert tok.encode('<AUDSEG>',add_special_tokens=False)==[self.token_id]
        base=self.qwen.get_input_embeddings();assert self.token_id<base.weight.shape[0],'Reserved vocabulary row required; do not silently resize'
        initial=base.weight[tok.encode('audio target',add_special_tokens=False)].mean(0).detach()
        self.qwen.set_input_embeddings(SpecialEmbedding(base,self.token_id,initial))
        self.qwen.lm_head=SparseHead(self.qwen.lm_head,self.token_id,initial)
        matches=[n for n,m in self.qwen.named_modules() if n.startswith('model.layers.') and n.endswith(('.self_attn.q_proj','.self_attn.v_proj')) and isinstance(m,nn.Linear)]
        assert len(matches)==self.qwen.config.text_config.num_hidden_layers*2
        for name in matches:
            parent,attr=name.rsplit('.',1);base=getattr(self.qwen.get_submodule(parent),attr)
            layer=LoRALinear(base,rank=8,alpha=16,dropout=.05).to(device)
            setattr(self.qwen.get_submodule(parent),attr,layer)
        self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.qwen.enable_input_require_grads();self.qwen.config.use_cache=False
        self.hidden_size=self.qwen.config.text_config.hidden_size
        self._original_video=self.qwen.get_video_features;self._original_audio=self.qwen.get_audio_features
        # Frozen towers are cacheable. Language hidden states are NEVER cached.
        self.active_key=None;self._av_memory={};self._processor_memory={}
        def video(inner,*args,**kw):return self.cached_tower('video',self._original_video,args,kw)
        def audio(inner,*args,**kw):return self.cached_tower('audio',self._original_audio,args,kw)
        self.qwen.get_video_features=types.MethodType(video,self.qwen)
        self.qwen.get_audio_features=types.MethodType(audio,self.qwen)
        assert not any(p.requires_grad for p in self.qwen.visual.parameters())
        assert not any(p.requires_grad for p in self.qwen.audio_tower.parameters())
    def train(self,mode=True):
        super().train(mode);self.qwen.visual.eval();self.qwen.audio_tower.eval();return self
    def cached_tower(self,kind,fn,args,kwargs):
        ident=key([self.identity,self.active_key,kind]);p=self.cache/'encoders'/ident[:2]/(ident+'.pt')
        if ident not in self._av_memory:
            p.parent.mkdir(parents=True,exist_ok=True)
            with open(str(p)+'.lock','w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if p.exists():record=torch.load(p,map_location='cpu',weights_only=True)
                else:
                    with torch.no_grad():result=fn(*args,**kwargs)
                    record={'identity':ident,'value':tree(result,lambda x:x.detach().cpu())}
                    assert finite(record);atomic_torch(p,record)
                assert record['identity']==ident and finite(record)
            if len(self._av_memory)>=4:self._av_memory.clear()
            self._av_memory[ident]=record['value']
        return tree(self._av_memory[ident],lambda x:x.to(self.device_))
    def prepared(self,media_key):
        from qwen_omni_utils import process_mm_info
        sentinel='<|SAM_REF_SLOT|>'
        if media_key in self._processor_memory:return self._processor_memory[media_key]
        meta=self.media[media_key];ident=key([self.identity,media_key,meta]);p=self.cache/'processor'/ident[:2]/(ident+'.pt');p.parent.mkdir(parents=True,exist_ok=True)
        with open(str(p)+'.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if p.exists():rec=torch.load(p,map_location='cpu',weights_only=True)
            else:
                for m in meta.values():assert sha(m['path'])==m['sha256'],'Media changed after CPU audit'
                messages=[{'role':'system','content':'Identify the exact visible sound source described by the user using the video and synchronized mixture audio. Return only: The target sound is <AUDSEG>.'},
                    {'role':'user','content':[{'type':'video','video':meta['video']['path'],'fps':1,'max_pixels':100352},{'type':'audio','audio':meta['audio']['path']},{'type':'text','text':'Referring expression:\n'+sentinel+'\n'}]},
                    {'role':'assistant','content':'The target sound is <AUDSEG>.'}]
                txt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
                a,im,v=process_mm_info(messages,use_audio_in_video=False)
                assert len(a)==1
                x=dict(self.processor(text=txt,audio=a,images=im,videos=v,return_tensors='pt',padding=True,use_audio_in_video=False))
                assert x['pixel_values_videos'].numel() and x['input_features'].numel()
                ids=x['input_ids'][0].tolist();slot=self.processor.tokenizer.encode(sentinel,add_special_tokens=False)
                at=[i for i in range(len(ids)-len(slot)+1) if ids[i:i+len(slot)]==slot];assert len(at)==1,at
                x=tree(x,lambda t:t.to(torch.bfloat16) if t.is_floating_point() else t)
                rec={'identity':ident,'x':x,'slot':at[0],'slot_length':len(slot)}
                assert finite(rec);atomic_torch(p,rec)
            assert rec['identity']==ident
        if len(self._processor_memory)>=2:self._processor_memory.clear()
        self._processor_memory[media_key]=rec;return rec
    def forward(self,media_key,prompt):
        self.active_key=media_key;rec=self.prepared(media_key);x=dict(rec['x'])
        base=x['input_ids'];tokens=self.processor.tokenizer.encode(prompt,add_special_tokens=False)
        assert self.token_id not in tokens,'Prompt contains reserved marker'
        ids=torch.cat([base[:,:rec['slot']],torch.tensor([tokens],dtype=base.dtype),base[:,rec['slot']+rec['slot_length']:]],1)
        x['input_ids']=ids;x['attention_mask']=torch.ones_like(ids)
        marker=(ids[0]==self.token_id).nonzero().flatten()
        # The system instruction also spells the marker: extract only assistant's last occurrence.
        marker=int(marker[-1]);assert marker>rec['slot']+len(tokens)
        imstart=self.processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        assistant=int((ids[0,:marker]==imstart).nonzero().flatten()[-1])
        # Fixed short response and terminator; no GT text supplied. Empty thinking block is template-only.
        positions=torch.arange(assistant+2,ids.shape[1]-1,device=self.device_)
        self.qwen.lm_head.positions=positions;self.qwen.lm_head.marker_position=marker-1
        x=tree(x,lambda t:t.to(self.device_));self.qwen.rope_deltas=None
        out=self.qwen(**x,use_audio_in_video=False,use_cache=False,output_hidden_states=False,output_router_logits=False,return_dict=True)
        labels=x['input_ids'][:,positions+1]
        loss=torch.nn.functional.cross_entropy(out.logits.float().reshape(-1,out.logits.shape[-1]),labels.reshape(-1))
        h=self.qwen.lm_head.hidden;self.qwen.lm_head.hidden=None
        if getattr(self,'audit_flow_gradient',False) and h.requires_grad:
            # This view only feeds SAM through the bridge; format CE does not use it.
            def record_flow_gradient(g):self.flow_hidden_gradient_norm=float(g.detach().float().norm())
            h.register_hook(record_flow_gradient)
        assert h.shape==(1,self.hidden_size)
        j=(positions==marker-1).nonzero().item();acc=(out.logits[:,j].argmax(-1)==self.token_id).float().mean()
        return h,loss,acc,ids.shape[1]

class Joint(nn.Module):
    def __init__(self,sam,qwen,device):
        super().__init__();self.sam=sam;self.vlm=qwen
        self.bridge=nn.Sequential(nn.LayerNorm(qwen.hidden_size),nn.Linear(qwen.hidden_size,1024),nn.GELU(),nn.Linear(1024,sam.memory_proj.in_features)).to(device)
    def train(self,mode=True):
        super().train(mode)
        # Frozen SAM backbone dropout remains off; only LoRA dropout participates.
        self.sam.eval()
        from sam_audio.native_finetuning.lora import LoRALinear
        for m in self.sam.modules():
            if isinstance(m,LoRALinear):m.dropout.train(mode)
        self.vlm.train(mode);self.bridge.train(mode);return self
    def condition(self,item):
        h,ce,acc,tokens=self.vlm(item['media_key'],item.get('prompt',item.get('row',{}).get('prompt')))
        return self.bridge(h.float()).unsqueeze(1),ce,acc,tokens
    def forward(self,item,args,clean,weight=1.):
        from sam_audio.native_finetuning.objective import make_flow_matching_state
        from sam_audio.native_finetuning.benchmark_utils import per_target_mse
        cond,ce,acc,tokens=self.condition(item)
        assert 'text_features' not in args and 'text_mask' not in args,'T5 bypass forbidden'
        state=make_flow_matching_state(clean,sigma_min=1e-5)
        pred=self.sam(noisy_audio=state.noisy_latent,time=state.time,text_features=cond,text_mask=torch.ones((1,1),dtype=torch.bool,device=cond.device),**args)
        t=per_target_mse(pred[...,:128],state.target_velocity[...,:128],args['audio_pad_mask']).mean()
        r=per_target_mse(pred[...,128:],state.target_velocity[...,128:],args['audio_pad_mask']).mean()
        return (.5*t+.5*r+.1*ce)*weight,torch.stack([t.detach(),r.detach(),ce.detach(),acc.detach()]),tokens
