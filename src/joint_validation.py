"""Joint AUDSEG -> SAM midpoint inference; target memory qualification is mandatory."""
import contextlib,hashlib,random
import numpy as np
import torch
from validation_inputs import condition_pairs
from production_data import tree

@contextlib.contextmanager
def evaluation_state(joint,device):
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
         torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
    modes=[(m,m.training) for m in joint.modules()]
    try:
        joint.eval()
        with torch.inference_mode():yield
    finally:
        for m,mode in modes:m.training=mode
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if rng[3] is not None:torch.cuda.set_rng_state(rng[3],device)

def generate(joint,codec,store,part,device,*,conditioner=None,collator=None,solver=None):
    assert not joint.training and not torch.is_grad_enabled()
    assert 1<=len(part)<=8 and all(x['item']['split'].endswith('_validation') for x in part)
    if conditioner is None:
        from batch2_joint_round4 import condition_batch
        conditioner=condition_batch
    if collator is None:
        from sam_audio.native_finetuning.benchmark_utils import collate_features
        collator=collate_features
    if solver is None:
        from torchdiffeq import odeint
        solver=odeint
    cast=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
    conditions=[]
    with cast():
        for pair,real in condition_pairs(part):
            cond,_,_,_,_=conditioner(joint,[x['item'] for x in pair])
            assert cond.shape[0]==2
            conditions.append(cond[:real])
    conditions=torch.cat(conditions,0)
    samples=[];frames=[]
    for x in part:
        args,clean=store.assemble(x['item'])
        assert 'text_features' not in args and 'text_mask' not in args
        assert args['anchor_ids'].tolist()==[[0,3]]
        assert bool((args['anchor_alignment'][args['audio_pad_mask']]==0).all()),'VALIDATION_SPAN_LEAK'
        frames.append(args['audio_features'].shape[1]);samples.append(dict(forward_args=args,clean=clean))
    forward=tree(collator(samples)['forward_args'],lambda t:t.to(device,non_blocking=True))
    forward['text_features']=conditions.to(device)
    forward['text_mask']=torch.ones((len(part),1),device=device,dtype=torch.bool)
    maximum=max(frames);noises=[];hashes=[]
    for x,t in zip(part,frames):
        noise=torch.randn((1,t,256),generator=torch.Generator(device=device).manual_seed(x['protocol']['noise_seed']),device=device,dtype=torch.bfloat16)
        hashes.append(hashlib.sha256(noise.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        noises.append(torch.nn.functional.pad(noise,(0,0,0,maximum-t)))
    noise=torch.cat(noises)
    def field(at,z):
        with cast():v=joint.sam(noisy_audio=z,time=at.float().expand(len(part)),**forward)
        assert v.shape==z.shape and bool(torch.isfinite(v).all())
        return v
    with torch.jit.optimized_execution(False):
        latent=solver(field,noise,torch.tensor([0.,1.],device=device),method='midpoint',options={'step_size':1/16})[-1]
        predictions=[None]*len(part)
        # Decode equal original lengths together; never decode padded latent tails.
        for t in sorted(set(frames)):
            ids=[i for i,n in enumerate(frames) if n==t];z=latent[ids,:t]
            with cast():wav=codec.decode(z.transpose(1,2).reshape(len(ids)*2,128,t)).view(len(ids),2,-1)[:,:1,:]
            sizes=codec.feature_idx_to_wav_idx(torch.full((len(ids),),t))
            for i,pred in zip(ids,joint.sam.unbatch(wav,sizes)):
                predictions[i]=pred[0].float().cpu().numpy()
    assert all(p is not None and np.isfinite(p).all() for p in predictions)
    return [dict(sample_id=x['item']['sample_id'],prediction=p,noise_sha256=h,latent_frames=t)
            for x,p,h,t in zip(part,predictions,hashes,frames)]
