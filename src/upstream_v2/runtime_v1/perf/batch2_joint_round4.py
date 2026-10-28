"""Staged B2 language forward; not enabled in the running training job.

Keep the global batch at eight (four ranks, two examples each). Frozen AV
preparation follows the existing per-example processor; all language hidden
states are computed together and differentiated on every forward.
"""
import torch
from torch.nn import functional as F
from bridge import Joint


class _PreparedInputCaptured(Exception):
    pass


def condition_batch(joint, items):
    assert len(items) == 2
    vlm = joint.vlm
    qwen = vlm.qwen
    device = vlm.device_
    captures = []
    for item in items:
        current = {}

        def capture_ids(module, args, kwargs):
            current['ids'] = kwargs['input_ids']

        def capture_prepared(module, args, kwargs):
            current['kwargs'] = dict(kwargs)
            raise _PreparedInputCaptured()

        ids_hook = qwen.register_forward_pre_hook(capture_ids, with_kwargs=True)
        model_hook = qwen.model.register_forward_pre_hook(capture_prepared, with_kwargs=True)
        try:
            vlm(item['media_key'], item.get('prompt', item.get('row', {}).get('prompt')))
            raise AssertionError('Expected prepared language inputs')
        except _PreparedInputCaptured:
            current['positions'] = qwen.lm_head.positions.clone()
            current['marker'] = qwen.lm_head.marker_position
        finally:
            model_hook.remove()
            ids_hook.remove()
        captures.append(current)
    lengths = [x['ids'].shape[1] for x in captures]
    maximum = max(lengths)
    kwargs = dict(captures[0]['kwargs'])
    kwargs['inputs_embeds'] = torch.cat([
        F.pad(x['kwargs']['inputs_embeds'], (0,0,0,maximum-n))
        for x,n in zip(captures,lengths)], 0)
    kwargs['attention_mask'] = torch.cat([
        F.pad(torch.ones_like(x['ids']), (0,maximum-n))
        for x,n in zip(captures,lengths)], 0)
    kwargs['position_ids'] = torch.cat([
        F.pad(x['kwargs']['position_ids'], (0,maximum-n))
        for x,n in zip(captures,lengths)], 1)
    if kwargs.get('cache_position') is not None:
        kwargs['cache_position'] = torch.arange(maximum, device=device)
    if kwargs.get('input_ids') is not None:
        kwargs['input_ids'] = None
    if kwargs.get('visual_pos_masks') is not None:
        kwargs['visual_pos_masks'] = torch.cat([
            F.pad(x['kwargs']['visual_pos_masks'], (0,0,0,maximum-n))
            for x,n in zip(captures,lengths)], 0)
        depth = len(captures[0]['kwargs']['deepstack_visual_embeds'])
        kwargs['deepstack_visual_embeds'] = tuple(torch.cat([
            x['kwargs']['deepstack_visual_embeds'][j] for x in captures],0)
            for j in range(depth))
        assert all(v.shape[0] == int(kwargs['visual_pos_masks'][...,0].sum())
                   for v in kwargs['deepstack_visual_embeds'])
    kwargs['use_cache'] = False
    hidden = qwen.model(**kwargs).last_hidden_state.to(device)
    head = qwen.lm_head
    hs, losses, accuracies = [], [], []
    for i,x in enumerate(captures):
        hs.append(hidden[i:i+1,x['marker'],:])
        z = hidden[i:i+1,x['positions'],:]
        logits = head.base(z)
        special = F.linear(z, head.row.to(z.dtype).unsqueeze(0))
        logits = torch.cat([logits[...,:head.token_id], special,
                            logits[...,head.token_id+1:]],-1)
        labels = x['ids'][:,x['positions']+1]
        losses.append(F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]), labels.reshape(-1)))
        j = (x['positions'] == x['marker']).nonzero().item()
        accuracies.append((logits[:,j].argmax(-1) == head.token_id).float().mean())
    h = torch.cat(hs,0)
    if getattr(vlm,'audit_flow_gradient',False) and h.requires_grad:
        def record_gradient(g):
            vlm.flow_hidden_gradient_norm = float(g.detach().float().norm())
        h.register_hook(record_gradient)
    conditions = joint.bridge(h.float()).unsqueeze(1)
    return conditions, torch.stack(losses), torch.stack(accuracies), lengths, h


class Batch2Joint(Joint):
    def forward(self, item, args, clean, weight=1.):
        if not isinstance(item, list):
            return super().forward(item,args,clean,weight)
        assert len(item) == len(args) == len(clean) == len(weight) == 2
        from sam_audio.native_finetuning.objective import make_flow_matching_state
        from sam_audio.native_finetuning.benchmark_utils import per_target_mse
        cond, ce, acc, lengths, _ = condition_batch(self,item)
        loss_terms, metrics = [], []
        sam_device = next(self.sam.parameters()).device
        for i,(a,target,w) in enumerate(zip(args,clean,weight)):
            assert 'text_features' not in a and 'text_mask' not in a
            state = make_flow_matching_state(target, sigma_min=1e-5)
            pred = self.sam(noisy_audio=state.noisy_latent,time=state.time,
                text_features=cond[i:i+1].to(sam_device),
                text_mask=torch.ones((1,1),dtype=torch.bool,device=sam_device),**a)
            t = per_target_mse(pred[...,:128],state.target_velocity[...,:128],a['audio_pad_mask']).mean()
            r = per_target_mse(pred[...,128:],state.target_velocity[...,128:],a['audio_pad_mask']).mean()
            t=t.to(ce.device); r=r.to(ce.device)
            loss_terms.append((.5*t+.5*r+.1*ce[i])*w)
            metrics.append(torch.stack([t.detach(),r.detach(),ce[i].detach(),acc[i].detach()]))
        return torch.stack(loss_terms).sum(), torch.stack(metrics), lengths
