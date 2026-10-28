"""Opt-in same-mixture DAC contrastive auxiliary; not the paper's AED loss.

Pairs must be explicitly reviewed for each exact prompt. No negative mining from
validation, residual decomposition, waveform decoding, or cache regeneration.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ReferringAux(nn.Module):
    def __init__(self, hidden_dim, *, layer=5, weight=0.1, temperature=0.07, head_dim=2048):
        super().__init__()
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('referring weight must be finite and positive')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('referring temperature must be finite and positive')
        self.layer, self.weight, self.temperature = int(layer), weight, temperature
        self.head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.GELU(),
                                  nn.LayerNorm(head_dim), nn.Linear(head_dim, head_dim),
                                  nn.GELU(), nn.LayerNorm(head_dim), nn.Linear(head_dim, 128))

    def forward(self, hidden, positive, negatives, negative_valid, frame_valid):
        prediction = self.head(hidden)
        return self.weight * contrastive_per_example(
            prediction, positive, negatives, negative_valid, frame_valid, self.temperature)


def contrastive_per_example(prediction, positive, negatives, negative_valid, frame_valid, temperature):
    """[B] InfoNCE, positive index 0; zero for examples without negatives.

    Average over ALL real training examples using the existing DDP reduction,
    not rank-local eligible counts. Teachers are detached; padding never scores.
    """
    if prediction.shape != positive.shape or negatives.shape[0] != prediction.shape[0] or negatives.shape[2:] != prediction.shape[1:]:
        raise ValueError('Contrastive tensors must share exact latent timeline/features')
    if negative_valid.shape != negatives.shape[:2] or frame_valid.shape != prediction.shape[:2]:
        raise ValueError('Invalid contrastive masks')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid temperature')
    p = F.normalize(prediction.float(), dim=-1)
    a = F.normalize(positive.detach().float(), dim=-1)
    n = F.normalize(negatives.detach().float(), dim=-1)
    frame = frame_valid.to(p.dtype)
    denom = frame.sum(1).clamp_min(1)
    pos = ((p * a).sum(-1) * frame).sum(1) / denom
    neg = (torch.einsum('btf,bktf->bkt', p, n) * frame[:, None]).sum(-1) / denom[:, None]
    logits = torch.cat([pos[:, None], neg.masked_fill(~negative_valid.bool(), -torch.inf)], 1) / temperature
    loss = F.cross_entropy(logits, torch.zeros(p.shape[0], dtype=torch.long, device=p.device), reduction='none')
    return loss * (negative_valid.any(1) & frame_valid.any(1)).to(loss.dtype)


class ReviewedPairs:
    """Resolve explicitly approved B/C IDs only within the training catalog."""
    def __init__(self, path, entries):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        by_id = {e['sample_id']: e for e in entries}
        if len(by_id) != len(entries):
            raise ValueError('Duplicate training sample IDs')
        self.pairs = {}
        for pair in json.loads(raw)['pairs']:
            sid, prompt = pair['sample_id'], pair['prompt']
            entry = by_id[sid]  # missing/validation-only IDs fail closed
            refs = next((v for v in entry['variants'] if v['prompt'] == prompt), None)
            if refs is None:
                raise ValueError('Reviewed prompt is absent from training catalog')
            key = (sid, prompt)
            if key in self.pairs:
                raise ValueError('Duplicate reviewed prompt')
            ids = pair['negative_sample_ids']
            if not ids or len(ids) != len(set(ids)):
                raise ValueError('Negative IDs must be nonempty and unique')
            clean_keys = []
            for other in ids:
                other_refs = by_id[other]['variants'][0]
                if (other == sid or other.rsplit('::', 1)[0] != sid.rsplit('::', 1)[0]
                    or any(other_refs[k] != refs[k] for k in ('mixture', 'video'))
                    or other_refs['clean'] == refs['clean']):
                    raise ValueError('Negative must be a distinct GT in the same training clip/mixture/video')
                if other_refs['clean'] in clean_keys:
                    raise ValueError('Duplicate negative GT cache key')
                clean_keys.append(other_refs['clean'])
            self.pairs[key] = clean_keys
        if not self.pairs:
            raise ValueError('No reviewed contrastive pairs')

    def attach(self, value, sample_id, refs, store):
        value = dict(value)  # never mutate store/LRU values
        negatives = []
        for key in self.pairs.get((sample_id, refs['prompt']), []):
            clean = store.get('clean', key)
            if clean.shape[:2] != value['clean'].shape[:2] or clean.shape[-1] not in (128, 256):
                raise ValueError('Negative cache has different timeline or codec dimensions')
            if not torch.isfinite(clean).all():
                raise ValueError('Nonfinite negative cache')
            negatives.append(clean[0, :, :128].detach())
        value['referring_negatives'] = negatives
        return value


def collate_negatives(items, frames):
    k = max(1, max(len(i['referring_negatives']) for i in items))
    exemplar = items[0]['clean']
    negatives = exemplar.new_zeros((len(items), k, frames, 128))
    valid = torch.zeros(len(items), k, dtype=torch.bool, device=exemplar.device)
    for b, item in enumerate(items):
        for j, tensor in enumerate(item['referring_negatives']):
            negatives[b, j, :tensor.shape[0]] = tensor
            valid[b, j] = True
    return negatives, valid
