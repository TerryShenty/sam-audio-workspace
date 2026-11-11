"""CPU checks for the ablation layer (no Qwen/SAM weights, no GPU).

Covers: A/B pair sampler, the four arm objectives on a tiny stand-in SAM, the
scorer's -30 dB floored mean and division-by-zero handling, the data-contract
builder and its fail-closed loader. Not evidence of GPU memory, speed or quality.
Run: python -m pytest -q tests/test_ablation_cpu.py
"""
import importlib, json, math, random, sys, types, wave
from pathlib import Path

import numpy as np
import pytest
import torch

P = Path(__file__).resolve().parent.parent
V = P / 'src'
RUNTIME = V / 'upstream_v2' / 'runtime_v1'
for p in (V, RUNTIME, RUNTIME / 'perf', RUNTIME / 'code'):
    sys.path.insert(0, str(p))
# Import sam_audio sub-modules without executing the package __init__ (which needs model deps).
if 'sam_audio' not in sys.modules:
    pkg = types.ModuleType('sam_audio'); pkg.__path__ = [str(RUNTIME / 'code' / 'sam_audio')]; sys.modules['sam_audio'] = pkg
    sub = types.ModuleType('sam_audio.native_finetuning'); sub.__path__ = [str(RUNTIME / 'code' / 'sam_audio' / 'native_finetuning')]
    sys.modules['sam_audio.native_finetuning'] = sub

from ablation_sampler import PairSampler  # noqa: E402


def pair_rows(n):
    rows = []
    for i in range(n):
        for role in ('A', 'B'):
            rows.append(dict(sample_id=f'm{i:04d}::{role}', pair_id=f'm{i:04d}', role=role))
    random.Random(0).shuffle(rows)
    return rows


@pytest.mark.parametrize('world', [1, 2, 3, 4, 8])
def test_pair_sampler_covers_each_pair_once_per_epoch(world):
    rows = pair_rows(24)
    samplers = [PairSampler(rows, rank=r, seed=7, data_identity='d', world_size=world) for r in range(world)]
    seen = []
    while samplers[0].cursor < 2 * len(rows):  # two epochs
        picks = [s.peek() for s in samplers]
        for a, b in picks:
            assert rows[a]['pair_id'] == rows[b]['pair_id'] and (rows[a]['role'], rows[b]['role']) == ('A', 'B')
            seen.append(rows[a]['pair_id'])
        for s in samplers:
            s.commit()
    first = seen[:24]
    assert sorted(first) == sorted({r['pair_id'] for r in rows})
    assert samplers[0].completed_epochs >= 2


def test_pair_sampler_state_roundtrip_and_identity():
    rows = pair_rows(10)
    s = PairSampler(rows, rank=1, seed=3, data_identity='d', world_size=2)
    for _ in range(7):
        s.peek(); s.commit()
    state = s.state_dict(); expected = s.peek(); s.commit()
    t = PairSampler(rows, rank=1, seed=3, data_identity='d', world_size=2); t.load_state_dict(state)
    assert t.peek() == expected
    other = PairSampler(rows, rank=1, seed=4, data_identity='d', world_size=2)
    with pytest.raises(AssertionError):
        other.load_state_dict(state)


def test_pair_sampler_rejects_missing_partner():
    rows = pair_rows(3)[:-1]
    with pytest.raises(AssertionError):
        PairSampler(rows, rank=0, seed=1, data_identity='d', world_size=1)


# ----------------------------------------------------------------- objectives
class TinySAM(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.lin = torch.nn.Linear(256 + 1 + 4, 256); self.calls = []

    def forward(self, noisy_audio, time, text_features, text_mask, **kw):
        self.calls.append(dict(noisy=noisy_audio.detach().clone(), time=time.detach().clone()))
        B, T, _ = noisy_audio.shape
        x = torch.cat([noisy_audio, time[:, None, None].expand(B, T, 1), text_features.expand(B, T, 4)], -1)
        return self.lin(x)


def make_joint(arm, monkeypatch, weight=1.0):
    import ablation_joint, sam_batch_round23
    cond = torch.nn.Parameter(torch.randn(2, 1, 4))
    def fake_condition(joint, items):
        return cond, torch.tensor([.3, .4], requires_grad=True), torch.tensor([1., 1.]), [10, 10], None
    monkeypatch.setattr(ablation_joint, 'condition_batch', fake_condition)
    monkeypatch.setattr(sam_batch_round23, 'condition_batch', fake_condition)
    j = ablation_joint.AblationJoint.__new__(ablation_joint.AblationJoint)
    torch.nn.Module.__init__(j)
    j.sam = TinySAM(); j.cond = cond
    return j.configure_arm(arm, weight)


def pair_inputs(T=6):
    args = [dict(audio_features=torch.zeros(1, T, 256), audio_pad_mask=torch.ones(1, T, dtype=torch.bool)) for _ in range(2)]
    clean = [torch.randn(1, T, 256), torch.randn(1, T, 256)]
    items = [dict(pair_id='m1', role='A'), dict(pair_id='m1', role='B')]
    return items, args, clean


def manual(pred, vel, mask):
    from sam_audio.native_finetuning.benchmark_utils import per_target_mse
    return per_target_mse(pred[..., :128], vel[..., :128], mask), per_target_mse(pred[..., 128:], vel[..., 128:], mask)


@pytest.mark.parametrize('arm', ['baseline', 'dual_t0', 'contrastive', 'dual_t0_contrastive'])
def test_arm_losses_match_definition(arm, monkeypatch):
    from ablation_joint import ARMS
    torch.manual_seed(0)
    j = make_joint(arm, monkeypatch, weight=2.0)
    items, args, clean = pair_inputs()
    torch.manual_seed(1)
    loss, terms, _ = j(items, args, clean, [.5, .5])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in j.sam.parameters())
    dual, contrast = ARMS[arm]
    calls = j.sam.calls
    assert len(calls) == (1 if arm == 'baseline' else 2)
    ce = torch.tensor([.3, .4])
    with torch.no_grad():
        mask = torch.ones(2, 6, dtype=torch.bool)
        c = torch.cat(clean, 0)
        def velocity(call):
            # Recover the noise from x_t: noise = (x_t - t*c) / (1 - (1-s) t)
            t = call['time'][:, None, None]; s = 1e-5
            noise = (call['noisy'] - t * c) / (1 - (1 - s) * t)
            return c - (1 - s) * noise
        pred_r = j.sam.lin(torch.cat([calls[0]['noisy'], calls[0]['time'][:, None, None].expand(2, 6, 1), j.cond.expand(2, 6, 4)], -1))
        t, r = manual(pred_r, velocity(calls[0]), mask)
        extra = torch.zeros(())
        if arm != 'baseline':
            assert torch.equal(calls[1]['time'], torch.zeros(2))
            pred0 = j.sam.lin(torch.cat([calls[1]['noisy'], calls[1]['time'][:, None, None].expand(2, 6, 1), j.cond.expand(2, 6, 4)], -1))
            v0 = velocity(calls[1])
            assert torch.equal(calls[1]['noisy'][0], calls[1]['noisy'][1]), 't=0 inputs of A and B must be identical in every non-baseline arm'
            if contrast:
                from sam_audio.native_finetuning.benchmark_utils import per_target_mse
                extra = 2.0 * per_target_mse(pred0[0:1] - pred0[1:2], v0[0:1] - v0[1:2], mask[0:1]).mean()
            if dual:
                t0, r0 = manual(pred0, v0, mask); t, r = .5 * (t + t0), .5 * (r + r0)
        expected = ((.5 * t + .5 * r + .1 * ce) * .5).sum() + extra
    assert torch.allclose(loss.detach(), expected, rtol=1e-5, atol=1e-6), (float(loss), float(expected))


def test_contrastive_requires_one_mixture(monkeypatch):
    j = make_joint('contrastive', monkeypatch)
    items, args, clean = pair_inputs(); items[1]['pair_id'] = 'm2'
    with pytest.raises(AssertionError):
        j(items, args, clean, [.5, .5])


def test_contrastive_term_is_zero_for_identical_targets_with_perfect_model(monkeypatch):
    from ablation_joint import pair_difference_loss
    v = torch.randn(1, 5, 256); pred = torch.cat([v, v]); vel = torch.cat([v, v])
    assert float(pair_difference_loss(pred, vel, torch.ones(2, 5, dtype=torch.bool))) == 0.0


# ------------------------------------------------------------------- scoring
def test_scorer_floor_no_clip_and_zero_prediction():
    import referring_metrics_v2 as m
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(48000); mix = ref + rng.standard_normal(48000)
    zero = m.score_triplet(ref, np.zeros(48000), mix)
    assert zero['status'] == m.STATUS_PRED_SILENT and zero['aligned_si_sdr_db'] is None  # no eps -> 0 dB
    noise = rng.standard_normal(48000); noise -= noise.dot(ref - ref.mean()) / (ref - ref.mean()).dot(ref - ref.mean()) * (ref - ref.mean())
    value, status = m.si_sdr(ref, noise + 1e-6 * ref)
    assert status == m.STATUS_OK and value < -100  # v1 would have clipped this to -100
    bad = m.score_triplet(ref, rng.standard_normal(48000), mix)
    assert bad['aligned_si_sdr_db'] < -30
    good = m.score_triplet(ref, ref + .1 * rng.standard_normal(48000), mix)
    quiet = m.score_triplet(ref, 1e-5 * (ref + .1 * rng.standard_normal(48000)), mix)
    assert quiet['silent_failure'] and math.isfinite(quiet['aligned_si_sdr_db'])  # quiet scored normally
    same = m.score_triplet(ref, ref + .1 * rng.standard_normal(48000), ref.copy())
    assert same['si_sdri_db'] is None and same['si_sdri_status'] == 'sdri_undefined_reference_equals_mixture'
    s = m.summarize([zero, bad, good])
    expected = (-30 + -30 + good['aligned_si_sdr_db']) / 3
    assert math.isclose(s['aligned_si_sdr_db_mean_floored'], expected)
    assert math.isclose(s['aligned_si_sdr_db_floor_rate'], 2 / 3)
    assert math.isclose(s['aligned_si_sdr_db_mean_finite'], (bad['aligned_si_sdr_db'] + good['aligned_si_sdr_db']) / 2)


# -------------------------------------------------------------- data contract
def write_wav(path, x, sr=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(path), x.astype(np.float32), sr)


def test_prepare_contract_and_fail_closed_loader(tmp_path, monkeypatch):
    sys.path.insert(0, str(P / 'scripts'))
    prep = importlib.import_module('ablation_prepare')
    import ablation_data
    monkeypatch.setattr(ablation_data, 'CHECK_VIDEOS', 2); monkeypatch.setattr(prep, 'CHECK_VIDEOS', 2)
    animal = tmp_path / 'animal'; rng = np.random.default_rng(1); queries = []
    for i in range(3000):
        vid = f'animal_mix_{i:04d}'; folder = animal / 'media' / vid
        if i < 3:  # only a few real files; the rest share them through symlinks to stay small
            a, b = rng.standard_normal(4800) * .1, rng.standard_normal(4800) * .1
            write_wav(folder / 'target_A.wav', a); write_wav(folder / 'target_B.wav', b); write_wav(folder / 'mixture.wav', a + b)
            (folder / 'video.mp4').write_bytes(b'fake-video-%d' % i)
        else:
            folder.mkdir(parents=True)
            src = animal / 'media' / f'animal_mix_{i % 3:04d}'
            for n in ('target_A.wav', 'target_B.wav', 'mixture.wav', 'video.mp4'):
                (folder / n).symlink_to(src / n)
        for role in ('A', 'B'):
            queries.append(dict(video_id=vid, role=role, prompt=f'the {role} animal {i}', mode='spatial'))
    (animal / 'queries.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in queries))
    bench = tmp_path / 'bench'; (bench / 'metadata').mkdir(parents=True)
    write_wav(bench / 'data/gs/c1/gt/target_01.wav', rng.standard_normal(4800) * .1)
    (bench / 'data/gs/c1/video.mp4').write_bytes(b'gs-video')
    (bench / 'metadata/prompts.csv').write_text('dataset,clip_id,video_path,target_source_id,gt_audio_path,prompt,task_type\n'
                                               'general_sound,c1,data/gs/c1/video.mp4,1,data/gs/c1/gt/target_01.wav,the dog,Spatial\n'
                                               'urmp,u1,data/u/video.mp4,1,data/u/gt.wav,the violin,Spatial\n')
    monkeypatch.setattr(prep, 'extract_audio', lambda mp4, wav: write_wav(wav, rng.standard_normal(4800) * .1))
    out = tmp_path / 'data'
    monkeypatch.setattr(sys, 'argv', ['x', '--animal', str(animal), '--benchmark', str(bench), '--benchmark-revision', 'test', '--out', str(out)])
    prep.main()
    contract = json.loads((out / 'DATA_CONTRACT.json').read_text())
    assert contract['train_rows'] == 6000 and contract['check_rows'] == 4 and contract['gs_rows'] == 1
    rows = json.loads((out / 'ROWS.json').read_text())
    train_ids = {r['sample_id'] for r in rows['train']}
    assert all(r['sample_id'].removesuffix('::check') in train_ids for r in rows['check'])  # check set is trained on
    assert all(r['split'] == 'gs_benchmark_validation' and r['dataset'] == 'general_sound' for r in rows['gs_benchmark'])
    a0 = next(r for r in rows['train'] if r['sample_id'].endswith('0000::A')); b0 = next(r for r in rows['train'] if r['sample_id'].endswith('0000::B'))
    assert a0['refs']['mixture'] == b0['refs']['mixture'] and a0['refs']['clean'] != b0['refs']['clean']
    plan = json.loads((out / 'SAM_CACHE_PLAN.json').read_text())['mixtures']
    gs_targets = [t for spec in plan.values() for t in spec['targets'].values() if t['residual'] is None]
    assert len(gs_targets) == 1  # GS residual = mixture - target, only for shape (not scored)
    with pytest.raises(AssertionError):
        ablation_data.load(out)  # caches not built yet -> fail closed


# ------------------------------------------------------------- epoch budget loop
def test_run_loop_validates_each_epoch_and_stops_at_budget(monkeypatch, tmp_path):
    import production_loop, ablation_train
    monkeypatch.setattr(production_loop, 'collective_decision', lambda proposed: proposed)
    rows = pair_rows(6)
    class Trainer:
        def __init__(self): self.sampler = PairSampler(rows, rank=0, seed=1, data_identity='d', world_size=4); self.step_number = 0
    class Session:
        def __init__(self): self.trainer = Trainer(); self.saved = []
        def step(self):
            self.trainer.sampler.peek(); self.trainer.sampler.commit(); self.trainer.step_number += 1; return (0., 0.)
        def checkpoint(self, path): self.saved.append(self.trainer.step_number)
    session = Session(); epochs_seen = []
    now = [1000.]
    def clock():
        now[0] += 1; return now[0]
    result = ablation_train.run_loop(session, tmp_path, deadline_epoch=10**9, epochs=3, on_step=lambda r: None,
                                     on_epoch=lambda e, p: epochs_seen.append((e, session.trainer.sampler.completed_epochs)), clock=clock)
    assert result['reason'] == 'epoch_budget_complete' and result['completed_epochs'] == 3
    assert [e for e, _ in epochs_seen] == [1, 2, 3]  # 6 pairs, 4 per update: epoch boundaries cross updates, none skipped
    assert session.saved[-1] == session.trainer.step_number


# ------------------------------------------------------------ validation batching
def test_validation_oom_split_keeps_every_row_in_order():
    from ablation_validation import generate_with_fallback
    class OOM(Exception):
        pass
    calls = []
    def guarded(entries):
        calls.append(len(entries))
        if len(entries) > 2:
            raise OOM()
        return [dict(sample_id=e['item']['sample_id']) for e in entries]
    rows = [dict(sample_id=f's{i}', protocol={'noise_seed': i}) for i in range(8)]
    fallbacks = []
    out = generate_with_fallback(guarded, rows, fallbacks, rank=0, on_attempt=lambda: None, oom=OOM, empty_cache=lambda: None)
    assert [o['sample_id'] for o in out] == [r['sample_id'] for r in rows]
    assert [f['size'] for f in fallbacks] == [8, 4, 4] and calls == [8, 4, 2, 2, 4, 2, 2]
    with pytest.raises(OOM):
        generate_with_fallback(lambda e: (_ for _ in ()).throw(OOM()), rows[:1], [], rank=0, on_attempt=lambda: None, oom=OOM, empty_cache=lambda: None)
