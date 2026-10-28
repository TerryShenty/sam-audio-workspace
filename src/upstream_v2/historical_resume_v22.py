"""Isolated complete-state resume contract; GPU acceptance requires integration.

Hashing and source validation belong to CPU preflight. The runtime gate checks
that the audited file did not change, restores runtime tensors exactly, owns
and advances the saved cohort cursor, and saves effective continuation state.
It does not assert quality equivalence or bitwise GPU arithmetic.
"""
import copy
import hashlib
import math
import os
import time
from pathlib import Path

SECTIONS = {'model', 'optimizer', 'scheduler', 'ema', 'rng', 'data_position', 'identity'}
RUNTIME = ('model', 'optimizer', 'scheduler', 'ema', 'rng')


def validate_checkpoint(state, expected_identity, current_identity, source_job, current_job,
                        cohort_pairs=32):
    if set(state) != SECTIONS:
        raise ValueError('Incomplete or unexpected checkpoint sections')
    if state['identity'] != expected_identity:
        raise ValueError('Checkpoint identity differs from audited identity')
    if str(source_job) == str(current_job) or not str(source_job) or not str(current_job):
        raise ValueError('Independent allocation identity required')
    ident = state['identity']
    for key in ('encoder', 'cohort', 'seed', 'global_batch'):
        if key not in ident or ident[key] != current_identity.get(key):
            raise ValueError('Incompatible identity: ' + key)
    old = ident.get('seal', {})
    new = current_identity.get('seal', {})
    if not old or any(new.get(k) != v for k, v in old.items()):
        raise ValueError('Historical sealed sources changed or missing')
    if not state['model'] or set(state['model']) != set(state['ema']):
        raise ValueError('Model/EMA keys differ')
    opt = state['optimizer']
    if not opt.get('state') or not opt.get('param_groups'):
        raise ValueError('Missing initialized optimizer')
    ids = [p for group in opt['param_groups'] for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != set(opt['state']) or len(ids) != len(state['model']):
        raise ValueError('Optimizer parameter coverage mismatch')
    if set(state['rng']) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('Incomplete RNG state')
    if not ident.get('effective_config') or ident['optimizer_updates'] != state['scheduler']['last_epoch']:
        raise ValueError('Missing effective configuration or scheduler identity mismatch')
    pos = state['data_position']
    if set(pos) != {'cohort_pair_index', 'cohort_epochs_since_anchor', 'samples_consumed_since_anchor'}:
        raise ValueError('Unsupported data position')
    if any(type(v) is not int or v < 0 for v in pos.values()):
        raise ValueError('Invalid data position values')
    batch = ident['global_batch']
    if batch != 2 or cohort_pairs != 32 or pos['cohort_pair_index'] >= cohort_pairs:
        raise ValueError('Benchmark cohort/batch contract changed')
    if pos['samples_consumed_since_anchor'] != batch * (cohort_pairs * pos['cohort_epochs_since_anchor'] + pos['cohort_pair_index']):
        raise ValueError('Inconsistent data position')


def advance_position(position, updates, cohort_pairs=32, global_batch=2):
    if type(updates) is not int or updates < 0:
        raise ValueError('Invalid update count')
    result = copy.deepcopy(position)
    total = position['cohort_pair_index'] + updates
    result['cohort_pair_index'] = total % cohort_pairs
    result['cohort_epochs_since_anchor'] += total // cohort_pairs
    result['samples_consumed_since_anchor'] += updates * global_batch
    return result


def cpu_source_manifest(path, state, source_job, source_pid, audit_sha256):
    """Call only on the CPU/data-transfer host, never inside a GPU allocation."""
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only source audit required')
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != audit_sha256:
        raise ValueError('Source checksum differs from final CPU audit')
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) != (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev):
        raise ValueError('Source changed while auditing')
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=digest.hexdigest(), source_job=str(source_job), source_pid=int(source_pid),
                identity=copy.deepcopy(state['identity']), cpu_audit_only=True,
                independent_gpu_restore_tested=False)


def load_audited_source(manifest):
    """Load once; fail closed if metadata changed since CPU checksum preflight."""
    import torch
    path = Path(manifest['path'])
    before = path.stat()
    if before.st_size != manifest['bytes'] or before.st_mtime_ns != manifest['mtime_ns']:
        raise ValueError('Checkpoint changed since CPU preflight')
    state = torch.load(path, map_location='cpu', weights_only=False)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) != (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev):
        raise ValueError('Checkpoint changed while loading')
    return state


def effective_resume(state, manifest, current_identity, current_job, restore, snapshot,
                     step, pairs, save, tick, *, updates=32, device_kind='cuda'):
    """Callbacks use real full-model operations; this helper owns cursor/identity.

    restore/snapshot cover RUNTIME; the saved source identity is independently
    validated and data_position selects each actual next pair. A caller must
    restore its benchmark anchor after this gate, preserving comparison inputs.
    """
    import torch
    from resume_audit_v6 import compare
    started = time.monotonic()
    if device_kind not in ('cpu', 'cuda') or (device_kind == 'cuda' and not torch.cuda.is_initialized()):
        raise ValueError('Requested runtime device is not initialized')
    if device_kind == 'cuda' and os.environ.get('SLURM_JOB_ID') != str(current_job):
        raise ValueError('Current GPU allocation identity mismatch')
    if type(updates) is not int or not 1 <= updates <= 32 or len(pairs) != 32:
        raise ValueError('Invalid bounded resume window')
    validate_checkpoint(state, manifest['identity'], current_identity,
                        manifest['source_job'], current_job, len(pairs))
    tick('historical_resume_exact_restore', 0)
    restore(state)
    restored = snapshot()
    # Only tensor/runtime sections belong to restore; identity/cursor are owned
    # here and validated above, never silently compared to a new benchmark ID.
    exact = compare({k: state[k] for k in RUNTIME}, {k: restored[k] for k in RUNTIME})
    del restored
    if not exact['exact']:
        raise ValueError('Incomplete runtime restoration: ' + repr(exact))
    position = copy.deepcopy(state['data_position'])
    selected_indices = []
    values = []
    for n in range(updates):
        index = position['cohort_pair_index']
        tick('train_historical_resume', n)
        loss, norm, _ = step(pairs[index])
        loss, norm = float(loss), float(norm)
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise ValueError('Nonfinite effective continuation')
        values.append(dict(loss=loss, norm=norm))
        selected_indices.append(index)
        position = advance_position(position, 1)
    tick('historical_resume_save', updates)
    post = snapshot()
    if post['scheduler']['last_epoch'] != state['scheduler']['last_epoch'] + updates:
        raise ValueError('Scheduler did not advance with optimizer updates')
    changed = {key: not compare(state[key], post[key])['exact'] for key in ('model', 'optimizer', 'ema')}
    if not all(changed.values()):
        raise ValueError('No effective model/optimizer/EMA continuation: ' + repr(changed))
    post['data_position'] = position
    post['identity'] = copy.deepcopy(current_identity)
    post['identity'].update(optimizer_updates=post['scheduler']['last_epoch'],
        effective_config=copy.deepcopy(state['identity']['effective_config']),
        historical_checkpoint_restored=True,
        resumed_from=dict(job=manifest['source_job'], sha256=manifest['sha256']),
        purpose='independent checkpoint continuation gate; not production adoption')
    save(post)
    return dict(status='PASS_EXACT_RESTORE_AND_EFFECTIVE_CONTINUATION', device=device_kind,
        current_job=str(current_job), current_pid=os.getpid(), source_job=manifest['source_job'],
        source_pid=manifest['source_pid'], source_sha256=manifest['sha256'],
        runtime_restoration_exact=True, source_identity_validated=True,
        source_seal_is_unchanged_subset=True, metadata_migration_explicit=True,
        selected_pair_indices=selected_indices, updates=updates, values=values,
        data_position_before=state['data_position'], data_position_after=position,
        changed=changed, elapsed_seconds=time.monotonic()-started,
        historical_checkpoint_restored=True, independent_allocation=True,
        independent_gpu_process_resume_tested=(device_kind == 'cuda'),
        bitwise_gpu_continuation_established=False, quality_equivalence_established=False,
        production_adopted=False, timing_is_resume_validation_not_training_throughput=True)
