"""Explicit process policy and observation-only replay diagnostics.

Candidate only: CPU validation cannot establish CUDA determinism or throughput.
Install once before CUDA initialization, model construction or torch.compile.
Never silently retry a failed replay or relax its historical acceptance limits.
"""
import hashlib
import math
import os
import torch

POLICY='strict_process_replay_v1'
WORKSPACE=':4096:8'
GROUPS=('qwen','bridge','sam_lora','sam_conditions')


def descriptor():
    cuda=torch.backends.cuda
    return dict(schema=1,policy=POLICY,torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        sdp_flash=cuda.flash_sdp_enabled(),sdp_efficient=cuda.mem_efficient_sdp_enabled(),
        sdp_math=cuda.math_sdp_enabled(),sdp_cudnn=cuda.cudnn_sdp_enabled(),
        matmul_tf32=cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)


def assert_policy():
    d=descriptor()
    expected=dict(deterministic=True,warn_only=False,cublas_workspace=WORKSPACE,
        cudnn_deterministic=True,cudnn_benchmark=False,sdp_flash=True,
        sdp_efficient=True,sdp_math=True,sdp_cudnn=False)
    if any(d[k]!=v for k,v in expected.items()):raise RuntimeError('REPLAY_POLICY_CHANGED')
    return d


def install_before_cuda():
    if torch.cuda.is_initialized():raise RuntimeError('REPLAY_POLICY_MUST_PRECEDE_CUDA')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=WORKSPACE:
        raise RuntimeError('SET_CUBLAS_WORKSPACE_IN_LAUNCH_ENVIRONMENT')
    # Strict (not warn-only) controls the supported fused attention backward paths.
    # Custom kernels still require their independent numerical/replay GPU gates.
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    return assert_policy()


def stamp_state(state):
    identity=state.setdefault('identity',{})
    d=assert_policy()
    if 'replay_execution_descriptor' in identity and identity['replay_execution_descriptor']!=d:
        raise ValueError('REPLAY_DESCRIPTOR_CONFLICT')
    identity['replay_execution_descriptor']=d
    return state


def validate_state(state):
    if state.get('identity',{}).get('replay_execution_descriptor')!=assert_policy():
        raise ValueError('MISSING_OR_CHANGED_REPLAY_DESCRIPTOR')
    # No implicit migration of old checkpoints. Caller needs an audited source path.


def capture_gradients(params,grouped):
    if set(grouped)!=set(GROUPS):raise ValueError('FOUR_GRADIENT_GROUPS_REQUIRED')
    names=[n for v in grouped.values() for n in v]
    if len(names)!=len(set(names)) or set(names)!=set(params):raise ValueError('GROUP_PARTITION_MISMATCH')
    tensors={};groups={}
    for group in GROUPS:
        squares=0.;records={};missing=[]
        for name in grouped[group]:
            p=params[name]
            if p.grad is None:missing.append(name);continue
            # Copy only in explicitly untimed diagnostics; never retain a CUDA graph.
            value=p.grad.detach().cpu().clone().contiguous()
            if not bool(torch.isfinite(value).all()):raise FloatingPointError(name)
            tensors[name]=value
            squared=float(value.double().square().sum());squares+=squared
            records[name]=dict(shape=list(value.shape),dtype=str(value.dtype),squared_l2=squared,
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if not records:raise ValueError('INACTIVE_GRADIENT_GROUP:'+group)
        groups[group]=dict(norm=math.sqrt(squares),tensors=records,missing=missing)
    return tensors,dict(groups=groups,pre_clip=True,diagnostic_only=True,
        total_norm_fp64=math.sqrt(sum(v['norm']**2 for v in groups.values())))


def gradient_delta(reference,candidate,grouped):
    if reference.keys()!=candidate.keys():raise ValueError('GRADIENT_KEYS_CHANGED')
    result={}
    for group,names in grouped.items():
        aa=dd=0.;changed=[]
        for name in names:
            if name not in reference:continue
            a,b=reference[name],candidate[name]
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('GRADIENT_STRUCTURE_CHANGED')
            a=a.double();b=b.double();den=float(a.square().sum());diff=float((a-b).square().sum())
            aa+=den;dd+=diff
            if diff:changed.append(dict(name=name,absolute_l2=math.sqrt(diff),relative_l2=math.sqrt(diff/max(den,1e-300))))
        result[group]=dict(relative_l2=math.sqrt(dd/max(aa,1e-300)),
            absolute_l2=math.sqrt(dd),changed_tensors=len(changed),
            largest_absolute=sorted(changed,key=lambda x:x['absolute_l2'],reverse=True)[:8])
    return dict(groups=result,observation_only=True,changes_acceptance_thresholds=False)


def replay_decision(replay,comparisons):
    from resume_audit_v6 import acceptable
    if len(replay)!=3 or set(comparisons)!={'memory_disk','memory_repeat','disk_repeat'}:
        raise ValueError('INCOMPLETE_REPLAY')
    if [v['label'] for v in replay]!=['memory_A','disk_B','memory_C']:
        raise ValueError('REPLAY_ORDER_CHANGED')
    ref=replay[0];failures=[]
    for row in replay:
        for key,threshold in (('loss',1e-6),('norm',1e-4)):
            a,b=float(ref[key]),float(row[key])
            if not math.isfinite(a) or not math.isfinite(b):failures.append(dict(label=row['label'],metric=key,reason='nonfinite'));continue
            rel=abs(b-a)/max(abs(a),1e-12)
            if rel>threshold:failures.append(dict(label=row['label'],metric=key,relative=rel,threshold=threshold))
    for key,value in comparisons.items():
        if not acceptable(value):failures.append(dict(comparison=key,metric='post_update_state_groups'))
    return dict(effective_continuation_pass=not failures,failures=failures,
        original_thresholds_preserved=True,automatic_retry=False)
