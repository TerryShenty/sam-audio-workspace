"""User-authorized speed-first arithmetic; finite training and exact I/O remain required."""
import os
import torch
from replay_determinism_candidate_v1 import descriptor as observed_descriptor

POLICY='speed_first_finite_training_v29'

def descriptor():
    d=observed_descriptor();d.update(policy=POLICY,arithmetic_distance_is_acceptance_gate=False)
    return d

def assert_policy():
    d=descriptor()
    assert not d['deterministic'] and not d['cudnn_deterministic'] and not d['cudnn_benchmark']
    assert d['cublas_workspace'] is None
    assert d['sdp_flash'] and d['sdp_efficient'] and d['sdp_math'] and d['sdp_cudnn']
    return d

def install_before_cuda():
    assert not torch.cuda.is_initialized()
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_cudnn_sdp(True)
    return assert_policy()

def stamp_state(state):
    d=assert_policy();identity=state.setdefault('identity',{})
    assert identity.get('replay_execution_descriptor',d)==d
    identity['replay_execution_descriptor']=d
    return state

def validate_state(state):
    assert state.get('identity',{}).get('replay_execution_descriptor')==assert_policy(),'EXECUTION_DESCRIPTOR_MISMATCH'
