"""Eight exact ret25 hypotheses; old retention scope and memory gates remain intact."""
import copy
import pointwise_identity_v50 as ident
ROOT={'name': 'session_incumbent', 'singleton': True, 'cache': 'pinned', 'retained': 24, 'waves': 4, 'group': 1, 'static': True, 'native_sam_mode': 'native_rms', 'compiler_mode': 'both_dynamic_gemm', 'memo_policy': 'all', 'memo_roles': ['gate_proj', 'up_proj', 'down_proj'], 'memo_bytes': 3221225472, 'memory_tier': 'expanded92', 'nf4_row_reuse_direction': 'both', 'nf4_memo_consume_once': True, 'backward_kernel': {'bn': 128, 'bk': 32, 'warps': 4, 'stages': 1, 'waves': 8, 'group': 1}, 'forward_kernel': {'bn': 64, 'bk': 32, 'warps': 4, 'stages': 1, 'waves': 8, 'group': 1}, 'nf4_row_reuse_waves': {'gate_proj': {'forward_kernel': 32, 'backward_kernel': 32}, 'up_proj': {'forward_kernel': 32, 'backward_kernel': 32}, 'down_proj': {'forward_kernel': 32, 'backward_kernel': 32}}, 'nf4_same_expert_fused64': {'gate_proj': {'backward_kernel': True}, 'up_proj': {'backward_kernel': True}, 'down_proj': {'backward_kernel': True}}, 'nf4_row_scale_cache': {'precision': 'bf16', 'roles': ['gate_proj', 'up_proj', 'down_proj'], 'direction': 'forward', 'max_bytes': 2147483648}}
def proposal_pool():
 out=[]
 for block in (512,1024):
  for label,roles,front in (('all',list(ident.ROLES),False),('gateup',['gate_proj','up_proj'],False),('down',['down_proj'],False),('frontfused',list(ident.ROLES),True)):
   c=copy.deepcopy(ROOT);c.update(retained=25,memo_bytes=3087007744)
   c[ident.POINTWISE]=dict(mode='recompute',block=block);c[ident.SCALE]['roles']=roles
   if front:
    for role in ident.ROLES:c[ident.FUSION][role]['forward_kernel']=True
   c['name']='retention_v51_ret25_'+label+'_b'+str(block);out.append(ident.normalize(c))
 return out
ALLOWED=frozenset(ident.computation_identity(c) for c in proposal_pool())
def allows_ret25(c):return ident.computation_identity(c) in ALLOWED
