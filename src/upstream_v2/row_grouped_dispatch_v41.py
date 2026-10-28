"""Install below pair dispatch and memo; GPU timings remain unmeasured."""
import torch
from projection_backend_candidate_v25 import packed_roles
from projection_launch_candidate_v25 import select
from row_grouped_identity_v41 import FLAG,FUSION,GROUPED,normalize,geometry as tuning_geometry
import nf4_row_reuse_candidate_v1 as kernel
import nf4_row_pair_grouped_v41 as grouped_kernel
import nf4_same_expert_fused64_v39 as fused_kernel
import nf4_row_quad_candidate_v1 as quad

class _Graph:
 def __init__(self,owner):self.owner=owner;owner.live_graphs+=1
 def __del__(self):self.owner.live_graphs-=1
class _Boundary(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,owner):ctx.live=_Graph(owner);return x
 @staticmethod
 def backward(ctx,g):return g,None

class RowReuse:
 def __init__(self,module,choose,blocks,memo):
  self.module=module;self.choose=choose;self.memo=memo;self.original=module.grouped_mm;self.live_graphs=0
  self.roles=packed_roles(blocks);self.counts={r+'_'+d:0 for r in ('gate_proj','up_proj','down_proj') for d in ('forward','backward')}
  self.quad_counts=dict(self.counts)
  self.bk64_counts=dict(self.counts)
  self.wave_counts=dict(self.counts)
  self.tuning_counts=dict(self.counts)
  self.fusion_counts=dict(self.counts)
  self.group_counts=dict(self.counts)
  self.device_branch_counts={}
  def run(a,p,ends,rows,transposed):
   cfg=self.choose();direction=cfg.get(FLAG,'off');side='forward' if transposed else 'backward'
   if direction not in ('both',side):return self.original(a,p,ends,rows,transposed)
   item=self.roles.get(id(p))
   if item is None or item[0] is not p:raise ValueError('Unknown packed row-reuse identity')
   selected=quad if cfg.get('nf4_row_reuse_tiles',2)==4 else kernel
   geometry=tuning_geometry(cfg,item[1],transposed)
   key=item[1]+'_'+side
   if cfg.get(GROUPED,{}).get(item[1],{}).get(side+'_kernel'):
    out=grouped_kernel.launch(a,p,ends,rows,transposed,geometry,write_once=cfg.get('nf4_static_write_once',False),group_pairs=cfg[GROUPED][item[1]][side+'_kernel'])
    self.group_counts[key]+=1
   elif cfg.get(FUSION,{}).get(item[1],{}).get(side+'_kernel'):
    if key not in self.device_branch_counts:self.device_branch_counts[key]=torch.zeros(2,device=a.device,dtype=torch.int64)
    out=fused_kernel.launch(a,p,ends,rows,transposed,geometry,write_once=cfg.get('nf4_static_write_once',False),branch_counts=self.device_branch_counts[key])
    self.fusion_counts[key]+=1
   else:
    out=selected.launch(a,p,ends,rows,transposed,geometry,write_once=cfg.get('nf4_static_write_once',False))
   self.counts[item[1]+'_'+side]+=1
   if selected is quad:self.quad_counts[item[1]+'_'+side]+=1
   if selected is kernel and geometry['bk']==64:self.bk64_counts[item[1]+'_'+side]+=1
   if side+'_kernel' in cfg.get('nf4_row_reuse_waves',{}).get(item[1],{}):self.wave_counts[item[1]+'_'+side]+=1
   if side+'_kernel' in cfg.get('nf4_row_reuse_tuning',{}).get(item[1],{}):self.tuning_counts[item[1]+'_'+side]+=1
   return out
  self.installed=run;module.grouped_mm=run
 def branch_snapshot(self):
  # Read only outside timed windows. Atomics inside the launched kernel are timed.
  return {key:dict(zip(('fused_output_tiles','fallback_output_tiles'),self.device_branch_counts[key].detach().cpu().tolist() if key in self.device_branch_counts else (0,0))) for key in self.counts}
 def boundary(self):
  self.memo.assert_boundary()
  if self.live_graphs:raise RuntimeError('Release row-reuse graph before changing policy')
 def guard(self,value):
  if torch.is_tensor(value):return _Boundary.apply(value,self) if value.requires_grad and torch.is_grad_enabled() else value
  if isinstance(value,tuple):return tuple(self.guard(x) for x in value)
  if isinstance(value,list):return [self.guard(x) for x in value]
  return value
 def close(self):
  self.boundary()
  if self.module.grouped_mm is not self.installed:raise RuntimeError('Remove outer wrappers before row reuse')
  self.module.grouped_mm=self.original
