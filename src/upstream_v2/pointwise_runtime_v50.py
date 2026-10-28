"""Configuration and measured-window evidence for the isolated routing policy."""
from pointwise_identity_v50 import normalize,POINTWISE
class PointwiseRuntime:
 def __init__(self,routing):self.policy=routing.pointwise
 def configure(self,c):
  p=normalize(c).get(POINTWISE,{})
  self.policy.configure(p.get('mode','disabled'),p.get('block',256))
 def boundary(self):self.policy.boundary()
 def statistics(self):return self.policy.statistics()
 def record_window(self,row,before,c,*,require_execution=True):
  self.boundary();after=self.statistics();delta={k:after[k]-before[k] for k in ('forward_calls','backward_calls','fallback_calls','elided_saved_activation_bytes')}
  assert all(v>=0 for v in delta.values()),'POINTWISE_COUNTER_RESET'
  if POINTWISE in normalize(c):
   if require_execution:assert delta['forward_calls']>=delta['backward_calls']>0 and delta['elided_saved_activation_bytes']>0,'POINTWISE_NOT_EXECUTED'
  else:assert delta['forward_calls']==delta['backward_calls']==delta['elided_saved_activation_bytes']==0,'POINTWISE_DISABLED_EXECUTED'
  row.update(pointwise=after,pointwise_calls=delta,pointwise_live_graphs=after['live_graphs'],pointwise_elided_bytes_are_not_peak_reduction=True)
