"""Graph boundaries and actual backward preparation evidence for each window."""
from route_backward_identity_v55 import normalize,PREP
class RouteBackwardRuntime:
 def __init__(self,routing):self.policy=routing.route_backward
 def configure(self,c):
  p=normalize(c).get(PREP,{})
  self.policy.configure(p.get('mode','disabled'),p.get('warps',4))
 def boundary(self):self.policy.boundary()
 def statistics(self):return self.policy.statistics()
 def record_window(self,row,before,c,*,require_execution=True):
  self.boundary();after=self.statistics();p=normalize(c).get(PREP,{})
  assert after['mode']==p.get('mode','disabled') and after['warps']==p.get('warps',4),'ROUTE_BACKWARD_CONFIG_MISMATCH'
  keys=('calls','fallbacks','kernel_launches','route_rows','elided_expanded_tensor_bytes')
  delta={k:after[k]-before[k] for k in keys}
  assert all(v>=0 for v in delta.values()),'ROUTE_BACKWARD_COUNTER_RESET'
  if not p:assert not any(delta.values()),'ROUTE_BACKWARD_DISABLED_EXECUTED'
  elif require_execution:
   assert delta['calls']>0 and delta['kernel_launches']==delta['calls'] and delta['fallbacks']==0 and delta['route_rows']>0 and delta['elided_expanded_tensor_bytes']>0,'ROUTE_BACKWARD_NOT_EXECUTED'
  row.update(route_backward_prepare=after,route_backward_prepare_calls=delta,route_backward_prepare_live_graphs=after['live_graphs'],route_backward_structural_bytes_are_not_peak_reduction=True,route_backward_no_call_numeric_may_use_dynamic_fallback=not require_execution and not delta['calls'])
