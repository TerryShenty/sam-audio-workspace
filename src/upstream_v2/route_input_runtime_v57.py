"""Per-window forward preparation evidence; policy retains no graph tensors."""
from route_input_identity_v57 import normalize,INPUT
class RouteInputRuntime:
 def __init__(self,routing):self.policy=routing.route_input
 def boundary(self):
  assert not self.policy.statistics()['tensor_cache'] and self.policy.statistics()['live_graphs']==0,'ROUTE_INPUT_RETAINED_GRAPH'
 def configure(self,c):
  self.boundary();p=normalize(c).get(INPUT,{})
  self.policy.configure(p.get('mode','disabled'),p.get('block',1024))
 def statistics(self):return self.policy.statistics()
 def record_window(self,row,before,c,*,require_execution=True):
  self.boundary();after=self.statistics();p=normalize(c).get(INPUT,{})
  assert after['mode']==p.get('mode','disabled') and after['block']==p.get('block',1024),'ROUTE_INPUT_CONFIG_MISMATCH'
  keys=('calls','fallbacks','kernel_launches','route_rows','padded_rows_written','elided_gather_tensor_bytes')
  delta={k:after[k]-before[k] for k in keys}
  assert all(v>=0 for v in delta.values()),'ROUTE_INPUT_COUNTER_RESET'
  assert delta['kernel_launches']==2*delta['calls'],'ROUTE_INPUT_LAUNCH_COUNT'
  if delta['calls']:assert delta['route_rows']>0 and delta['padded_rows_written']>=delta['route_rows'] and delta['elided_gather_tensor_bytes']>0,'ROUTE_INPUT_INCOMPLETE_EVIDENCE'
  if not p:assert not any(delta.values()),'ROUTE_INPUT_DISABLED_EXECUTED'
  elif require_execution:
   assert delta['calls']>0 and delta['kernel_launches']==2*delta['calls'] and delta['fallbacks']==0 and delta['route_rows']>0 and delta['padded_rows_written']>=delta['route_rows'] and delta['elided_gather_tensor_bytes']>0,'ROUTE_INPUT_NOT_EXECUTED'
  row.update(route_input_prepare=after,route_input_prepare_calls=delta,route_input_prepare_live_graphs=after['live_graphs'],route_input_structural_bytes_are_not_peak_reduction=True,route_input_no_call_numeric_may_use_dynamic_fallback=not require_execution and not delta['calls'])
