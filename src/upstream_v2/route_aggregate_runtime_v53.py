"""Graph boundaries and actual direction-specific route aggregation evidence."""
from route_aggregate_identity_v53 import normalize,AGGREGATE
class RouteAggregateRuntime:
 def __init__(self,routing):self.policy=routing.route_aggregate
 def configure(self,c):
  p=normalize(c).get(AGGREGATE,{})
  self.policy.configure(p.get('mode','disabled'),p.get('block',256))
 def boundary(self):self.policy.boundary()
 def statistics(self):return self.policy.statistics()
 def record_window(self,row,before,c,*,require_execution=True):
  self.boundary();after=self.statistics();p=normalize(c).get(AGGREGATE,{})
  mode=p.get('mode','disabled');block=p.get('block',256)
  assert after['mode']==mode and after['block']==block,'ROUTE_AGGREGATE_CONFIG_MISMATCH'
  keys=('forward_calls','backward_calls','forward_fallbacks','backward_fallbacks','elided_pointwise_tensor_bytes')
  delta={k:after[k]-before[k] for k in keys}
  assert all(v>=0 for v in delta.values()),'ROUTE_AGGREGATE_COUNTER_RESET'
  for side,enabled in [('forward',mode in ('weighted_forward','both')),('backward',mode in ('input_backward','both'))]:
   if enabled and require_execution:
    assert delta[side+'_calls']>0 and delta[side+'_fallbacks']==0,'ROUTE_AGGREGATE_NOT_EXECUTED_'+side
   if not enabled:assert delta[side+'_calls']==delta[side+'_fallbacks']==0,'ROUTE_AGGREGATE_UNSELECTED_EXECUTED_'+side
  if mode=='disabled':assert delta['elided_pointwise_tensor_bytes']==0,'ROUTE_AGGREGATE_DISABLED_ELISION'
  elif require_execution:assert delta['elided_pointwise_tensor_bytes']>0,'ROUTE_AGGREGATE_NO_ELISION'
  row.update(route_aggregate=after,route_aggregate_calls=delta,route_aggregate_live_graphs=after['live_graphs'],route_aggregate_structural_bytes_are_not_peak_reduction=True,route_aggregate_no_call_numeric_may_use_dynamic_fallback=not require_execution and not delta['forward_calls'] and not delta['backward_calls'])
