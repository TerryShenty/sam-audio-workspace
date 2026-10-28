"""Scalar-only accounting around the unchanged accepted replay implementation.

No tensor-value reads, additional source traversal, activation references, or
peak subtraction. Logical writes are not HBM traffic or measured saved time.
"""
from checkpoint_call_binding_v58 import LayerReplayMemo

OBS=('source_read_attempts','source_read_failures','bound_scopes_started',
     'capture_attempts','capture_without_entry','replay_take_attempts','replay_take_fallbacks',
     'hit_route_rows','hit_padded_rows','skipped_routed_logical_bytes','skipped_pointwise_logical_bytes')
DELTA=('route_captures','route_hits','route_misses','metadata_budget_skips',
       'bypassed_input_calls','bypassed_pointwise_calls','forward_calls','replay_hits',
       'replay_misses','replay_transfers','replay_transfer_bytes','repeated_replay_recomputes')

class ObservedLayerReplayMemo(LayerReplayMemo):
    def __init__(self,**kwargs):
        self.observation=dict.fromkeys(OBS,0)
        super().__init__(**kwargs)
    def source_stamp(self):
        self.observation['source_read_attempts']+=1
        try:return super().source_stamp()
        except Exception:
            self.observation['source_read_failures']+=1
            raise
    def contexts_for(self,binding):
        contexts=super().contexts_for(binding)
        self.observation['bound_scopes_started']+=1
        return contexts
    def capture(self,x,selected,packed,rows,policies,metadata):
        scope=self.scope();eligible=self.enabled and scope is not None and not scope.replay and not self.verify
        before=self.counters['route_captures']
        if eligible:self.observation['capture_attempts']+=1
        super().capture(x,selected,packed,rows,policies,metadata)
        if eligible and self.counters['route_captures']==before:self.observation['capture_without_entry']+=1
    def take(self,x,selected,packed,rows,policies,pointwise):
        scope=self.scope();eligible=self.enabled and scope is not None and scope.replay and not self.verify
        if eligible:self.observation['replay_take_attempts']+=1
        result=super().take(x,selected,packed,rows,policies,pointwise)
        if eligible:
            if result is None:self.observation['replay_take_fallbacks']+=1
            else:
                obs=self.observation;obs['hit_route_rows']+=x.shape[0]*selected.shape[1]
                obs['hit_padded_rows']+=rows+16
                obs['skipped_routed_logical_bytes']+=(rows+16)*x.shape[1]*x.element_size()
                obs['skipped_pointwise_logical_bytes']+=(rows+16)*packed['gate_proj'].shape[0]*x.element_size()
        return result
    def statistics(self):return dict(super().statistics(),observation=dict(self.observation))

class ReplayWindowObservation:
    def __init__(self,memo):self.memo=memo
    def snapshot(self):
        self.memo.assert_boundary()
        return self.memo.statistics()
    def record(self,row,before,config,lengths,*,require_hit):
        self.memo.assert_boundary();after=self.memo.statistics()
        delta={k:after[k]-before[k] for k in DELTA}
        obs={k:after['observation'][k]-before['observation'][k] for k in OBS}
        if not all(v>=0 for v in (*delta.values(),*obs.values())):raise AssertionError('REPLAY_COUNTER_RESET')
        hits=delta['route_hits']
        assert delta['bypassed_input_calls']==delta['bypassed_pointwise_calls']==hits
        assert obs['replay_take_attempts']==hits+obs['replay_take_fallbacks']
        assert obs['capture_attempts']==delta['route_captures']+obs['capture_without_entry']
        assert obs['source_read_failures']==0
        assert obs['source_read_attempts']==3*obs['bound_scopes_started']
        assert delta['replay_hits']>=3*hits
        if config.get('nf4_memo_consume_once'):assert delta['replay_transfers']==delta['replay_hits']
        if not config.get('checkpoint_atomic_route_replay'):
            assert hits==delta['route_captures']==obs['source_read_attempts']==0
        elif require_hit:assert hits>0,'ATOMIC_REPLAY_NOT_EXECUTED'
        for key in ('live_bytes','active_contexts','outstanding_scopes','metadata_reserved_bytes'):assert after[key]==0,key
        pair_lengths=[];sample_lengths=[]
        for pair in lengths:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2 or not all(type(n) is int and n>0 for n in pair):
                raise ValueError('Require already-returned host token lengths for two samples')
            sample_lengths.extend(pair);pair_lengths.append(max(pair))
        row['atomic_replay']=dict(counters=delta,observation=obs,
          pair_padded_lengths=pair_lengths,sample_token_lengths=sample_lengths,
          pair_length_sum=sum(pair_lengths),pair_length_max=max(pair_lengths,default=0),
          process_metadata_high_water_bytes=after['metadata_peak_bytes'],
          high_water_is_process_cumulative=True,logical_bytes_are_not_hbm_peak_or_time=True,
          enabled_without_hit_may_be_dynamic_or_lazy=bool(config.get('checkpoint_atomic_route_replay') and not hits))
