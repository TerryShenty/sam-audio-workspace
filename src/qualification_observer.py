"""Instrument qualification only; sampling overhead is excluded from speed claims."""
import copy

class QualificationObserver:
    def __init__(self,trainer,runtime):
        self.trainer=trainer;self.runtime=runtime
        self.updates=0;self.language_updates=0;self.changed=set()

    def samples(self):
        # Bound diagnostic copies to 1024 elements per trainable tensor.
        return {n:p.detach().reshape(-1)[:1024].cpu().clone() for n,p in self.trainer.params.items()}

    def step(self,operation):
        import torch
        before=self.samples();step=self.trainer.step_number
        result=operation()
        assert self.trainer.step_number==step+1,'NO_COMMITTED_UPDATE'
        after=self.samples();changed=set()
        for group,names in self.trainer.groups.items():
            assert all(bool(torch.isfinite(after[n]).all()) for n in names),'NONFINITE_SAMPLED_PARAMETER'
            if any(not torch.equal(before[n],after[n]) for n in names):changed.add(group)
        assert changed==set(self.trainer.groups),'NO_OBSERVED_NUMERICAL_UPDATE_IN_GROUP'
        self.changed.update(changed);self.updates+=1;self.language_updates+=1
        return result

    def snapshot(self):
        s=self.runtime.statistics();c=s['compiler'];r=s['routes']
        return dict(attention=copy.deepcopy(s['attention']),retained_layers=s['retention']['requested'],memo_cap_bytes=s['memo']['cap_bytes'],
            language_updates=self.language_updates,optimizer_updates=self.updates,
            retention_fallback_calls=s['retention']['fallback_calls'],
            native_rms_graph_executions=sum(x.get('executions',0) for x in c['compiled_regions']
                if x['region']=='sam' and x.get('native_rms_nodes',0)>0),
            sam_compiled_calls=c['calls']['sam']-c['fallback_calls']['sam'],
            qwen_compiled_calls=c['calls']['qwen_attention']-c['fallback_calls']['qwen_attention'],
            route_input_calls=r['input']['calls']-r['input']['fallbacks'],
            route_backward_calls=r['backward']['calls']-r['backward']['fallbacks'],
            route_aggregate_calls=r['aggregate']['forward_calls']-r['aggregate']['forward_fallbacks'],
            pointwise_calls=r['pointwise']['forward_calls']-r['pointwise']['fallback_calls'],
            memo_hits=s['memo']['replay_hits'],finite_updated_groups=sorted(self.changed),
            word8=copy.deepcopy(s['word8']),word4=copy.deepcopy(s['word4']),bytepair=copy.deepcopy(s['bytepair']))
