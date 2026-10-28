"""Explicit longest-input prefix for qualification only; never production sampling."""
import hashlib,json
from joint_sampler import JointSampler

class QualificationSampler(JointSampler):
    def __init__(self,base,cases):
        assert base.cursor==0 and base.pending is None
        super().__init__(base.ids,rank=base.rank,seed=base.seed,data_identity=base.identity,world_size=base.world_size)
        lookup={sid:i for i,sid in enumerate(base.ids)}
        self.prefix=[];self.case_names=[]
        for name,case in sorted(cases.items()):
            ranks=sorted(case['ranks'],key=lambda x:x['rank'])
            assert [r['rank'] for r in ranks]==list(range(4))
            ids=[x['sample_id'] for r in ranks for x in r['samples']]
            assert all(len(r['samples'])==2 for r in ranks) and len(set(ids))==8
            assert all(sid in lookup for sid in ids)
            # Repack the fixed eight demanding examples for N ranks.
            # Diagnostic-only padding repeats known cases; production never pads.
            width=self.world_size*self.per_rank
            for offset in range(0,len(ids),width):
                part=ids[offset:offset+width]
                part += [ids[i%len(ids)] for i in range(width-len(part))]
                self.prefix.extend(lookup[sid] for sid in part)
                self.case_names.append(name+':'+str(offset//width))
        assert self.prefix and len(self.prefix)%(self.world_size*self.per_rank)==0
        self.base=base
        self.identity=hashlib.sha256(json.dumps(['qualification_only',base.identity,self.prefix],sort_keys=True).encode()).hexdigest()
    def _index(self,position):
        if position<len(self.prefix):return self.prefix[position]
        # Start the unchanged production permutation only after the explicit prefix.
        # This is diagnostic replay, not credited production epochs or data position.
        return self.base._index(position-len(self.prefix))
    def state_dict(self):
        state=super().state_dict()
        state.update(qualification_only=True,prefix_length=len(self.prefix),case_names=self.case_names)
        return state
    def load_state_dict(self,state):
        assert state['qualification_only'] is True and state['prefix_length']==len(self.prefix)
        assert state['case_names']==self.case_names
        super().load_state_dict(state)

def attach(session,cases):
    trainer=session.trainer
    assert trainer.step_number==0 and not trainer.busy and not trainer.restore_failed
    sampler=QualificationSampler(trainer.sampler,cases)
    trainer.sampler=sampler
    trainer.identity=hashlib.sha256(json.dumps(['qualification_only',trainer.identity,sampler.identity]).encode()).hexdigest()
    trainer.config['qualification_only']=True
    return dict(qualification_only=True,initialization_usable_for_production=False,
                prefix_updates=len(sampler.prefix)//(sampler.world_size*sampler.per_rank),cases=sampler.case_names,sampler_identity=sampler.identity)
