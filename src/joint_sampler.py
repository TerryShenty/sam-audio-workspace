"""Deterministic infinite four-rank sampler, two examples per rank/update.

Cross epoch boundaries inside a batch: no dropped/padded training rows and no
epoch cap. Commit cursor only after a globally successful optimizer update.
"""
import hashlib, json, random

class JointSampler:
    def __init__(self, sample_ids, *, rank, seed, data_identity, world_size=4, per_rank=2):
        assert world_size==4 and per_rank==2 and 0<=rank<world_size
        assert sample_ids and len(set(sample_ids))==len(sample_ids)
        assert isinstance(seed,int) and data_identity
        self.ids=tuple(sample_ids);self.rank=rank;self.seed=seed
        self.world_size=world_size;self.per_rank=per_rank
        self.identity=hashlib.sha256(json.dumps([data_identity,self.ids,seed,world_size,per_rank]).encode()).hexdigest()
        self.cursor=0;self.cached_epoch=None;self.order=None;self.pending=None

    def _index(self, position):
        epoch,offset=divmod(position,len(self.ids))
        if self.cached_epoch!=epoch:
            order=list(range(len(self.ids)))
            # Avoid process hash randomization; identical permutation on all ranks.
            seed=int.from_bytes(hashlib.sha256(f'{self.seed}:{epoch}'.encode()).digest(),'big')
            random.Random(seed).shuffle(order)
            self.cached_epoch=epoch;self.order=order
        return self.order[offset]

    def peek(self):
        if self.pending is None:
            start=self.cursor+self.rank*self.per_rank
            self.pending=tuple(self._index(start+i) for i in range(self.per_rank))
        return self.pending

    def commit(self):
        assert self.pending is not None,'No pending complete update'
        self.cursor+=self.world_size*self.per_rank;self.pending=None

    def state_dict(self):
        assert self.pending is None,'Checkpoint only at a committed update boundary'
        return dict(identity=self.identity,global_samples=self.cursor,rank=self.rank,
                    world_size=self.world_size,per_rank=self.per_rank,seed=self.seed,
                    completed_epochs=self.cursor//len(self.ids),epoch_offset=self.cursor%len(self.ids))

    def load_state_dict(self,state):
        assert self.pending is None,'Cannot restore over a pending update'
        assert state['identity']==self.identity and state['rank']==self.rank
        assert (state['world_size'],state['per_rank'],state['seed'])==(4,2,self.seed)
        cursor=state['global_samples'];assert type(cursor) is int and cursor>=0 and cursor%8==0
        assert divmod(cursor,len(self.ids))==(state['completed_epochs'],state['epoch_offset'])
        self.cursor=cursor;self.cached_epoch=None;self.order=None
