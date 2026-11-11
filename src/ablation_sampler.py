"""Deterministic N-rank A/B pair sampler: one mixture (its A and B targets) per rank per update.

All four arms use this sampler, so data order, batch composition and update count
are identical across arms; only the objective differs. Per-rank batch stays 2
(global 2N), as in V6. Like the V6 JointSampler it never drops or pads rows and
commits its cursor only after a globally successful optimizer update.
"""
import hashlib, json, random


class PairSampler:
    def __init__(self, rows, *, rank, seed, data_identity, world_size=4, per_rank=2):
        assert type(world_size) is int and 1 <= world_size <= 8 and per_rank == 2 and 0 <= rank < world_size
        assert isinstance(seed, int) and data_identity
        pairs = {}
        for index, row in enumerate(rows):
            pairs.setdefault(row['pair_id'], {})[row['role']] = index
        assert pairs and all(set(v) == {'A', 'B'} for v in pairs.values()), 'Every training mixture needs exactly A and B'
        assert sum(len(v) for v in pairs.values()) == len(rows)
        self.pair_ids = tuple(sorted(pairs))
        self.pairs = tuple((pairs[p]['A'], pairs[p]['B']) for p in self.pair_ids)
        self.ids = tuple(r['sample_id'] for r in rows)
        assert len(set(self.ids)) == len(self.ids)
        self.rank, self.seed, self.world_size, self.per_rank = rank, seed, world_size, per_rank
        self.identity = hashlib.sha256(json.dumps([data_identity, self.ids, self.pair_ids, seed, world_size, per_rank, 'pair-v1']).encode()).hexdigest()
        self.cursor = 0  # global samples consumed (2 per pair), committed only
        self.cached_epoch = None; self.order = None; self.pending = None

    @property
    def samples_per_epoch(self):
        return 2 * len(self.pairs)

    def _pair(self, position):
        epoch, offset = divmod(position, len(self.pairs))
        if self.cached_epoch != epoch:
            order = list(range(len(self.pairs)))
            seed = int.from_bytes(hashlib.sha256(f'{self.seed}:pairs:{epoch}'.encode()).digest(), 'big')
            random.Random(seed).shuffle(order)
            self.cached_epoch, self.order = epoch, order
        return self.pairs[self.order[offset]]

    def peek(self):
        if self.pending is None:
            self.pending = self._pair(self.cursor // 2 + self.rank)
        return self.pending

    def commit(self):
        assert self.pending is not None, 'No pending complete update'
        self.cursor += self.world_size * self.per_rank; self.pending = None

    @property
    def completed_epochs(self):
        return self.cursor // self.samples_per_epoch

    def state_dict(self):
        assert self.pending is None, 'Checkpoint only at a committed update boundary'
        return dict(identity=self.identity, global_samples=self.cursor, rank=self.rank,
                    world_size=self.world_size, per_rank=self.per_rank, seed=self.seed,
                    completed_epochs=self.completed_epochs, epoch_offset=self.cursor % self.samples_per_epoch)

    def load_state_dict(self, state):
        assert self.pending is None, 'Cannot restore over a pending update'
        assert state['identity'] == self.identity and state['rank'] == self.rank
        assert (state['world_size'], state['per_rank'], state['seed']) == (self.world_size, self.per_rank, self.seed)
        cursor = state['global_samples']
        assert type(cursor) is int and cursor >= 0 and cursor % (self.world_size * self.per_rank) == 0
        assert divmod(cursor, self.samples_per_epoch) == (state['completed_epochs'], state['epoch_offset'])
        self.cursor = cursor; self.cached_epoch = None; self.order = None
