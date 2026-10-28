"""Isolated per-layer live supplier; no production worker imports this yet.

Roots are supplied by the controlled worker installation. Each read follows
their actual current parents, but visits only one layer and its three packed
roles. LayerSource separately stamps the tensors. Counters, caches, graph sets
and optimizer/RNG/data state are intentionally absent. This does not attempt
to protect arbitrary mutation of every imported Python helper or closure.
"""
import types
import weakref

from attention_origin_binding_v58b import function_stamp
from bp_identity import CODEBOOK,BYTEPAIR,WORD4,WORD8


ROLES = ('gate_proj', 'up_proj', 'down_proj')
POLICIES = (('input', 'route_input', 'block'),
            ('backward', 'route_backward', 'warps'),
            ('aggregate', 'route_aggregate', 'block'),
            ('pointwise', 'pointwise', 'block'))
GEOMETRY = ('BM', 'BN', 'BK', 'WARPS', 'WAVES', 'GROUP_M')


def config_value(value):
    """Copy only bounded host configuration, preserving keys and value types.

    No opaque objects/tensors, normalization, statistics or tensor-value reads.
    Lists are needed for the actual normalized memo/scale roles configuration.
    """
    if value is None or type(value) in (bool, int, float, str):
        return (type(value).__name__, value)
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(config_value(v) for v in value))
    if type(value) is frozenset and all(type(v) is str for v in value):
        return ('frozenset', tuple(sorted(value)))
    if type(value) is dict and all(type(k) is str for k in value):
        return ('dict', tuple((k, config_value(value[k])) for k in sorted(value)))
    raise ValueError('Live runtime configuration must contain host values only')


def callable_stamp(function):
    if type(function) is types.MethodType:
        return (id(function.__self__), function_stamp(function.__func__))
    return (None, function_stamp(function))


def require_identity(actual, expected_id, name):
    if id(actual) != expected_id:
        raise RuntimeError('Runtime parent replaced; rebuild at boundary: '+name)


class LayerRuntimeBinding:
    def __init__(self, owner, index, tables):
        owner.require_open()
        if type(index) is not int or not 0 <= index < 48:
            raise ValueError('Require one current Qwen layer')
        if tables is not owner.stack.row_reuse.tables:
            raise ValueError('LayerSource and runtime must use the same live table parent')
        layer = owner.model.layers[index]
        packed = layer.mlp._packed_nf4
        if set(packed) != set(ROLES):
            raise ValueError('Require the three current packed projections')
        self.owner, self.index = owner, index
        self.parents = (id(layer), id(layer.mlp), id(packed), id(layer.self_attn))
        self.packed_ids = tuple(id(packed[role]) for role in ROLES)
        self.table_ids = tuple(id(tables.get(key)) for key in self.packed_ids)
        self.book_ids=tuple(id(owner.stack.row_reuse.codebooks.get(key)) for key in self.packed_ids)
        self.book_payloads=tuple((b.values,b.source) if b else None for b in (owner.stack.row_reuse.codebooks.get(key) for key in self.packed_ids))
        self.origin_id = id(owner.compiler.origin_for(layer.self_attn))
        self.attention_runtime_id=id(owner.compiler.attention_nf4_runtime)
        self.attention_source=owner.compiler.attention_nf4_runtime.layer_signature(index)

    def __call__(self):
        owner = self.owner
        owner.require_open()
        stack, compiler, policy = owner.stack, owner.compiler, owner.policy()
        if policy is None:
            raise RuntimeError('Runtime registry owner was released')
        # Parent edges are read live, not from a captured old tables dictionary.
        owner.check_parents(policy)
        layer = owner.model.layers[self.index]
        for actual, expected, name in zip(
                (layer, layer.mlp, layer.mlp._packed_nf4, layer.self_attn),
                self.parents, ('layer', 'mlp', 'packed roles', 'attention')):
            require_identity(actual, expected, name)
        packed, reuse = layer.mlp._packed_nf4, stack.row_reuse
        if set(packed) != set(ROLES):
            raise RuntimeError('Packed role keyset changed')
        for role, key, table_id in zip(ROLES, self.packed_ids, self.table_ids):
            p = packed[role]
            require_identity(p, key, role)
            item = reuse.roles.get(key)
            if type(item) is not tuple or len(item) != 2 or item[0] is not p or item[1] != role:
                raise RuntimeError('Live row-reuse role mapping changed: '+role)
            require_identity(reuse.tables.get(key), table_id, 'scale table '+role)
            if stack.current.get(BYTEPAIR,False):reuse.bound_bp_source(p)
            if stack.current.get(WORD4,False):reuse.bound_word_source(p)
            if stack.current.get(WORD8,False):reuse.bound_eight_source(p)
        pair_source=stack.pair.validate_sources(packed['gate_proj'])
        if pair_source is None or pair_source[1] is not packed['up_proj']:raise RuntimeError('Layer gate/up pair changed')
        attention_runtime=compiler.attention_nf4_runtime
        require_identity(attention_runtime,self.attention_runtime_id,'attention NF4 runtime')
        attention_source=attention_runtime.layer_signature(self.index)
        if attention_source!=self.attention_source:raise RuntimeError('Attention source changed after publication')
        origin = compiler.origin_for(layer.self_attn)
        require_identity(origin, self.origin_id, 'published attention origin')

        memo, routing, pair = stack.memo, stack.routing, stack.pair
        if stack.closed or pair.closed:
            raise RuntimeError('Require an open stack')
        if stack.probe is not None:raise RuntimeError('No diagnostic wrappers in v60 candidate')
        codebook_signature=[]
        for key,book_id,payload in zip(self.packed_ids,self.book_ids,self.book_payloads):
            book=reuse.codebooks.get(key)
            require_identity(book,book_id,'constant codebook')
            if stack.current.get(CODEBOOK,False):
                if not payload or book.values is not payload[0] or book.source is not payload[1]:
                    raise RuntimeError('Codebook immutable values or source replaced')
                reuse.bound_codebook(reuse.roles[key][0])
                codebook_signature.append((key,id(book),id(book.values),book.values,book.source))
            elif book is not None:raise RuntimeError('Disabled codebook retained')
        probe_signature=('constant_codebook_v60',tuple(codebook_signature))
        if policy.linear.max_bytes or policy.active:
            raise RuntimeError('This candidate requires the inactive plain LoRA path')
        if stack.current.get('nf4_gate_up_shared_input', False):
            raise RuntimeError('This fixed candidate does not enable shared-input fusion')
        if reuse.choose() is not stack.current or pair.choose() is not stack.current:
            raise RuntimeError('Backend selectors do not read the live current configuration')
        if routing.Experts is not pair.installed_experts:
            raise RuntimeError('Actual Experts dispatch differs from the installed pair wrapper')
        if stack.alias.grouped_mm is not stack.modules[0].grouped_mm:
            raise RuntimeError('Dynamic grouped-mm alias disagrees with its backend')

        runtime_values = []
        for (name, attribute, tuning), runtime, expected in zip(
                POLICIES, owner.runtimes, owner.policy_ids):
            current = getattr(routing, attribute)
            require_identity(current, expected, attribute)
            if runtime.policy is not current:
                raise RuntimeError('Runtime and routing policy disagree: '+name)
            runtime_values.append((name, config_value(current.mode),
                                   config_value(getattr(current, tuning))))
        experts = pair.original_experts
        # The outer DispatchExperts has apply; the selected original class has
        # forward/backward. Reading nonexistent outer forward would be wrong.
        expert_calls = (id(routing.Experts), callable_stamp(routing.Experts.apply),
                        id(experts), callable_stamp(experts.forward),
                        callable_stamp(experts.backward))
        backend_values = tuple((id(module), callable_stamp(module.grouped_mm),
                                tuple((key, hasattr(module, key),
                                       config_value(getattr(module, key, None)))
                                      for key in GEOMETRY))
                               for module in stack.modules)
        return ('live_runtime_v60_1', self.index, probe_signature,
                config_value(stack.current), config_value(reuse.scale_policy),
                tuple(runtime_values),
                config_value((memo.enabled, memo.roles, memo.max_bytes, memo.verify,
                              memo.consume_once, memo.metadata_cap)),
                config_value((policy.first, policy.count)),
                config_value((routing.PAD, routing.FP32_OUTPUT)),
                callable_stamp(routing.forward), expert_calls, backend_values,
                callable_stamp(reuse.choose), callable_stamp(pair.choose),
                callable_stamp(layer.mlp.forward), origin(), attention_source)


class LiveRuntimeBindings:
    """One factory per graph-free registry generation, closed on invalidation.

    The worker must retain these exact stack/compiler/runtime roots and route
    configuration changes through TransitionRegistry. A source signature is
    compared by the existing checkpoint adapter on all three reads; creating a
    supplier alone does not implement a training-state or graph boundary gate.
    """
    def __init__(self, policy, stack, compiler, runtimes):
        if len(policy.model.layers) != 48 or set(runtimes) != {p[0] for p in POLICIES}:
            raise ValueError('Require 48 layers and all four actual runtimes')
        if type(stack.modules) is not tuple or len(stack.modules) != 2:
            raise ValueError('Require the installed dynamic/static backend pair')
        if stack.modules[0] is stack.modules[1]:
            raise ValueError('Backends must be distinct')
        self.policy = weakref.ref(policy)
        self.model, self.stack, self.compiler = policy.model, stack, compiler
        self.runtimes = tuple(runtimes[p[0]] for p in POLICIES)
        self.policy_ids = tuple(id(runtime.policy) for runtime in self.runtimes)
        self.parent_ids = self.parents(policy)
        self.closed = False
        self.check_parents(policy)

    def parents(self, policy):
        stack = self.stack
        return tuple(id(v) for v in
                     (policy.model, policy.model.layers, stack.routing, stack.modules,
                      stack.modules[0], stack.modules[1], stack.alias, stack.memo,
                      stack.row_reuse, stack.row_reuse.tables, stack.row_reuse.roles,
                      stack.row_reuse.codebooks,stack.row_reuse.codebook_bindings,
                      stack.row_reuse.bp_sources,stack.row_reuse.bp_source_ids,
                      stack.row_reuse.word_sources,stack.row_reuse.word_source_ids,
                      stack.row_reuse.eight_sources,stack.row_reuse.eight_source_ids,
                      stack.pair, stack.pair.pairs, self.compiler.attention_origins, policy.linear))

    def require_open(self):
        if self.closed:
            raise RuntimeError('Runtime binding generation was closed')

    def check_parents(self, policy):
        if self.parents(policy) != self.parent_ids:
            raise RuntimeError('A live runtime parent was replaced; rebuild at boundary')
        stack = self.stack
        if (policy.memo is not stack.memo or stack.routing.replay_memo is not stack.memo
                or stack.row_reuse.memo is not stack.memo or stack.pair.memo is not stack.memo
                or stack.routing.backend is not stack.modules[1]
                or stack.row_reuse.module is not stack.modules[1]
                or stack.pair.module is not stack.modules[1]
                or stack.pair.routing is not stack.routing or stack.pair.row_reuse is not stack.row_reuse):
            raise RuntimeError('Runtime, backend, routing and memo roots disagree')

    def for_layer(self, index, tables):
        return LayerRuntimeBinding(self, index, tables)

    def __call__(self, index):
        # A factory is callable for the inherited rebuild API, but it cannot
        # silently supply a global signature to every layer.
        return self.for_layer(index, self.stack.row_reuse.tables)()

    def close(self):
        # The registry's outer boundary must have rejected all live graphs first.
        self.closed = True
        self.model = self.stack = self.compiler = self.policy = None
        self.runtimes = self.policy_ids = self.parent_ids = ()
