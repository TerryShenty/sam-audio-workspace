"""Isolated Qwen decoder source registry. No production worker imports this.

Construct after configuring forwards/derived tables. Reads inspect only one
layer, never enumerate modules/parameters. Object replacement requires rebuild;
normal tensor updates between released graphs receive a fresh scope signature.
This covers the pinned architecture, not arbitrary modules or mutable closures.
"""
from functools import partial
import torch
from checkpoint_call_binding_v58 import LayerCheckpointInvoker
from lora_policy_v21 import JointCheckpointPolicy

ROLES = ('gate_proj', 'up_proj', 'down_proj')

def immutable(v):
    if isinstance(v, torch.Tensor):
        return ('tensor', id(v), v._version, v.data_ptr(), tuple(v.shape),
                tuple(v.stride()), v.dtype, v.device, v.requires_grad)
    if v is None or type(v) in (bool, int, float, str) or isinstance(v, (torch.dtype, torch.device)):
        return v
    if type(v) in (tuple, list, torch.Size):
        return tuple(immutable(x) for x in v)
    if type(v) is dict and all(type(k) is str for k in v):
        return tuple((k, immutable(v[k])) for k in sorted(v))
    raise ValueError('Unsupported source metadata; no opaque mutable values')

def contains_tensor(v):
    if isinstance(v, torch.Tensor): return True
    if type(v) is tuple: return any(contains_tensor(x) for x in v)
    return type(v) not in (bool, int, float, str, type(None)) and not isinstance(v, (torch.dtype, torch.device))

def function_identity(f):
    if f is None: return None
    return (id(getattr(f, '__self__', None)), id(getattr(f, '__func__', f)))

def hooks(v):
    return tuple((k, function_identity(f)) for k, f in v.items())

class LayerSource:
    def __init__(self, model, index, tables, runtime_binding, epoch):
        self.index, self.epoch = index, epoch
        self.runtime_binding = runtime_binding
        self.guards, self.fields, self.quantized = [], [], []
        self.reads = 0
        self.guard(partial(getattr, model, 'layers'), model.layers, 'model.layers')
        layer = model.layers[index]
        self.guard(partial(model.layers.__getitem__, index), layer, 'layer')
        self.module(layer)
        self.field(layer, 'hidden_size')
        attention = self.child(layer, 'self_attn', module=True)
        for name in ('layer_idx','head_dim','num_key_value_groups','scaling',
                     'attention_dropout','is_causal','sliding_window'):
            self.field(attention, name)
        cfg = self.child(attention, 'config')
        self.field(cfg, '_attn_implementation')
        for name in ('q_proj','v_proj'):
            lora = self.child(attention, name, module=True)
            for key in ('rank','alpha','scaling'): self.field(lora, key)
            dropout = self.child(lora, 'dropout', module=True)
            for key in ('p','inplace'): self.field(dropout, key)
            for key in ('lora_A','lora_B','base_layer'):
                self.linear(self.child(lora, key, module=True))
        for name in ('k_proj','o_proj'):
            self.linear(self.child(attention, name, module=True))
        for parent, names in ((attention, ('q_norm','k_norm')),
                              (layer, ('input_layernorm','post_attention_layernorm'))):
            for name in names:
                norm = self.child(parent, name, module=True)
                self.field(norm, 'weight'); self.field(norm, 'variance_epsilon')
        mlp = self.child(layer, 'mlp', module=True)
        for name in ('num_experts','top_k','norm_topk_prob'): self.field(mlp, name)
        self.linear(self.child(mlp, 'gate', module=True))
        packed = self.child(mlp, '_packed_nf4')
        if set(packed) != set(ROLES): raise ValueError('Require all three packed roles')
        self.fields.append((lambda p=packed: tuple(sorted(p)), immutable))
        for role in ROLES:
            p = packed[role]
            self.guard(partial(packed.__getitem__, role), p, 'packed.'+role)
            if p.sources is not None: raise ValueError('Require prepared static packed sources')
            for name in ('sources','shape','experts','blocksize','data','absmax','offsets','code'):
                self.field(p, name)
            self.quant(self.child(p, 'scale_state'))
            # Table lookup remains live, so dictionary replacement/removal is not hidden.
            table = tables.get(id(p))
            self.guard(partial(tables.get, id(p)), table, 'scale table '+role)
            if table is not None:
                self.field(table, 'values')
                for name in ('source','payload'):
                    value = self.child(table, name)
                    if type(value) is not tuple or contains_tensor(value):
                        raise ValueError('Scale binding must be immutable host metadata')
                    immutable(value)  # Validate once; a tuple identity guard suffices later.
        self.field(layer, 'gradient_checkpointing')

    def guard(self, getter, expected, name):
        self.guards.append((getter, expected, name))

    def field(self, parent, name, transform=immutable):
        getattr(parent, name)  # Required schema, checked once at configuration.
        self.fields.append((partial(getattr, parent, name), transform))

    def child(self, parent, name, module=False):
        value = getattr(parent, name)
        self.guard(partial(getattr, parent, name), value, name)
        if module: self.module(value)
        return value

    def module(self, module):
        if not isinstance(module, torch.nn.Module): raise ValueError('Require actual module')
        self.field(module, 'training')
        for name in ('forward','__call__','_call_impl','_compiled_call_impl'):
            self.field(module, name, function_identity)
        for name in ('_forward_pre_hooks','_forward_hooks','_backward_pre_hooks','_backward_hooks'):
            self.field(module, name, hooks)

    def quant(self, state, depth=0):
        if state is None: return
        if depth > 1: raise ValueError('Only installed two-level quantization supported')
        for name in ('absmax','code','offset','shape','dtype','blocksize','quant_type','nested'):
            self.field(state, name)
        nested = self.child(state, 'state2')
        self.quant(nested, depth+1)

    def linear(self, module):
        for name in ('weight','bias','in_features','out_features'): self.field(module, name)
        if hasattr(module, 'compute_type_is_set'):
            self.quantized.append(module)
            for name in ('compute_type_is_set','compute_dtype','quant_storage'): self.field(module, name)
            weight = self.child(module, 'weight')
            state = self.child(weight, 'quant_state')
            if state is None: raise ValueError('Prepare quantized weights before registry construction')
            self.quant(state)
            self.quant(self.child(module, 'quant_state'))

    def stable(self, hidden):
        # A few local scalar checks at actual invocation, never at B2 capture hooks.
        return all(m.compute_type_is_set and m.compute_dtype is not None
                   and getattr(m.weight, 'quant_state', None) is not None
                   and (m.bias is None or m.bias.dtype == hidden.dtype)
                   for m in self.quantized)

    def __call__(self):
        self.reads += 1
        for getter, expected, name in self.guards:
            if getter() is not expected:
                raise RuntimeError('Registry object replaced; rebuild at boundary: '+name)
        return (self.epoch, self.index,
                tuple(convert(getter()) for getter, convert in self.fields),
                self.runtime_binding())

class GuardedLayerInvoker:
    def __init__(self, layer, memo, source, original):
        self.memo, self.source, self.original = memo, source, original
        self.bound = LayerCheckpointInvoker(layer, memo, source)
        self.lazy_fallbacks = 0
    def __call__(self, function, *args):
        if self.memo.enabled and (len(args) != 1 or not self.source.stable(args[0])):
            self.lazy_fallbacks += 1
            return self.original(function, *args)
        return self.bound(function, *args)

class RegistryCheckpointPolicy(JointCheckpointPolicy):
    """Isolated extension; selected v57 winner requires LoRA output cache disabled."""
    def __init__(self, model, memo):
        if len(model.layers) != 48: raise ValueError('Require the current 48-layer selection')
        self.registry = None
        self.epoch = 0
        super().__init__(model, memo)

    def configure(self, max_bytes):
        self.memo.boundary()
        if self.memo.enabled and max_bytes:
            raise ValueError('v58 structural candidate requires zero LoRA cache')
        super().configure(max_bytes)
        self.registry = None

    def rebuild(self, tables, runtime_binding):
        self.memo.boundary()
        if self.linear.max_bytes: raise ValueError('Bound route replay requires zero LoRA cache')
        self.epoch += 1
        # The supplier must report live effective settings, not a frozen digest.
        # Full runtime supplier/identity/fullstate/window integration is a separate gate.
        sources = [LayerSource(self.model, i, tables, runtime_binding, self.epoch) for i in range(48)]
        self.invokers = [GuardedLayerInvoker(layer, self.memo, source, original)
            for layer, source, original in zip(self.model.layers, sources, self.originals)]
        self.registry = sources

    def select(self, model, args, kwargs):
        if not self.memo.enabled:
            return super().select(model, args, kwargs)
        if self.registry is None: raise RuntimeError('Rebuild registry after configuration')
        if self.linear.max_bytes: raise RuntimeError('LoRA cache unsupported for this candidate')
        x = kwargs.get('inputs_embeds')
        if x is None: x = kwargs.get('input_ids')
        if x is None and args: x = args[0]
        if x is None: raise ValueError('Missing model input')
        eligible = x.shape[1] <= 1024 and x.shape[0]*x.shape[1] <= 2048
        self._activate(set())
        self.last_eligible, self.last_cached_layers = bool(eligible), []
        for i, (layer, original) in enumerate(zip(model.layers, self.originals)):
            layer.self_attn._qwen_lora_cache_variant = 'plain'
            selected = eligible and layer.gradient_checkpointing and self.first <= i < self.first+self.count
            layer._gradient_checkpointing_func = self.invokers[i] if selected else original
