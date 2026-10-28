"""Local-only bounded attention origin binding; not installed in a worker.

The installer must verify the pinned installed source hashes first. Only the
known deprecate_kwarg -> raw forward chain is accepted. Runtime reads use the
preselected closure cells; they do not recursively inspect arbitrary closures.
This guards configured callable replacement/code/default changes, not arbitrary
monkey patches of every helper/global in the Python or Torch environment.
"""
import types
import weakref


def host_value(value):
    if value is None or type(value) in (bool, int, float, str):
        return (type(value).__name__, value)
    if type(value) is tuple:
        return ('tuple', tuple(host_value(v) for v in value))
    if type(value) is dict and all(type(k) is str for k in value):
        return ('dict', tuple((k, host_value(value[k])) for k in sorted(value)))
    raise ValueError('Origin metadata must contain only immutable host values')


def function_stamp(function):
    if type(function) is not types.FunctionType:
        raise ValueError('Only verified Python functions are supported')
    return (id(function), id(function.__code__), id(function.__globals__),
            host_value(function.__defaults__), host_value(function.__kwdefaults__))


def child_code(parent, name):
    matches = [c for c in parent.co_consts
               if isinstance(c, types.CodeType) and c.co_name == name]
    if len(matches) != 1:
        raise ValueError('Unexpected installed decorator layout: '+name)
    return matches[0]


class AttentionOrigin:
    """Construct before RegionalCompiler._attach, publish its exact dispatch.

    deprecator must be the verified installed deprecate_kwarg callable; namespace
    must be the verified modeling module dictionary. This object does not unwrap
    the compiler-created FunctionType, which omits __wrapped__; it binds the
    original closure that the compiler actually copies. Publication does not
    certify full worker readiness or checkpoint restoration.
    """
    def __init__(self, module, bound, deprecator, namespace,
                 expected_qualname='Qwen3OmniMoeThinkerTextAttention.forward'):
        if type(bound) is not types.MethodType or bound.__self__ is not module:
            raise ValueError('Require the original bound attention method')
        if type(namespace) is not dict:
            raise ValueError('Require the actual modeling namespace')
        function_stamp(deprecator)
        outer = bound.__func__
        wrapper_code = child_code(child_code(deprecator.__code__, 'wrapper'), 'wrapped_func')
        if outer.__code__ is not wrapper_code:
            raise ValueError('Unknown attention wrapper; do not recursively unwrap')
        cells = dict(zip(outer.__code__.co_freevars, outer.__closure__ or ()))
        if 'func' not in cells:
            raise ValueError('Missing actual decorator target cell')
        raw = cells['func'].cell_contents
        function_stamp(raw)
        if (getattr(outer, '__wrapped__', None) is not raw or
                raw.__globals__ is not namespace or raw.__qualname__ != expected_qualname):
            raise ValueError('Decorator closure, wrapped target and namespace disagree')
        # Known decorator captures only host scalars in addition to func.
        scalar_cells = tuple((key, cell) for key, cell in sorted(cells.items()) if key != 'func')
        for _, cell in scalar_cells:
            host_value(cell.cell_contents)
        self.module = weakref.ref(module)
        self.outer, self.raw, self.namespace = outer, raw, namespace
        self.target_cell, self.scalar_cells = cells['func'], scalar_cells
        self.outer_code, self.raw_code = outer.__code__, raw.__code__
        self.compiler = self.dispatch = None
        self.published = self.closed = False

    def publish(self, compiler, dispatch):
        module = self.module()
        if self.closed or self.published or module is None:
            raise RuntimeError('Origin publication requires a fresh live installation')
        if type(dispatch) is not types.FunctionType or module.forward is not dispatch:
            raise ValueError('Publish the dispatch actually installed on this attention')
        # No optimized/cache/frame/counter objects are retained or scanned.
        self.compiler, self.dispatch = weakref.ref(compiler), dispatch
        self.published = True

    def __call__(self):
        if self.closed or not self.published:
            raise RuntimeError('Attention origin is not published')
        module, compiler = self.module(), self.compiler()
        if module is None or compiler is None or module.forward is not self.dispatch:
            raise RuntimeError('Attention/compiler installation was replaced or released')
        if (self.outer.__code__ is not self.outer_code or self.raw.__code__ is not self.raw_code
                or self.target_cell.cell_contents is not self.raw
                or getattr(self.outer, '__wrapped__', None) is not self.raw):
            raise RuntimeError('Verified attention wrapper chain changed')
        variant = getattr(module, '_qwen_lora_cache_variant', 'plain')
        if variant != 'plain':
            raise RuntimeError('This candidate requires the plain LoRA leaf path')
        name = module.config._attn_implementation
        if type(name) is not str:
            raise ValueError('Attention backend name must be a string')
        rope = self.namespace['apply_rotary_pos_emb']
        if name == 'eager':
            registry_stamp = None
            selected = self.namespace['eager_attention_forward']
        else:
            # Same live parent and selected lookup as the actual raw forward.
            # GeneralInterface implementation still requires installed evidence.
            registry = self.namespace['ALL_ATTENTION_FUNCTIONS']
            registry_stamp = (id(registry), id(type(registry)),
                              function_stamp(type(registry).__getitem__))
            selected = registry[name]
        factory = compiler.backend_factory
        factory_stamp = None if factory is None else function_stamp(factory)
        return ('attention_origin_v58b_1', id(module), id(compiler),
                function_stamp(self.dispatch), function_stamp(self.outer), function_stamp(self.raw),
                tuple((key, host_value(cell.cell_contents)) for key, cell in self.scalar_cells),
                host_value(compiler.mode), host_value(compiler.native_mode), variant,
                factory_stamp, name, registry_stamp, function_stamp(rope), function_stamp(selected))

    def close(self):
        # Existing scopes must be released by the outer transition boundary.
        self.closed = True
        self.compiler = self.dispatch = None
        self.outer = self.raw = self.namespace = None
        self.target_cell = self.scalar_cells = None
