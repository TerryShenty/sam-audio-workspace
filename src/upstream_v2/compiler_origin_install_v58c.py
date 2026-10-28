"""Controlled origin publication around the frozen RegionalCompiler._attach.

This module imports no Torch/HF/training implementation. The caller must supply
the sealed RegionalCompiler class and already verified installed decorator and
modeling namespace. It creates a new compiler; it never patches a live worker.
Installed HF/GeneralInterface and full training integration remain required.
"""
import types

from attention_origin_binding_v58b import AttentionOrigin, child_code


def same_bound(left, right):
    return (type(left) is types.MethodType and type(right) is types.MethodType
            and left.__self__ is right.__self__ and left.__func__ is right.__func__)


def verify_attachment(attach_function, compiler, module, prior, original):
    """Read the known dispatch closure once, before publishing its origin.

    The optimized mapping must be fresh at installation. Later normal cache
    growth is intentionally not part of an origin signature. This does not
    protect arbitrary external writes to closure cells after installation.
    """
    if type(attach_function) is not types.FunctionType:
        raise TypeError('Require the sealed Python attachment function')
    if not same_bound(prior, original) or original.__self__ is not module:
        raise ValueError('Qwen installation requires its unmodified bound method')
    dispatch = module.forward
    expected_code = child_code(attach_function.__code__, 'dispatch')
    if type(dispatch) is not types.FunctionType or dispatch.__code__ is not expected_code:
        raise ValueError('Installed dispatch did not come from the sealed attachment')
    cells = dict(zip(dispatch.__code__.co_freevars, dispatch.__closure__ or ()))
    if set(cells) != {'self', 'module', 'original', 'prior', 'region', 'optimized'}:
        raise ValueError('Unexpected compiler dispatch closure layout')
    values = {key: cell.cell_contents for key, cell in cells.items()}
    if (values['self'] is not compiler or values['module'] is not module
            or values['region'] != 'qwen_attention'
            or not same_bound(values['original'], original)
            or not same_bound(values['prior'], prior)):
        raise ValueError('Compiler dispatch closure points to a different origin')
    if type(values['optimized']) is not dict or values['optimized']:
        raise ValueError('Origin publication must precede compilation or execution')
    if (not compiler.handles or compiler.handles[-1][0] is not module
            or not same_bound(compiler.handles[-1][1], prior)):
        raise ValueError('Compiler rollback handle does not match the installation')
    return dispatch


def origin_compiler_class(sealed_class):
    """Make an isolated subclass; source/seal verification is the caller's gate.

    The base class remains unchanged. Publication follows its actual _attach,
    including the closure that will later feed FunctionType. Construction
    failure restores every forward installed by this new compiler and closes
    all origins, rather than leaving a partially usable instance on modules.
    """
    attach_function = sealed_class._attach
    if type(attach_function) is not types.FunctionType:
        raise TypeError('Unsupported RegionalCompiler attachment')

    class OriginCompiler(sealed_class):
        def __init__(self, sam_layers, qwen_layers, deprecator, namespace,
                     backend_factory=None):
            # Store immutable installation inputs, not runtime cache snapshots.
            self.attention_origins = {}
            self._origin_deprecator = deprecator
            self._origin_namespace = namespace
            self.origins_ready = False
            try:
                super().__init__(sam_layers, qwen_layers, backend_factory=backend_factory)
                if len(self.attention_origins) != len(qwen_layers):
                    raise RuntimeError('Not every Qwen attention has one origin')
                self.origins_ready = True
            except Exception:
                self._discard_origins()
                # Base construction initializes handles before attaching layers.
                # Only forwards installed by this new compiler are represented.
                for module, prior in getattr(self, 'handles', ()):
                    module.forward = prior
                self.handles = []
                raise
            finally:
                self._origin_deprecator = self._origin_namespace = None

        def _attach(self, module, prior, original, region):
            if sealed_class._attach is not attach_function:
                raise RuntimeError('Sealed compiler attachment was replaced')
            if region != 'qwen_attention':
                return attach_function(self, module, prior, original, region)
            if id(module) in self.attention_origins:
                raise ValueError('Duplicate attention installation')
            origin = AttentionOrigin(module, original, self._origin_deprecator,
                                     self._origin_namespace)
            try:
                attach_function(self, module, prior, original, region)
                dispatch = verify_attachment(attach_function, self, module, prior, original)
                origin.publish(self, dispatch)
                self.attention_origins[id(module)] = origin
            except Exception:
                origin.close()
                raise

        def _discard_origins(self):
            self.origins_ready = False
            for origin in self.attention_origins.values():
                origin.close()
            self.attention_origins.clear()

        def origin_for(self, module):
            if not self.origins_ready:
                raise RuntimeError('Compiler origins are not fully installed')
            return self.attention_origins[id(module)]

        def close(self):
            # The worker must first prove all checkpoint/runtime graphs released.
            # Closing a compiler is not itself a graph-free transition check.
            self._discard_origins()
            super().close()
            self.handles = []

    return OriginCompiler
