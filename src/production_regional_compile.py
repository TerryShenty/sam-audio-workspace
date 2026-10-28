"""Reversible regional compiler candidates; no model/state replacement."""
import inspect,types,copy
import torch


class RegionalCompiler:
    def __init__(self, sam_layers, qwen_layers, backend_factory=None):
        self.mode = 'existing'
        self.native_mode = 'existing'
        self.calls = {'sam': 0, 'qwen_attention': 0}
        self.compilations = []
        self.fallback_calls = {'sam':0,'qwen_attention':0}
        self.cache = {}
        self.frame_codes = {}
        self.handles = []
        self.backend_factory = backend_factory
        for layer in sam_layers:
            prior = layer.forward
            # B1Policy -> compiled B2 dispatch -> original SAM implementation.
            b2 = inspect.signature(prior).parameters['_prior'].default
            original = inspect.signature(b2).parameters['_original'].default
            self._attach(layer, prior, original, 'sam')
        for layer in qwen_layers:
            attention = layer.self_attn
            self._attach(attention, attention.forward, attention.forward, 'qwen_attention')

    def _backend(self, mode, region, native_mode, leaf_variant):
        key = (mode, region, native_mode, leaf_variant)
        if key not in self.cache:
            if self.backend_factory:
                compiler = self.backend_factory(mode, region)
            else:
                from torch._inductor import compile as inductor_compile
                compiler = inductor_compile
            def backend(gm, example_inputs):
                def observed(compiled):
                    targets=[str(n.target) for n in gm.graph.nodes if n.op=='call_function']
                    record=dict(mode=mode,region=region,native_mode=native_mode,
                        leaf_variant=leaf_variant,nodes=len(list(gm.graph.nodes)),backend_returned=True,
                        native_rms_nodes=sum('rms_norm' in t for t in targets),executions=0)
                    self.compilations.append(record)
                    def execute(*args,**kwargs):
                        result=compiled(*args,**kwargs)
                        record['executions']+=1
                        return result
                    return execute
                if self.backend_factory:
                    return observed(compiler(gm, example_inputs))
                options = {'triton.cudagraphs': False}
                if mode == 'sam_autotune' and region == 'sam':
                    options['max_autotune'] = True
                if mode in ('sam_dynamic_gemm','both_dynamic_gemm') and region == 'sam':
                    options['max_autotune_gemm'] = True
                compiled = compiler(gm, example_inputs, options=options)
                return observed(compiled)
            self.cache[key] = backend
        return self.cache[key]

    def _attach(self, module, prior, original, region):
        optimized = {}
        def dispatch(*args, **kwargs):
            mode = self.mode
            native_mode = self.native_mode if region == 'sam' else 'existing'
            leaf_variant = getattr(module, '_qwen_nf4_backward_int8_variant', getattr(module, '_qwen_nf4_forward_int8_variant', getattr(module, '_qwen_lora_cache_variant', 'plain'))) if region == 'qwen_attention' else 'plain'
            key = (mode,native_mode,leaf_variant)
            selected = ((region == 'sam' and (native_mode != 'existing' or mode in ('sam_dynamic', 'sam_autotune', 'both_dynamic', 'sam_dynamic_gemm', 'both_dynamic_gemm'))) or
                        (region == 'qwen_attention' and mode in ('qwen_attention_dynamic', 'both_dynamic', 'both_dynamic_gemm')))
            if not selected:
                self.fallback_calls[region] += 1
                return prior(*args, **kwargs)
            if region == 'sam':
                x = args[0] if args else kwargs['x']
                if x.shape[0] not in (1, 2) or x.shape[1] not in (250, 256, 500):
                    self.fallback_calls[region] += 1
                    return original(*args, **kwargs) if native_mode != 'existing' else prior(*args, **kwargs)
            if key not in optimized:
                # Isolate backend/policy guards by mode for BOTH regions. Reuse the
                # isolated frame across same-architecture layers to retain graph reuse.
                assert isinstance(original, types.MethodType)
                f = original.__func__
                frame_key = (f.__code__,region,mode,native_mode,leaf_variant)
                if frame_key not in self.frame_codes:
                    self.frame_codes[frame_key] = f.__code__.replace(
                        co_name=f.__name__+'__'+region+'__'+mode+'__'+native_mode+'__'+leaf_variant)
                code = self.frame_codes[frame_key]
                fn = types.FunctionType(code,f.__globals__,code.co_name,f.__defaults__,f.__closure__)
                fn.__kwdefaults__ = f.__kwdefaults__
                isolated = types.MethodType(fn,original.__self__)
                optimized[key] = torch.compile(
                    isolated, backend=self._backend(mode, region,native_mode,leaf_variant),
                    fullgraph=(region == 'sam'), dynamic=(mode not in ('existing','sam_autotune')))
            self.calls[region] += 1
            return optimized[key](*args, **kwargs)
        module.forward = dispatch
        self.handles.append((module, prior))

    def configure(self, mode, native_mode='existing'):
        assert mode in ('existing', 'sam_dynamic', 'sam_autotune', 'qwen_attention_dynamic', 'both_dynamic', 'sam_dynamic_gemm','both_dynamic_gemm')
        assert native_mode in ('existing','contiguous','native_rms','both')
        self.native_mode = native_mode
        self.mode = mode

    def statistics(self):
        return dict(isolated_frames=len(self.frame_codes),mode=self.mode, native_mode=self.native_mode,fallback_calls=dict(self.fallback_calls), calls=dict(self.calls), compiled_regions=copy.deepcopy(self.compilations),
                    cuda_graphs_enabled=False, fallback_is_not_compiled_acceptance=True)

    def close(self):
        for module, prior in self.handles:
            module.forward = prior
        self.mode = 'existing'
