"""Isolated candidate only; target GPU memory/full-update gates still required.

Keep accepted ret23 and registry invokers for actual real-data token bounds.
Outside the bound, reject before changing layer policy instead of falling back.
"""

def input_shape(args,kwargs,max_tokens,*,allow_capture=False):
    x=kwargs.get('inputs_embeds')
    if x is None:x=kwargs.get('input_ids')
    if x is None and args:x=args[0]
    if x is None:raise ValueError('Missing model input')
    if not (x.shape[0] in ((1,2) if allow_capture else (2,)) and 0<x.shape[1]<=max_tokens):
        raise ValueError('Input outside qualified per-rank B2 token bound')
    return x

def guard_model_body(model,max_tokens):
    """B1 input-capture prehooks may raise before the actual B2-only body."""
    original=model.forward
    def guarded(*args,**kwargs):
        input_shape(args,kwargs,max_tokens)
        return original(*args,**kwargs)
    model.forward=guarded
    return original

def policy_classes():
    from functools import partial
    from torch.utils.checkpoint import checkpoint
    from retention_runtime_v52 import Retention
    from bp_registry import RuntimeRegistry

    class ProductionRetention(Retention):
        def __init__(self,model,*,max_tokens):
            assert type(max_tokens) is int and max_tokens>=1024
            self.max_tokens=max_tokens
            super().__init__(model)
        def select(self,model,args,kwargs):
            input_shape(args,kwargs,self.max_tokens,allow_capture=True)
            assert self.count==23,'Production requires accepted ret23'
            for i,layer in enumerate(model.layers):layer.gradient_checkpointing=i>=23
            self.selection=(23,0)
            self._increment(self.attempts,*self.selection)

    class ProductionRegistry(RuntimeRegistry):
        def __init__(self,*args,max_tokens,**kwargs):
            assert type(max_tokens) is int and max_tokens>=1024
            self.max_tokens=max_tokens
            super().__init__(*args,**kwargs)
        def select(self,model,args,kwargs):
            self.require_step()
            input_shape(args,kwargs,self.max_tokens,allow_capture=True)
            # enabled controls rejected atomic ROUTE replay, not ordinary NF4 memo.
            assert not self.memo.enabled,'Atomic route replay is not accepted'
            assert self.memo.max_bytes==3623878656,'Accepted ordinary NF4 memo budget required'
            assert self.memo.consume_once and set(self.memo.roles)=={'gate_proj','up_proj','down_proj'}
            assert self.linear.max_bytes==0,'Accepted LoRA intermediate cache is disabled'
            self._activate(set())
            self.last_eligible,self.last_cached_layers=True,[]
            for i,(layer,original) in enumerate(zip(model.layers,self.originals)):
                layer.self_attn._qwen_lora_cache_variant='plain'
                selected=layer.gradient_checkpointing and self.first<=i<self.first+self.count
                layer._gradient_checkpointing_func=(partial(checkpoint,use_reentrant=False,
                    preserve_rng_state=True,context_fn=self.memo.contexts) if selected else original)

    return ProductionRetention,ProductionRegistry
