"""Current-graph activation retention; no persistent language hidden cache."""
ACTIVE=False
BYPASS=0

def install(model):
    assert len(model.layers)==48 and all(hasattr(x,'_gradient_checkpointing_func') for x in model.layers)
    def select(module,args,kwargs):
        x=kwargs.get('inputs_embeds')
        if x is None:x=kwargs.get('input_ids')
        if x is None and args:x=args[0]
        assert x is not None
        tokens=x.shape[0]*x.shape[1]
        bypass=BYPASS if ACTIVE and tokens<=2048 and x.shape[1]<=1024 else 0
        assert bypass in [0,12,24]
        for i,layer in enumerate(module.layers):layer.gradient_checkpointing=i>=bypass
    return model.register_forward_pre_hook(select,with_kwargs=True)
