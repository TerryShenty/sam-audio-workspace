"""Optional compiled SAM blocks; original parameters and checkpoint names preserved."""
import torch
import torch._inductor
ENABLED=True

def enable(transformer,expected_count=22):
    assert len(transformer.layers)==expected_count
    assert 'triton.cudagraphs' in torch._inductor.list_options()
    for layer in transformer.layers:
        assert not hasattr(layer,'compiled_sam_calls')
        original=layer.forward
        optimized=torch.compile(original,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        def dispatch(x,cross_x,t,padding_mask,memory_padding_mask,rope=None,_original=original,_optimized=optimized,_layer=layer):
            if ENABLED and x.shape[0]==2 and x.shape[1] in (250,500):
                _layer.compiled_sam_calls+=1
                return _optimized(x,cross_x,t,padding_mask,memory_padding_mask,rope)
            return _original(x,cross_x,t,padding_mask,memory_padding_mask,rope)
        layer.compiled_sam_calls=0
        layer.forward=dispatch

def calls(transformer):return sum(x.compiled_sam_calls for x in transformer.layers)

def ema_update(ema,parameters):
    with torch.no_grad():torch._foreach_lerp_(list(ema.values()),[parameters[n].detach().float() for n in ema],.001)
