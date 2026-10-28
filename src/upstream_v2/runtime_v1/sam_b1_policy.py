"""Reversible B1 SAM compilation; all checkpoint parameter identities unchanged."""
import inspect
import torch

class B1Policy:
    def __init__(self,transformer):
        self.layers=list(transformer.layers);assert len(self.layers)==22
        self.originals=[layer.forward for layer in self.layers]
        self.enabled=False;self.calls=0;self.shapes=set()
        for layer,prior in zip(self.layers,self.originals):
            signature=inspect.signature(prior)
            assert '_original' in signature.parameters
            original=signature.parameters['_original'].default
            assert callable(original)
            optimized=torch.compile(original,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
            def dispatch(x,cross_x,t,padding_mask,memory_padding_mask,rope=None,_prior=prior,_optimized=optimized):
                if self.enabled and x.shape[0]==1 and x.shape[1] in (250,500):
                    self.calls+=1;self.shapes.add(tuple(x.shape))
                    return _optimized(x,cross_x,t,padding_mask,memory_padding_mask,rope)
                return _prior(x,cross_x,t,padding_mask,memory_padding_mask,rope)
            layer.forward=dispatch
    def set(self,enabled):
        assert isinstance(enabled,bool);self.enabled=enabled
    def close(self):
        for layer,original in zip(self.layers,self.originals):layer.forward=original
        self.enabled=False
