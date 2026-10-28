"""Boundary-selected original v57 Experts for the matched disabled control.

Both paths use the identical installed backend and four policy objects.
The original Experts forward/backward remain unmodified; no atomic take or
capture call is made by the disabled control. Frozen source seals still apply.
"""
class MatchedRouteControl:
    def __init__(self, stack, control):
        stack._boundary()
        self.stack, self.control = stack, control
        self.candidate = stack.pair.original_experts
        self.original = control.Experts
        self.attributes = ('backend','route_input','route_backward','route_aggregate','pointwise','PAD','FP32_OUTPUT','reduce_routes')
        self.saved = {key:getattr(control,key) for key in self.attributes}
        for key in self.attributes:
            setattr(control,key,getattr(stack.routing,key))
        self.closed = False
    def select(self, enabled):
        if self.closed or type(enabled) is not bool:
            raise RuntimeError('Invalid matched-route selection')
        self.stack._boundary()
        for key in self.attributes:
            if getattr(self.control,key) is not getattr(self.stack.routing,key):
                raise RuntimeError('Matched control runtime diverged: '+key)
        self.stack.pair.original_experts = self.candidate if enabled else self.original
    def close(self):
        self.stack._boundary()
        self.stack.pair.original_experts = self.candidate
        for key,value in self.saved.items():setattr(self.control,key,value)
        self.closed = True
