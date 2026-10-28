"""Explicit installation order and unsupported-shape dispatch for isolated v23."""
from row_tile_candidate_v23 import install_backend,install_routing

def install(static_backend,static_routing,choose_config):
    undo_backend=install_backend(static_backend,choose_config)
    undo_routing=install_routing(static_routing,choose_config)
    return undo_backend,undo_routing

def dispatch(original,static_routing):
    def forward(block,hidden):
        if hidden.ndim==3 and hidden.shape[0]==2 and 0<hidden.shape[1]<=2048:
            return static_routing.forward(block,hidden)
        raise ValueError('PRODUCTION_STATIC_DISPATCH_REQUIRES_B2_T1_TO_2048')
    return forward
