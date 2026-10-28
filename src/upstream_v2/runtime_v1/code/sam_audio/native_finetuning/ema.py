"""Optional FP32 EMA of trainable parameters, including an auxiliary head."""
import torch


class TrainableEMA:
    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0, 1)')
        self.decay, self.updates = float(decay), 0
        self.shadow = {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        current = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if current.keys() != self.shadow.keys():
            raise ValueError('EMA parameter set changed')
        for n, p in current.items():
            self.shadow[n].lerp_(p.detach().float(), 1 - self.decay)
        self.updates += 1

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates,
                    shadow={n: p.detach().cpu().clone() for n, p in self.shadow.items()})

    def load_state_dict(self, state):
        if state['decay'] != self.decay or state['shadow'].keys() != self.shadow.keys():
            raise ValueError('EMA configuration or parameter set changed')
        for n, value in state['shadow'].items():
            self.shadow[n].copy_(value.to(self.shadow[n]))
        self.updates = int(state['updates'])
