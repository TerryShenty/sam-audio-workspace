"""Candidate only: batched route discovery, unchanged expert math and order.

Not installed into the running model. Stable sorting of top-k-major routes
preserves each expert's torch.where ordering in transformers 4.57.6.
"""
import torch
from torch.nn import functional as F


def route_indices(selected_experts, num_experts):
    flat = selected_experts.transpose(0, 1).reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts).cpu().tolist()
    length = selected_experts.shape[0]
    token_order = order.remainder(length)
    topk_order = torch.div(order, length, rounding_mode='floor')
    return counts, topk_order, token_order


def forward(self, hidden_states):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    counts, topk_order, token_order = route_indices(selected_experts, self.num_experts)
    offset = 0
    for expert_idx, count in enumerate(counts):
        if count == 0:
            continue
        idx = topk_order[offset:offset + count]
        top_x = token_order[offset:offset + count]
        offset += count
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits
