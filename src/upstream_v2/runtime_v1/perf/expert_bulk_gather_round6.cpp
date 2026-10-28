#include <torch/extension.h>
#include <vector>

// Only top-k expert scatter has unique token indices within each expert.
// Avoid the generic repeated-index sorting path for those proven-unique writes.
// Preserve the process setting after each individual scatter, including errors.
struct UniqueScatterGuard {
  bool enabled; bool warn;
  UniqueScatterGuard() : enabled(at::globalContext().deterministicAlgorithms()),
      warn(at::globalContext().deterministicAlgorithmsWarnOnly()) {
    at::globalContext().setDeterministicAlgorithms(false, warn);
  }
  ~UniqueScatterGuard() { at::globalContext().setDeterministicAlgorithms(enabled, warn); }
};


std::vector<torch::Tensor> expert_loop(torch::Tensor x, torch::Tensor weights,
                         torch::Tensor ks, torch::Tensor tokens,
                         std::vector<int64_t> counts,
                         torch::Tensor gates, torch::Tensor ups,
                         torch::Tensor downs) {
  auto out = torch::zeros_like(x);
  std::vector<torch::Tensor> saved{out};
  auto routed = x.index_select(0, tokens);
  auto route_weights = weights.flatten().index_select(0, tokens * weights.size(1) + ks).unsqueeze(-1);
  int64_t offset = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(counts.size()); ++i) {
    const auto count = counts[i];
    if (!count) continue;
    auto ix = tokens.narrow(0, offset, count);
    auto k = ks.narrow(0, offset, count);
    auto cur = routed.narrow(0, offset, count);
    auto w = route_weights.narrow(0, offset, count);
    offset += count;
    auto g = at::linear(cur, gates.select(0, i));
    auto u = at::linear(cur, ups.select(0, i));
    auto a = at::silu(g);
    auto mid = a * u;
    auto y = at::linear(mid, downs.select(0, i));
    { auto update = y * w; UniqueScatterGuard guard; out.index_add_(0, ix, update); }
    saved.insert(saved.end(), {g,u,a,y});
  }
  return saved;
}


std::vector<torch::Tensor> expert_backward(torch::Tensor x, torch::Tensor weights,
                         torch::Tensor ks, torch::Tensor tokens,
                         std::vector<int64_t> counts,
                         torch::Tensor gates, torch::Tensor ups,
                         torch::Tensor downs, torch::Tensor upstream, bool gate_first, std::vector<torch::Tensor> saved) {
  auto gx = torch::zeros_like(x);
  auto gw = torch::zeros_like(weights);
  auto route_positions = tokens * weights.size(1) + ks;
  auto route_weights = weights.flatten().index_select(0, route_positions).unsqueeze(-1);
  auto route_upstream = upstream.index_select(0, tokens);
  auto route_dw = torch::empty({tokens.numel()}, weights.options());
  int64_t offset = tokens.numel();
  int64_t saved_offset = saved.size();
  for (int64_t i = static_cast<int64_t>(counts.size())-1; i >= 0; --i) {
    const auto count = counts[i];
    offset -= count;
    if (!count) continue;
    auto ix = tokens.narrow(0, offset, count);
    auto k = ks.narrow(0, offset, count);
    auto gate = gates.select(0, i);
    auto up = ups.select(0, i);
    auto down = downs.select(0, i);
    saved_offset -= 4;
    auto g = saved[saved_offset];
    auto u = saved[saved_offset+1];
    auto a = saved[saved_offset+2];
    auto y = saved[saved_offset+3];
    auto w = route_weights.narrow(0, offset, count);
    auto dy = route_upstream.narrow(0, offset, count);
    auto dw = at::sum(dy * y, at::IntArrayRef{-1}, false);
    route_dw.narrow(0, offset, count).copy_(dw);
    auto dm = at::mm(dy * w, down);
    auto dg = at::silu_backward(dm * u, g);
    auto du = dm * a;
    auto dcur = gate_first ? at::mm(dg, gate) + at::mm(du, up) : at::mm(du, up) + at::mm(dg, gate);
    { UniqueScatterGuard guard; gx.index_add_(0, ix, dcur); }
  }
  gw.flatten().index_copy_(0, route_positions, route_dw);
  return {gx, gw};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expert_loop", &expert_loop);
  m.def("expert_backward", &expert_backward);
}
