#include <torch/extension.h>
#include <vector>

// Keep each expert GEMM's dimensions and operation unchanged. Only pointwise
// work is batched across experts. Ordered BF16 route reduction is done outside.
std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor weights,
    torch::Tensor positions, torch::Tensor tokens, std::vector<int64_t> counts,
    torch::Tensor gates, torch::Tensor ups, torch::Tensor downs) {
  auto routed = x.index_select(0, tokens);
  auto w = weights.flatten().index_select(0, positions).unsqueeze(-1);
  std::vector<torch::Tensor> gs, us, ys;
  int64_t offset = 0;
  for (int64_t i=0; i<(int64_t)counts.size(); ++i) {
    auto n=counts[i]; if (!n) continue;
    auto cur=routed.narrow(0,offset,n); offset+=n;
    gs.push_back(at::linear(cur,gates.select(0,i)));
    us.push_back(at::linear(cur,ups.select(0,i)));
  }
  auto g=at::cat(gs,0), u=at::cat(us,0);
  gs.clear(); us.clear();
  auto a=at::silu(g), mid=a*u;
  offset=0;
  for (int64_t i=0; i<(int64_t)counts.size(); ++i) {
    auto n=counts[i]; if (!n) continue;
    ys.push_back(at::linear(mid.narrow(0,offset,n),downs.select(0,i)));
    offset+=n;
  }
  auto y=at::cat(ys,0);
  return {y*w,g,u,a,y};
}

std::vector<torch::Tensor> backward(torch::Tensor weights, torch::Tensor positions,
    torch::Tensor tokens, std::vector<int64_t> counts, torch::Tensor gates,
    torch::Tensor ups, torch::Tensor downs, torch::Tensor upstream,
    torch::Tensor g, torch::Tensor u, torch::Tensor a, torch::Tensor y) {
  auto w=weights.flatten().index_select(0,positions).unsqueeze(-1);
  auto dy=upstream.index_select(0,tokens);
  auto dw=at::sum(dy*y,at::IntArrayRef{-1},false);
  auto weighted=dy*w;
  std::vector<torch::Tensor> dms, dxs;
  int64_t offset=0;
  for (int64_t i=0; i<(int64_t)counts.size(); ++i) {
    auto n=counts[i]; if (!n) continue;
    dms.push_back(at::mm(weighted.narrow(0,offset,n),downs.select(0,i)));
    offset+=n;
  }
  auto dm=at::cat(dms,0); dms.clear();
  auto dg=at::silu_backward(dm*u,g), du=dm*a;
  offset=0;
  for (int64_t i=0; i<(int64_t)counts.size(); ++i) {
    auto n=counts[i]; if (!n) continue;
    auto dx=at::mm(dg.narrow(0,offset,n),gates.select(0,i))
           +at::mm(du.narrow(0,offset,n),ups.select(0,i));
    dxs.push_back(dx); offset+=n;
  }
  auto gw=at::zeros_like(weights);
  gw.flatten().index_copy_(0,positions,dw);
  return {at::cat(dxs,0),gw};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("forward",&forward); m.def("backward",&backward);
}
