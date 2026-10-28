#include <torch/extension.h>
#include <vector>

// Research only. Keep K and per-expert M unchanged. Mode 3/4 also merges
// independent gate/up output columns; this still requires numerical validation.
std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor weights,
 torch::Tensor positions, torch::Tensor tokens, std::vector<int64_t> counts,
 torch::Tensor gates, torch::Tensor ups, torch::Tensor downs, int64_t mode) {
 const bool pre = mode==1 || mode==4;
 const bool merge = mode==3 || mode==4;
 auto routed=x.index_select(0,tokens);
 auto w=weights.flatten().index_select(0,positions).unsqueeze(-1);
 auto rows=tokens.numel(), width=gates.size(1);
 auto g=x.new_empty({rows,width}), u=x.new_empty({rows,width});
 auto gu=merge ? at::cat({gates,ups},1) : gates;
 std::vector<torch::Tensor> gs,us,ys;
 int64_t offset=0;
 for(int64_t i=0;i<(int64_t)counts.size();++i){
  auto n=counts[i]; if(!n)continue;
  auto cur=routed.narrow(0,offset,n);
  if(merge){
   auto both=at::linear(cur,gu.select(0,i));
   gs.push_back(both.narrow(1,0,width));us.push_back(both.narrow(1,width,width));
  }else if(pre){
   auto go=g.narrow(0,offset,n),uo=u.narrow(0,offset,n);
   at::mm_out(go,cur,gates.select(0,i).t());at::mm_out(uo,cur,ups.select(0,i).t());
  }else{gs.push_back(at::linear(cur,gates.select(0,i)));us.push_back(at::linear(cur,ups.select(0,i)));}
  offset+=n;
 }
 if(!pre || merge){g=at::cat(gs,0);u=at::cat(us,0);}
 gs.clear();us.clear();
 auto a=at::silu(g), mid=a*u;
 auto y=x.new_empty({rows,x.size(1)});offset=0;
 for(int64_t i=0;i<(int64_t)counts.size();++i){
  auto n=counts[i];if(!n)continue;
  if(pre){auto dst=y.narrow(0,offset,n);at::mm_out(dst,mid.narrow(0,offset,n),downs.select(0,i).t());}
  else ys.push_back(at::linear(mid.narrow(0,offset,n),downs.select(0,i)));
  offset+=n;
 }
 if(!pre)y=at::cat(ys,0);
 return {y*w,g,u,a,y};
}
std::vector<torch::Tensor> backward(torch::Tensor weights,torch::Tensor positions,
 torch::Tensor tokens,std::vector<int64_t> counts,torch::Tensor gates,
 torch::Tensor ups,torch::Tensor downs,torch::Tensor upstream,
 torch::Tensor g,torch::Tensor u,torch::Tensor a,torch::Tensor y,int64_t mode){
 const bool pre=mode==1 || mode==4, bulk=mode==2 || mode==4;
 auto w=weights.flatten().index_select(0,positions).unsqueeze(-1);
 auto dy=upstream.index_select(0,tokens);
 auto dw=at::sum(dy*y,at::IntArrayRef{-1},false),weighted=dy*w;
 auto dm=at::empty_like(g);
 std::vector<torch::Tensor>dms,dxs,dxgs,dxus;int64_t offset=0;
 for(int64_t i=0;i<(int64_t)counts.size();++i){
  auto n=counts[i];if(!n)continue;
  if(pre){auto dst=dm.narrow(0,offset,n);at::mm_out(dst,weighted.narrow(0,offset,n),downs.select(0,i));}
  else dms.push_back(at::mm(weighted.narrow(0,offset,n),downs.select(0,i)));
  offset+=n;
 }
 if(!pre)dm=at::cat(dms,0);dms.clear();
 auto dg=at::silu_backward(dm*u,g),du=dm*a;
 auto dxg=at::empty_like(dy),dxu=at::empty_like(dy);offset=0;
 for(int64_t i=0;i<(int64_t)counts.size();++i){
  auto n=counts[i];if(!n)continue;
  if(pre){
   auto ga=dxg.narrow(0,offset,n),ua=dxu.narrow(0,offset,n);
   at::mm_out(ga,dg.narrow(0,offset,n),gates.select(0,i));
   at::mm_out(ua,du.narrow(0,offset,n),ups.select(0,i));
   if(!bulk)ga.add_(ua);
  }else{
   auto ga=at::mm(dg.narrow(0,offset,n),gates.select(0,i));
   auto ua=at::mm(du.narrow(0,offset,n),ups.select(0,i));
   if(bulk){dxgs.push_back(ga);dxus.push_back(ua);}else dxs.push_back(ga+ua);
  }
  offset+=n;
 }
 torch::Tensor dx;
 if(pre)dx=bulk ? dxg+dxu : dxg;
 else dx=bulk ? at::cat(dxgs,0)+at::cat(dxus,0) : at::cat(dxs,0);
 auto gw=at::zeros_like(weights);gw.flatten().index_copy_(0,positions,dw);
 return {dx,gw};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);}
