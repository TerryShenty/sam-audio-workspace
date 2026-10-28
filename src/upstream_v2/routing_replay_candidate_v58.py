"""Isolated atomic-replay routing prototype; no runtime/worker integration.

No truncation, token dropping, changed top-k or hidden-state caching.
CPU routing proof does not establish GPU numerical or speed acceptance.
Never self-installs; not imported by the queued feature/model gate.
"""
import torch
from torch.nn import functional as F
from fused_decode_round7 import decode
from pointwise_moe_round8 import reduce_routes

import dynamic_nf4_static_v1 as backend
from pointwise_recompute_v50 import PointwisePolicy
from route_aggregate_v53 import RouteAggregatePolicy
from route_backward_prepare_v55 import RouteBackwardPolicy
from route_input_prepare_v57 import RouteInputPolicy
route_input = RouteInputPolicy()
route_backward = RouteBackwardPolicy()
route_aggregate = RouteAggregatePolicy()
pointwise = PointwisePolicy()
replay_memo = None  # Explicit isolated-fixture binding only.
PAD = 32
FP32_OUTPUT = False

class Experts(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weights,selected,packed,pad,fp32):
        n,d=x.shape;k=selected.shape[1];e=packed['gate_proj'].experts
        rows=((n*k+e*(pad-1)+pad-1)//pad)*pad
        policies=(pad,fp32,route_input.mode,route_input.block,pointwise.mode,pointwise.block,
                  route_aggregate.mode,route_aggregate.block,route_backward.mode,route_backward.warps)
        supported=(route_input.mode=='fused' and pointwise.mode=='recompute' and
                   x.dtype==torch.bfloat16 and x.is_contiguous() and n<=2048 and
                   1<=k<=8 and 1<=e<=128 and not fp32)
        hit=replay_memo.take(x,selected,packed,rows,policies,pointwise) if replay_memo and supported else None
        if hit is not None:
            token,positions,dst,indices,slots,ends,g,u,y=hit
            # Same ctx state/save sequence; the dead pointwise producer is omitted.
            ctx.pointwise=pointwise
            ctx.pointwise_block=pointwise.block
            pointwise._live.add(ctx)
        else:
            flat=selected.T.reshape(-1);order=torch.argsort(flat,stable=True)
            counts=torch.zeros(e,device=flat.device,dtype=torch.int64)
            counts.scatter_add_(0,flat,torch.ones_like(flat,dtype=torch.int64))
            padded=(counts+pad-1)//pad*pad;ends=padded.cumsum(0).to(torch.int32)
            # Metadata-derived bound is independent of actual routing values.
            # Each expert contributes at most pad-1 padding rows.
            rows=((flat.numel()+e*(pad-1)+pad-1)//pad)*pad
            starts=counts.cumsum(0)-counts
            prepared=route_input.prepare(x,selected,order,counts,padded,ends,starts,rows)
            if prepared is None:
                expert=flat[order]
                dst=ends.long()[expert]-padded[expert]+torch.arange(flat.numel(),device=x.device)-starts[expert]
                token=order.remainder(n);slot=torch.div(order,n,rounding_mode='floor')
                positions=token*k+slot
                inverse=torch.empty_like(positions);inverse.scatter_(0,positions,dst)
                slots=selected.argsort(dim=1)
                indices=inverse.view_as(selected).gather(1,slots)
                routed=x.new_zeros(rows+16,d);routed.index_copy_(0,dst,x.index_select(0,token))
            else:
                routed,token,positions,dst,indices,slots=prepared
            def mm(a,name):
                out=backend.grouped_mm(a.contiguous(),packed[name],ends,rows,True)
                out[rows:]=0
                return out
            g=mm(routed,'gate_proj');u=mm(routed,'up_proj')
            ctx.pointwise = pointwise if pointwise.select(g,u) else None
            if ctx.pointwise is not None:
                m=ctx.pointwise.forward(ctx,g,u)
            else:
                a=F.silu(g);m=a*u
            y=mm(m,'down_proj')
            if replay_memo and supported and prepared is not None and ctx.pointwise is not None:
                replay_memo.capture(x,selected,packed,rows,policies,(token,positions,dst,indices,slots,ends))
        if ctx.pointwise is not None:
            ctx.save_for_backward(weights,positions,token,dst,indices,ends,g,u,y)
        else:
            ctx.save_for_backward(weights,positions,token,dst,indices,ends,g,u,a,y)
        ctx.packed,ctx.rows,ctx.fp32=packed,rows,fp32
        ctx.route_aggregate = route_aggregate.attach(ctx)
        ctx.route_backward = route_backward.attach(ctx)
        out = ctx.route_aggregate.forward(ctx,y,weights,indices,slots) if ctx.route_aggregate else None
        if out is None:
            w=weights.new_zeros(rows+16,1)
            w.index_copy_(0,dst,weights.flatten().index_select(0,positions).unsqueeze(-1))
            out=reduce_routes(y*w,indices,False)
        return out

    @staticmethod
    def backward(ctx,upstream):
        try:
            if ctx.pointwise is not None:
                weights,positions,token,dst,indices,ends,g,u,y=ctx.saved_tensors
            else:
                weights,positions,token,dst,indices,ends,g,u,a,y=ctx.saved_tensors
            with torch.autocast('cuda',enabled=False):
                prepared=ctx.route_backward.prepare(ctx,upstream,y,weights,token,positions,dst) if ctx.route_backward else None
                if prepared is None:
                    dy=upstream.index_select(0,token)
                    dw=(dy*y.index_select(0,dst)).sum(-1)
                    gw=torch.zeros_like(weights);gw.flatten().index_copy_(0,positions,dw)
                    w=weights.flatten().index_select(0,positions).unsqueeze(-1)
                    weighted=torch.zeros_like(y);weighted.index_copy_(0,dst,dy*w)
                else:
                    weighted,gw=prepared
                def mm(v,name):
                    out=backend.grouped_mm(v.contiguous(),ctx.packed[name],ends,ctx.rows,False)
                    out[ctx.rows:]=0
                    return out
                dm=mm(weighted,'down_proj')
                if ctx.pointwise is not None:
                    dg,du=ctx.pointwise.backward(ctx,dm,u,g)
                else:
                    dg=torch.ops.aten.silu_backward.default(dm*u,g);du=dm*a
                gate_dx=mm(dg,'gate_proj');up_dx=mm(du,'up_proj')
                gx=ctx.route_aggregate.backward(ctx,gate_dx,up_dx,indices) if ctx.route_aggregate else None
                if gx is None:
                    gx=reduce_routes(gate_dx+up_dx,indices,True)
            return gx,gw,None,None,None,None
        finally:
            if ctx.route_backward is not None:
                ctx.route_backward.release(ctx)
            if ctx.route_aggregate is not None:
                ctx.route_aggregate.release(ctx)
            if ctx.pointwise is not None:
                ctx.pointwise.release(ctx)

def forward(self,hidden_states):
    shape=hidden_states.shape;x=hidden_states.reshape(-1,shape[-1])
    logits=self.gate(x);weights=F.softmax(logits,dim=1,dtype=torch.float32)
    weights,selected=torch.topk(weights,self.top_k,dim=-1)
    if self.norm_topk_prob:weights=weights/weights.sum(-1,keepdim=True)
    out=Experts.apply(x,weights.to(x.dtype),selected,self._packed_nf4,PAD,FP32_OUTPUT)
    return out.view(shape),logits
