"""Bounded real-update A/B; restores the full saved state after both paths."""
import gc,time,statistics
import torch
import torch.distributed as dist
from common import write,epoch_batches
from bridge import tree


def validate(ddp,joint,opt,sched,ema,items,store,out,rank,device,step,epoch,start_batch,restore,mode_select,synced):
    begin=time.monotonic();params={n:p for n,p in joint.named_parameters() if p.requires_grad}
    batches=list(epoch_batches(len(items),epoch))[start_batch:start_batch+8];assert len(batches)==8
    rows={};hashes={};joint.profile_step=False;joint.vlm.audit_flow_gradient=True
    from evaluate import rng_get,rng_set
    import sam_audio.native_finetuning.objective as objective
    original_flow=objective.make_flow_matching_state
    flow_tapes=[];rng_tapes=[];flow_matches=[]
    for variant in ['baseline','candidate']:
        opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        restored=restore(out/'latest.pt',joint,opt,sched,ema,rank,device);assert restored==(step,epoch,start_batch)
        mode_select(variant=='candidate');values=[]
        for offset,batch in enumerate(batches):
            expired=torch.tensor(int(time.monotonic()-begin>240),device=device);dist.all_reduce(expired,op=dist.ReduceOp.MAX);assert not bool(expired),'Bounded update comparison exceeded240seconds'
            assert all(keep for ix,keep in batch),'Compare complete global8 updates only'
            samples=[];args=[];clean=[]
            for micro in range(2):
                ix,keep=batch[micro*4+rank];item=items[ix];a,c=store.assemble(item['refs'])
                samples.append(item);args.append(tree(a,lambda t:t.to(device)));clean.append(c.to(device))
            opt.zero_grad(set_to_none=True)
            if variant=='baseline':rng_tapes.append(rng_get());flow_tapes.append([])
            else:rng_set(rng_tapes[offset])
            flow_index=[0]
            def paired_flow(clean_latent,**kwargs):
                i=flow_index[0];flow_index[0]+=1
                if variant=='baseline':
                    state=original_flow(clean_latent,**kwargs)
                    flow_tapes[offset].append((clean_latent.detach().clone(),state))
                    return state
                reference,state=flow_tapes[offset][i]
                assert torch.equal(clean_latent,reference),'Paired flow clean target differs'
                return state
            objective.make_flow_matching_state=paired_flow
            torch.cuda.synchronize();t=time.perf_counter()
            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms,tokens=ddp(samples,args,clean,[.5,.5])
            finally:objective.make_flow_matching_state=original_flow
            assert flow_index[0]==2
            if variant=='candidate':flow_matches.append(True)
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True)
            flow=float(joint.vlm.flow_hidden_gradient_norm);assert flow>0
            opt.step();sched.step()
            with torch.no_grad():
                for n,p in params.items():ema[n].lerp_(p.detach().float(),.001)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
            mean=loss.detach().clone();dist.all_reduce(mean);mean/=4
            values.append(dict(offset=offset,mean_loss=float(mean),gradient_norm=float(norm),flow_hidden_gradient=flow,seconds=seconds,tokens=tokens,sam_batch_used=getattr(joint,'sam_batch_used',False)))
            print('ROUND24_UPDATE_REPLAY',rank,variant,values[-1],flush=True)
            del samples,args,clean,loss,a,c,terms
        assert all(bool(torch.isfinite(p).all()) for p in params.values())
        hashes[variant]=synced(joint);rows[variant]=values
    opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    assert restore(out/'latest.pt',joint,opt,sched,ema,rank,device)==(step,epoch,start_batch)
    joint.vlm.audit_flow_gradient=False
    relative=[abs(a['mean_loss']-b['mean_loss'])/max(abs(a['mean_loss']),1e-8) for a,b in zip(rows['baseline'],rows['candidate'])]
    ratios=[b['gradient_norm']/max(a['gradient_norm'],1e-12) for a,b in zip(rows['baseline'],rows['candidate'])]
    all_batched=all(v['sam_batch_used'] for v in rows['candidate'])
    accepted=all_batched and len(flow_matches)==8 and all(flow_matches) and max(relative)<=.05 and statistics.median(relative)<=.02 and all(.5<=x<=2. for x in ratios)
    decision=torch.tensor(int(accepted),device=device);dist.all_reduce(decision,op=dist.ReduceOp.MIN);accepted=bool(decision.item())
    result=dict(status='PASS',meaning='Comparison executed with finite updates and full-state restoration; candidate acceptance is a separate field',candidate_accepted=accepted,step=step,updates_per_variant=8,rows=rows,max_relative_loss=max(relative),median_relative_loss=statistics.median(relative),gradient_norm_ratios=ratios,after_update_hashes=hashes,full_model_optimizer_scheduler_ema_rng_position_restored=True,no_production_steps_advanced=True,seconds=time.monotonic()-begin,bounds=dict(max_relative_loss=.05,median_relative_loss=.02,gradient_norm_ratio=[.5,2.]),final_369_score_equivalence_not_established=True,flow_dropout_rng_order_changed=True,all_candidate_updates_batched=all_batched,paired_flow_noise_and_time_verified=len(flow_matches)==8 and all(flow_matches),per_update_initial_rng_paired=True,adapter_dropout_masks_not_bitwise_paired=True,comparison_scope='Both arms use identical flow noise/time and perupdate initial RNG. The original raw stochastic comparison remains rejected. Production sampling is independent with altered draw order, not same random trajectory.')
    write(out/f'round24-update-replay-rank{rank}.json',result)
    return accepted
