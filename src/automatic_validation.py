"""Automatic real-state memory screening followed by all 817 held-out examples."""
import gc,hashlib,json,time
from pathlib import Path

def execute(session,root,output,watchdog,*,checkpoint_identity,batch_size=1):
 import torch,torch.distributed as dist
 from portable_codec import load
 from validation_inputs import bind,batches
 from joint_validation import evaluation_state,generate
 from validation_scoring import load as scorers,score_item
 from production_watchdog import cgroup_memory,write
 from validation_run import run
 root=Path(root);output=Path(output);rank=dist.get_rank();device=session.trainer.device
 protocol=json.loads((root/'VALIDATION_PROTOCOL_BINDING.json').read_text())
 vid=hashlib.sha256((root/'VALIDATION_PROTOCOL_BINDING.json').read_bytes()).hexdigest()
 rows=json.loads((root/'BOUND_ROWS.json').read_text());items=bind(rows,protocol)
 maximum={}
 for group in rows.values():
  for item in group:maximum[item['media_key']]=max(maximum.get(item['media_key'],0),item['prompt_tokens'])
 # Batch 1 is the explicitly conservative initial portable policy. Batch 8 is
 # not claimed safe until tested with this retained training state.
 assert batch_size==1,'Portable initial validation is qualified at batch 1'
 local=[x for i,x in enumerate(items) if i%dist.get_world_size()==rank]
 records={r['media_key']:r for r in json.loads((root/'PROCESSOR_COMPLETE.json').read_text())['records']}
 references={d:json.loads((Path(__file__).parent/'validation_reference'/d/'validation_inputs.json').read_text()) for d in ['music','speech']}
 lookup={d:{int(r['row']['row_index']):r for r in rows} for d,rows in references.items()}
 def frames(x):
  r=lookup[x['protocol']['domain']][x['protocol']['original_row_index']]
  return r['latent_frames'] if x['protocol']['domain']=='music' else r['features']['latent_frames']
 # Match the actual token arithmetic used by the original demanding-case selector.
 selectors=[lambda x:records[x['item']['media_key']]['max_actual_tokens']-maximum[x['item']['media_key']]+x['item']['prompt_tokens'],frames,lambda x:__import__('math').prod(records[x['item']['media_key']]['video_shape'])]
 cases=[]
 for key in selectors:
  x=max(local,key=key)
  if x not in cases:cases.append(x)
 codec=load(Path(__file__).parent.parent/'codec_assets',device)
 namespaces={d:scorers(root,d) for d in ['music','speech']}
 observed=[];status=None
 try:
  with evaluation_state(session.runtime.joint,device):
   guarded=session.runtime.guard_step(lambda part:generate(session.runtime.joint,codec,session.batch.store,part,device))
   for i,x in enumerate(cases):
    watchdog.tick('validate',100000+i)
    torch.cuda.reset_peak_memory_stats(device)
    pred=guarded([x])[0];score_item(namespaces[x['protocol']['domain']],x,pred['prediction'])
    torch.cuda.synchronize(device);m=watchdog.nv.nvmlDeviceGetMemoryInfo(watchdog.handle);host=cgroup_memory()
    assert m.free>=2*2**30 and host['current']<.9*host['limit'],'VALIDATION_HEADROOM_FAILED'
    observed.append(dict(sample_id=x['item']['sample_id'],physical_free=m.free,allocated_peak=torch.cuda.max_memory_allocated(device),host_current=host['current']))
  status=dict(rank=rank,passed=True,cases=observed)
 except Exception as e:status=dict(rank=rank,passed=False,error=repr(e),cases=observed)
 ranks=[None]*dist.get_world_size();dist.all_gather_object(ranks,status)
 assert all(r['passed'] for r in ranks),ranks
 approval=dict(status='PASS_TARGET_TRAIN_STATE_VALIDATION_MEMORY',checkpoint_identity=checkpoint_identity,validation_identity=vid,world_size=dist.get_world_size(),batch_size=1,longest_actual_padded_inputs=True,codec_and_scoring_included=True,ranks=ranks)
 if rank==0:write(output.with_name(output.stem+'-memory.json'),approval)
 try:
  return run(session,codec,session.batch.store,root=root,output=output,checkpoint_identity=checkpoint_identity,validation_identity=vid,expected_step=session.trainer.step_number,batch_size=1,memory_approval=approval,watchdog=watchdog)
 finally:
  del codec;gc.collect();torch.cuda.empty_cache()
