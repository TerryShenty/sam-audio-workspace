"""CPU/Gloo integration only: real DDP, complete restore, sampler and 817 coverage.
Run: python tests/test_world_size.py. Does not prove CUDA/NCCL/model performance.
"""
import copy,json,os,random,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from joint_sampler import JointSampler
from qualification_data import QualificationSampler
from validation_inputs import batches
from validation_coverage import collect,METRICS

def worker(rank,world,rendezvous,folder):
 import numpy as np
 import torch
 import torch.distributed as dist
 from joint_ddp import wrap
 from distributed_trainer import DistributedTrainer
 from qualification_restore import verify
 from joint_checkpoint import publish
 torch.set_num_threads(1)
 dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=world)
 try:
  class Model(torch.nn.Module):
   def __init__(self):
    super().__init__();self.vlm=torch.nn.Linear(1,1,bias=False);self.bridge=torch.nn.Linear(1,1,bias=False)
    self.sam=torch.nn.Module();self.sam.lora_w=torch.nn.Parameter(torch.tensor([.1]));self.sam.condition=torch.nn.Parameter(torch.tensor([.1]))
    self.register_buffer('buffer',torch.tensor([3.]))
   def forward(self,x):return self.vlm(x)+self.bridge(x)+self.sam.lora_w+self.sam.condition
  torch.manual_seed(32)
  model=Model();reference=copy.deepcopy(model)
  wrapped,receipt=wrap(model,device=torch.device('cpu'),immutable_identity='a'*64)
  assert receipt['world_size']==world
  if world==1:assert receipt['ddp_enabled'] is False
  else:assert isinstance(wrapped,torch.nn.parallel.DistributedDataParallel)
  sampler=JointSampler(list(map(str,range(111068))),rank=rank,seed=1,data_identity='x',world_size=world)
  config=dict(world_size=world,per_rank_batch=2,global_batch=2*world)
  opt=torch.optim.Adam(model.parameters(),lr=.001)
  trainer=DistributedTrainer(wrapped,opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1),sampler,identity='b'*64,configuration=config)
  ropt=torch.optim.Adam(reference.parameters(),lr=.001)
  def loss(m,indices):
   x=torch.tensor([[(i%17)/100] for i in indices]);return (m(x)**2).mean()
  indices=sampler.peek();allindices=[None]*world;dist.all_gather_object(allindices,indices)
  ropt.zero_grad();loss(reference,[i for rows in allindices for i in rows]).backward()
  torch.nn.utils.clip_grad_norm_(reference.parameters(),1.);ropt.step()
  trainer.step(loss)
  for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
  directory=Path(folder)/'checkpoint';trainer.checkpoint(directory)
  manifest=directory/'COMPLETE.json'
  saved=trainer.prepare_restore(manifest)
  trainer.step(loss);expected=trainer.snapshot()
  trainer.restore(manifest);verify(trainer,saved)
  trainer.step(loss);verify(trainer,expected)
  wrong=dict(config,world_size=world+1,global_batch=2*(world+1))
  from joint_checkpoint import load_rank
  try:load_rank(manifest,rank=rank,identity='b'*64,configuration=wrong)
  except AssertionError:pass
  else:raise AssertionError('Cross-world resume accepted')
  if rank==0:
   data=json.loads(manifest.read_text())
   try:publish(Path(folder)/'bad',data['ranks'][:-1])
   except AssertionError:pass
   else:raise AssertionError('Missing-rank checkpoint accepted')
  dist.barrier()
 finally:dist.destroy_process_group()

class Contracts(unittest.TestCase):
 def test_sampler_validation_qualification(self):
  for world in range(1,9):
   ids=list(map(str,range(111068)))
   samplers=[JointSampler(ids,rank=r,world_size=world,seed=5,data_identity='x') for r in range(world)]
   # Cross a real corpus boundary and restore exactly.
   cursor=(111068//(2*world))*(2*world)
   for s in samplers:s.cursor=cursor
   seen=[i for s in samplers for i in s.peek()];self.assertEqual(len(seen),2*world)
   for s in samplers:
    s.commit();state=s.state_dict();expected=s.peek();s.pending=None;s.load_state_dict(state);self.assertEqual(s.peek(),expected)
   parts=[x for r in range(world) for part in batches(list(range(817)),rank=r,world_size=world,batch_size=8) for x in part]
   self.assertEqual(sorted(parts),list(range(817)))
   protocol={'rows':[dict(sample_id=str(i),noise_seed=i,dataset='fixture',original_row_index=i) for i in range(817)]}
   ranks=[]
   for r in range(world):
    rows=[dict(sample_id=str(i),noise_seed=i,dataset='fixture',row_index=i,noise_sha256='c'*64,evaluation_samples=10,**{k:1. for k in METRICS}) for i in range(r,817,world)]
    ranks.append(dict(rank=r,complete=True,rows=rows,weight_choice='live',solver='midpoint',steps=16,checkpoint_identity='a'*64,validation_identity='b'*64))
   self.assertEqual(collect(ranks,protocol,world_size=world)['examples'],817)
   cases={'max':{'ranks':[dict(rank=r,samples=[dict(sample_id=str(2*r)),dict(sample_id=str(2*r+1))]) for r in range(4)]}}
   qs=[QualificationSampler(JointSampler(ids,rank=r,world_size=world,seed=5,data_identity='x'),cases) for r in range(world)]
   covered=[]
   for _ in qs[0].case_names:
    for s in qs:covered.extend(s.peek());s.commit()
   self.assertEqual(set(covered),set(range(8)))
 def test_real_ddp_and_restore(self):
  import torch.multiprocessing as mp
  for world in (1,2,4,8):
   with tempfile.TemporaryDirectory() as folder:
    mp.spawn(worker,args=(world,str(Path(folder)/'rendezvous'),folder),nprocs=world,join=True)
   print('PASS CPU/Gloo actual updates and full restore:',world,'ranks',flush=True)

if __name__=='__main__':unittest.main()
