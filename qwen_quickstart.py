#!/usr/bin/env python3
"""Run the checked-out Qwen30B + SAM V6 source."""
import argparse, json, os, pathlib, subprocess, sys, urllib.request, fcntl
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['setup','prepare','qualify','train'])
 p.add_argument('--root',required=True,type=pathlib.Path)
 p.add_argument('--python',type=pathlib.Path,help='Use an existing environment; do not install packages')
 p.add_argument('--account');p.add_argument('--partition');p.add_argument('--output',type=pathlib.Path);p.add_argument('--gate',type=pathlib.Path)
 p.add_argument('--gpus',type=int,default=4,help='GPUs on one node (1-8); per-GPU batch stays 2')
 a=p.parse_args(); root=a.root.expanduser().resolve(); package=pathlib.Path(__file__).resolve().parent
 if not 1<=a.gpus<=8:p.error('--gpus must be between 1 and 8 on one node')
 def run(args,**kw):subprocess.run([str(x) for x in args],check=True,**kw)
 if a.action in ('setup','prepare'):
  if os.environ.get('SLURM_JOB_GPUS'):p.error('Use CPU/data-transfer facilities for prepare, not a GPU allocation.')
  root.mkdir(parents=True,exist_ok=True)
  setup_lock=(root/'prepare.lock').open('a');fcntl.flock(setup_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run([sys.executable,package/'scripts/verify_package.py'])
  environment=root/'environment';python=a.python.expanduser().resolve() if a.python else environment/'bin/python'
  if a.python:
   run([python,package/'scripts/check_environment.py'])
  elif not (environment/'ENVIRONMENT_READY.json').is_file():
   run([sys.executable,package/'scripts/setup_environment.py','--venv',environment])
  if a.action=='setup':
   print('Next, sign in with your own HF account:',python.parent/'hf','auth login');return
  os.environ['LD_LIBRARY_PATH']=str(environment/'media/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
  os.environ['PATH']=str(environment/'media/bin')+':'+os.environ['PATH']
  run([python,package/'scripts/check_access.py'])
  # Check release availability before starting large model downloads.
  urllib.request.urlopen('https://huggingface.co/datasets/TerryShenty/qwen30b-sam-joint-111k/resolve/main/UPLOAD_COMPLETE.json').close()
  run([python,package/'scripts/download_models.py','--model-root',root/'models'])
  run([python,package/'scripts/download_data.py','--destination',root/'joint_data'])
  paths={'data_root':str(root/'joint_data/data'),'sam_namespace':str(root/'joint_data/sam_namespace'),'qwen_model':str(root/'models/Qwen3-Omni-30B-A3B-Thinking'),'sam_model':str(root/'models/sam-audio-large-tv')}
  config=root/'paths.json';config.write_text(json.dumps(paths,indent=2)+'\n')
  run([python,package/'scripts/prepare.py','--paths',config],cwd=package)
  print('Prepared. Next: run this entry with train, --account and --partition.')
 else:
  if not a.account or not a.partition:p.error('--account and --partition are required')
  if not (package/'CPU_READY.json').is_file():p.error('Run prepare successfully first')
  output=(a.output or root/'runs'/a.action).expanduser().resolve()
  if output.exists():p.error('Output already exists; use a new --output to avoid duplicate submissions')
  output.parent.mkdir(parents=True,exist_ok=True)
  lock=output.parent/(output.name+'.submission.json')
  env=os.environ.copy();env.update(QWEN_PACKAGE=str(package),QWEN_PYTHON=str(a.python.expanduser().resolve() if a.python else root/'environment/bin/python'),QWEN_OUTPUT=str(output),QWEN_MODE=a.action,QWEN_GPUS=str(a.gpus))
  env.pop('QWEN_GATE',None)
  if a.gate:env['QWEN_GATE']=str(a.gate.expanduser().resolve())
  # Never inherit a research/old resume path accidentally.
  env.pop('QWEN_RESUME',None)
  print(f'Request: {a.gpus} GPUs, per-GPU batch 2, global batch {2*a.gpus}; V6 fast path unchanged')
  with lock.open('x') as f:json.dump({'status':'SUBMISSION_INTENT','output':str(output)},f)
  result=subprocess.run(['sbatch','--parsable','--gres=gpu:'+str(a.gpus),'--cpus-per-task='+str(4*a.gpus),'--mem='+str(32*a.gpus)+'G','--account='+a.account,'--partition='+a.partition,'--time='+('01:00:00' if a.action=='qualify' else '24:00:00'),str(package/'scripts/train.sbatch')],cwd=package,env=env,text=True,capture_output=True)
  lock.write_text(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'output':str(output)},indent=2))
  print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
  if result.returncode:raise SystemExit(result.returncode)
  print('Submission recorded in',lock)
if __name__=='__main__':main()
