#!/usr/bin/env python3
"""Single entry for the pinned Qwen30B/SAM V6 handoff. English-only UI."""
import argparse, hashlib, json, os, pathlib, subprocess, sys, urllib.request, zipfile, fcntl
URL='https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/qwen30b-4gpu-v6-auto-source.zip'
SHA='665f287aac5456338f1aae192d2284ecf4c74f4788d762bc3970eb2cb67ba6ac'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['setup','prepare','qualify','train'])
 p.add_argument('--root',required=True,type=pathlib.Path)
 p.add_argument('--account');p.add_argument('--partition');p.add_argument('--output',type=pathlib.Path);p.add_argument('--gate',type=pathlib.Path)
 a=p.parse_args(); root=a.root.expanduser().resolve(); package=root/'qwen30b-4gpu-v6-auto'
 def run(args,**kw):subprocess.run([str(x) for x in args],check=True,**kw)
 if a.action in ('setup','prepare'):
  if os.environ.get('SLURM_JOB_GPUS'):p.error('Use CPU/data-transfer facilities for prepare, not a GPU allocation.')
  root.mkdir(parents=True,exist_ok=True)
  setup_lock=(root/'prepare.lock').open('a');fcntl.flock(setup_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if not package.exists():
   archive=root/'qwen-v6.zip.part'
   urllib.request.urlretrieve(URL,archive)
   if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA:raise RuntimeError('Source archive checksum mismatch')
   with zipfile.ZipFile(archive) as z:
    for n in z.namelist():
     if not (root/n).resolve().is_relative_to(root):raise RuntimeError('Unsafe archive path')
    z.extractall(root)
   archive.unlink()
  run([sys.executable,package/'verify_package.py'])
  environment=root/'environment';python=environment/'bin/python'
  if not (environment/'ENVIRONMENT_READY.json').is_file():
   run([sys.executable,package/'setup_environment.py','--venv',environment])
  if a.action=='setup':
   print('Next, sign in with your own HF account:',environment/'bin/hf','auth login');return
  os.environ['LD_LIBRARY_PATH']=str(environment/'media/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
  os.environ['PATH']=str(environment/'media/bin')+':'+os.environ['PATH']
  run([python,package/'check_access.py'])
  # Check release availability before starting large model downloads.
  urllib.request.urlopen('https://huggingface.co/datasets/TerryShenty/qwen30b-sam-joint-111k/resolve/main/UPLOAD_COMPLETE.json').close()
  run([python,package/'download_models.py','--model-root',root/'models'])
  run([python,package/'download_data.py','--destination',root/'joint_data'])
  paths={'data_root':str(root/'joint_data/data'),'sam_namespace':str(root/'joint_data/sam_namespace'),'qwen_model':str(root/'models/Qwen3-Omni-30B-A3B-Thinking'),'sam_model':str(root/'models/sam-audio-large-tv')}
  config=root/'paths.json';config.write_text(json.dumps(paths,indent=2)+'\n')
  run([python,package/'prepare.py','--paths',config],cwd=package)
  print('Prepared. Next: run this entry with qualify, --account and --partition.')
 else:
  if not a.account or not a.partition:p.error('--account and --partition are required')
  if not (package/'CPU_READY.json').is_file():p.error('Run prepare successfully first')
  if a.action=='train' and (not a.gate or not a.gate.expanduser().is_file()):p.error('train requires --gate pointing to a successful PORTABLE_GATE.json')
  output=(a.output or root/'runs'/a.action).expanduser().resolve()
  if output.exists():p.error('Output already exists; use a new --output to avoid duplicate submissions')
  output.parent.mkdir(parents=True,exist_ok=True)
  lock=output.parent/(output.name+'.submission.json')
  env=os.environ.copy();env.update(QWEN_PACKAGE=str(package),QWEN_PYTHON=str(root/'environment/bin/python'),QWEN_OUTPUT=str(output),QWEN_MODE=a.action)
  if a.gate:env['QWEN_GATE']=str(a.gate.expanduser().resolve())
  # Never inherit a research/old resume path accidentally.
  env.pop('QWEN_RESUME',None)
  with lock.open('x') as f:json.dump({'status':'SUBMISSION_INTENT','output':str(output)},f)
  result=subprocess.run(['sbatch','--parsable','--account='+a.account,'--partition='+a.partition,'--time='+('01:00:00' if a.action=='qualify' else '24:00:00'),str(package/'train.sbatch')],cwd=package,env=env,text=True,capture_output=True)
  lock.write_text(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'output':str(output)},indent=2))
  print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
  if result.returncode:raise SystemExit(result.returncode)
  print('Submission recorded in',lock)
if __name__=='__main__':main()
