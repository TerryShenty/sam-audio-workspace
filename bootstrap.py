#!/usr/bin/env python3
"""Download verified preparation source, inspect host/release, optionally install.
No sudo, GPU allocation, training, dataset bulk download or deletion.
"""
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path
COMMIT = '5f44a86b8cdbb538d62863d4159590aea58c4136'
SHA = '11241e9530b0f461569d8b6066a328b042ad336b7a1f94ba7e521869d8ef0f78'
BASE = 'https://raw.githubusercontent.com/TerryShenty/sam-audio-training/' + COMMIT
DATASET = 'TerryShenty/avspeech-93942'
def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r: return r.read()
def unpack(payload, root):
    if hashlib.sha256(payload).hexdigest() != SHA: raise ValueError('Source SHA256 mismatch')
    import io
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as t:
        for m in t.getmembers():
            parts=Path(m.name).parts
            if not parts or parts[0]!='training' or '..' in parts or m.name.startswith('/') or not (m.isfile() or m.isdir()):
                raise ValueError('Unsafe source archive member')
        t.extractall(root, filter='data')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=Path('sam-audio-delivery'))
    p.add_argument('--install', action='store_true', help='Attempt experimental Linux reference installation; not GPU-certified')
    o=p.parse_args(); root=o.directory.resolve()
    if root.exists(): raise SystemExit('Choose a NEW --directory; existing files are never overwritten')
    payload=fetch(BASE+'/training-source.tar.gz');root.mkdir(parents=True);unpack(payload,root)
    report={'code_commit':COMMIT,'source_sha256':SHA,'host':platform.platform(),'dataset':DATASET,'training_ready':False,'notes':['Preparation release; install and GPU train/eval/resume not yet certified','No bulk data download or training started']}
    report['tools']={k:shutil.which(k) for k in ['python3.12','git','ffmpeg','nvidia-smi']}
    if report['tools']['nvidia-smi']:
        q=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=30)
        report['gpu']=q.stdout.strip() if q.returncode==0 else 'nvidia-smi failed'
    try:
        meta=json.loads(fetch('https://huggingface.co/api/datasets/'+DATASET)); rev=meta['sha']
        report['dataset_revision']=rev
        report['dataset_status']=json.loads(fetch('https://huggingface.co/datasets/'+DATASET+'/resolve/'+rev+'/STATUS.json'))
    except Exception as e: report['dataset_lookup_error']=str(e)
    (root/'MACHINE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('Source prepared at:',root/'training')
    if not o.install: print('Inspection complete. Use --install with a new directory to attempt reference installation.');return
    if platform.system()!='Linux' or platform.machine() not in ['x86_64','amd64']:
        raise SystemExit('Reference installation requires Linux x86_64; host report saved')
    missing=[x for x in ['python3.12','git','ffmpeg'] if not report['tools'][x]]
    if missing: raise SystemExit('Ask server owner to supply: '+', '.join(missing)+'. No sudo attempted.')
    print('Installing isolated reference venv. Known decord compatibility issue may stop this attempt.',flush=True)
    with (root/'installation.log').open('w') as log:
        result=subprocess.run(['bash','setup.sh'],cwd=root/'training',stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise SystemExit('Installation failed; inspect '+str(root/'installation.log')+'. Training not started.')
    result=subprocess.run([str(root/'training/.venv/bin/python'),'tools/preflight.py','--output',str(root/'ENVIRONMENT_REPORT.json')],cwd=root/'training')
    if result.returncode: raise SystemExit('Environment check failed. Training not started.')
    print('Dependency checks passed. GPU train/eval/resume and complete dataset/cache acceptance STILL REQUIRED.')
if __name__=='__main__':main()
