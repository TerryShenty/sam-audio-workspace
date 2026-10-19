#!/usr/bin/env python3
"""Download verified preparation source, inspect host/release, optionally install.
No sudo, GPU allocation, training, dataset bulk download or deletion.
"""
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, tarfile, urllib.request, tempfile, time
from pathlib import Path
COMMIT = 'main'
SHA = '62dce9d12285e4f3e726a75ab340eba7c507573bc5e586a2d6a382478136b1c5'
ARCHIVE = 'workspace-source-62dce9d12285e4f3.tar.gz'
BASE = 'https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/' + COMMIT
DATASET = 'TerryShenty/avspeech-93942'
def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return r.read()
        except (OSError, TimeoutError):
            if attempt == 2: raise
            time.sleep(attempt + 1)
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
    if sys.version_info < (3, 12):
        raise SystemExit('Use Python 3.12 or newer for the preparation entry (installation uses 3.12).')
    root.mkdir(parents=True, exist_ok=True)
    payload=fetch(BASE+'/'+ARCHIVE)
    with tempfile.TemporaryDirectory(dir=root) as staging:
        stage=Path(staging);unpack(payload,stage)
        target=root/'training'
        if target.exists():
            for source in (stage/'training').rglob('*'):
                if not source.is_file(): continue
                dest=target/source.relative_to(stage/'training')
                if dest.is_symlink() or not dest.is_file() or dest.read_bytes()!=source.read_bytes():
                    raise SystemExit('Existing source differs: '+str(dest)+'. Choose a new --directory; no files overwritten.')
        else: (stage/'training').rename(target)
    report={'code_commit':COMMIT,'source_sha256':SHA,'host':platform.platform(),'dataset':DATASET,'training_ready':False,'notes':['Preparation release; install and GPU train/eval/resume not yet certified','No bulk data download or training started']}
    report['tools']={k:shutil.which(k) for k in ['python3.12','git','ffmpeg','nvidia-smi']}
    if report['tools']['nvidia-smi']:
        try:
            q=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=30)
            report['gpu']=q.stdout.strip() if q.returncode==0 else 'nvidia-smi failed'
        except (OSError, subprocess.TimeoutExpired) as exc: report['gpu_error']=str(exc)
    try:
        meta=json.loads(fetch('https://huggingface.co/api/datasets/'+DATASET)); rev=meta['sha']
        report['dataset_revision']=rev
        report['dataset_status']=json.loads(fetch('https://huggingface.co/datasets/'+DATASET+'/resolve/'+rev+'/STATUS.json'))
    except Exception as e: report['dataset_lookup_error']=str(e)
    (root/'MACHINE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('Source prepared at:',root/'training')
    if not o.install: print('Inspection complete. Use --install with this directory to attempt reference installation.');return
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
if __name__=='__main__':
    try: main()
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise SystemExit('Preparation stopped: '+str(exc))
