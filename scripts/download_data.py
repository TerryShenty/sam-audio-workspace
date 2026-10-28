"""Download immutable HF shards; verify hashes and safely extract once."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download

def main():
 a=argparse.ArgumentParser();a.add_argument('--destination',type=Path,required=True);a.add_argument('--revision',default='main');a=a.parse_args()
 repo='TerryShenty/qwen30b-sam-joint-111k';api=HfApi();revision=api.repo_info(repo,repo_type='dataset',revision=a.revision).sha
 assert api.file_exists(repo,'UPLOAD_COMPLETE.json',repo_type='dataset',revision=revision),'Dataset upload not complete yet'
 root=a.destination.resolve();root.mkdir(parents=True,exist_ok=True);stage=root/'.download';stage.mkdir(exist_ok=True)
 index=Path(hf_hub_download(repo,'SHARDS.json',repo_type='dataset',revision=revision,local_dir=stage))
 for item in json.loads(index.read_text()):
  receipt=stage/(Path(item['name']).name+'.extracted.json')
  if receipt.exists():
   assert json.loads(receipt.read_text())=={'revision':revision,'sha256':item['sha256']};continue
  p=Path(hf_hub_download(repo,item['name'],repo_type='dataset',revision=revision,local_dir=stage))
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  assert p.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256']
  with tarfile.open(p) as tf:
   for m in tf:
    target=(root/m.name).resolve();assert target.is_relative_to(root) and m.isfile()
    if target.exists():raise FileExistsError(f'Partial extraction needs reconciliation: {target}')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(m) as src,target.open('xb') as dst:
     import shutil
     shutil.copyfileobj(src,dst,8*1024**2)
  receipt.write_text(json.dumps({'revision':revision,'sha256':item['sha256']}));p.unlink()
 print('Verified and extracted dataset revision',revision)
if __name__=='__main__':main()
