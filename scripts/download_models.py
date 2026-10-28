"""Official model download + exact checked-file verification; transfer/CPU resources only."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import snapshot_download,HfApi
p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,required=True);p.add_argument('--qwen-revision',default='main');a=p.parse_args()
expected=json.loads((Path(__file__).resolve().parent.parent/'configs/MODEL_SHA256.json').read_text());receipts=[]
for kind,repo,rev,name in [('qwen','Qwen/Qwen3-Omni-30B-A3B-Thinking',a.qwen_revision,'Qwen3-Omni-30B-A3B-Thinking'),('sam','facebook/sam-audio-large-tv','b9fc94687ec044c570cbe30b8c28100cd056f1cb','sam-audio-large-tv')]:
    commit=HfApi().model_info(repo,revision=rev).sha
    root=a.model_root/name;snapshot_download(repo_id=repo,revision=commit,local_dir=root)
    for file,want in expected[kind].items():
        h=hashlib.sha256()
        with (root/file).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==want,('Model revision differs from cache identity',kind,file,commit)
    receipts.append({'repo':repo,'revision':commit,'checked_files':len(expected[kind]),'path':str(root.resolve())})
(a.model_root/'MODEL_DOWNLOAD_RECEIPT.json').write_text(json.dumps(receipts,indent=2)+'\n')
print('Model checked-file identities match. prepare.py still required.')
