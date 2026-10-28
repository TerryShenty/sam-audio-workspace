"""Explicitly record reviewed local source edits. Does not establish runtime correctness."""
import ast, hashlib, json
from pathlib import Path
P=Path(__file__).resolve().parent.parent

def main():
    manifest=P/'SOURCE_MANIFEST.json'
    # Fixed source scope excludes generated .so files, datasets and local paths.
    paths=[P/'README.md',P/'LICENSE',P/'.gitignore',P/'requirements.txt',P/'qwen_quickstart.py']
    for folder in ('scripts','configs','src','docs'):
        paths.extend(p for p in (P/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md','.txt','.cpp','.h','.cu','.cuh','.sbatch'))
    paths=[p for p in paths if '__pycache__' not in p.parts and p.name!='native_extensions.json']
    hashes={}
    for p in sorted(paths):
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        hashes[str(p.relative_to(P))]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'status':'SOURCE_IDENTITY_NOT_GPU_ACCEPTANCE','files':hashes},indent=2,sort_keys=True)+'\n')
    print('Recorded',len(hashes),'source files. Review changes and re-run affected preparation/checks before launch.')
if __name__=='__main__':main()
