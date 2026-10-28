"""Dependency-free archive integrity and syntax verification."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent.parent

def verify():
    manifest=json.loads((P/'SOURCE_MANIFEST.json').read_text())
    for name,want in manifest['files'].items():
        p=(P/name).resolve();assert p.is_relative_to(P)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==want,('PACKAGE_CHANGED',name)
        if p.suffix=='.py':ast.parse(p.read_text(),filename=name)
    return manifest
if __name__=='__main__':print('PASS source hashes and Python syntax:',len(verify()['files']),'files; GPU execution not tested')
