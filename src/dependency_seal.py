"""CPU preflight for explicit runtime assets; no model imports or GPU."""
from pathlib import Path
import hashlib,runpy,json

def verify(version):
    version=Path(version).resolve()
    runtime=version/'upstream_v2/runtime_v1'
    expected=dict(runpy.run_path(str(runtime/'fast_moe.py'))['EXPECTED'])
    perf=runtime/'perf'
    proof=perf/'deployment-round17/characterization.json'
    record=json.loads(proof.read_text())
    assert record['status']=='QUALIFIED_FOR_REAL_UPDATE_COMPARISON_NOT_ACCEPTED_FOR_PRODUCTION'
    expected['hybrid-exclusive-round16.json']=record['raw_proof_sha256']
    seal={}
    for name,digest in expected.items():
        p=(perf/name).resolve()
        assert p.is_relative_to(version),('ESCAPED_SOURCE',name)
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual==digest,('DEPENDENCY_CHANGED',name)
        seal[str(p)]=actual
    assert proof.resolve().is_relative_to(version)
    seal[str(proof.resolve())]=hashlib.sha256(proof.read_bytes()).hexdigest()
    return seal

def source_seal(version):
    version=Path(version).resolve();result=verify(version)
    for p in sorted(version.rglob('*.py')):
        assert p.resolve().is_relative_to(version)
        result[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result
