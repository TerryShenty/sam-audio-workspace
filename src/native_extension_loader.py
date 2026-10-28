"""Reuse exact accepted native binaries; do not compile on GPU allocations."""
import hashlib,importlib,importlib.util,json
from pathlib import Path

def install():
    version=Path(__file__).resolve().parent
    records=json.loads((version/'native_extensions.json').read_text())
    assert len(records)==4 and len({x['path'] for x in records})==4
    for item in records:
        path=Path(item['path']).resolve();assert path.is_relative_to(version)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        module=importlib.import_module(item['module'])
        if module._extension is None:
            spec=importlib.util.spec_from_file_location(path.stem,path)
            extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
            module._extension=extension
        assert Path(module._extension.__file__).resolve()==path
    return records
