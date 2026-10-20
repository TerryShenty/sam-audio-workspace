#!/usr/bin/env python3
"""Verify and run the pinned SAM Audio engineering preview."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ARCHIVE = 'workspace-source-03a38144466ad18f.tar.gz'
ARCHIVE_SHA256 = '03a38144466ad18f8b5250ad24222333c359474155e04e2bc33dbb8153894254'
MANIFEST_SHA256 = 'a819fea62bb09551347cf860cf34579e09623583643c02d96eb5100f6c791c50'
REPOSITORY = 'TerryShenty/sam-audio-workspace'
ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_directory(path):
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Directory path contains a symlink: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_source(source):
    if source.is_symlink():
        raise ValueError('Source directory must not be a symlink')
    manifest = source / 'SOURCE_MANIFEST.json'
    if manifest.is_symlink() or digest(manifest) != MANIFEST_SHA256:
        raise ValueError('Source manifest mismatch')
    files = json.loads(manifest.read_text())
    for name, expected in files.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or str(rel) != name:
            raise ValueError('Unsafe manifest path')
        path = source / name
        if any(p.is_symlink() for p in [path, *path.parents] if p != source.parent):
            raise ValueError('Symlink in source: ' + name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Source file changed: ' + name)
    return files


def extract_verified(archive, destination):
    if archive.is_symlink() or digest(archive) != ARCHIVE_SHA256:
        raise ValueError('Source archive SHA-256 mismatch')
    if destination.exists() or destination.is_symlink():
        verify_source(destination)
        return destination
    checked_directory(destination.parent)
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=destination.parent) as tmp:
        tmp = Path(tmp)
        seen = set()
        with tarfile.open(archive, 'r:gz') as tf:
            for item in tf:
                path = PurePosixPath(item.name)
                if (not item.isfile() or path.is_absolute() or '..' in path.parts
                        or len(path.parts) < 2 or path.parts[0] != 'training'
                        or str(path) != item.name or item.name in seen):
                    raise ValueError('Unsafe or duplicate source archive entry')
                seen.add(item.name)
                out = tmp.joinpath(*path.parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                stream = tf.extractfile(item)
                with out.open('xb') as f:
                    shutil.copyfileobj(stream, f)
                out.chmod(0o755 if item.mode & 0o111 else 0o644)
        source = tmp / 'training'
        manifest = verify_source(source)
        if seen != {'training/' + p for p in [*manifest, 'SOURCE_MANIFEST.json']}:
            raise ValueError('Archive contains files outside its source manifest')
        source.rename(destination)
    return destination


def obtain_archive(cache):
    bundled = ROOT / ARCHIVE
    if bundled.exists() or bundled.is_symlink():
        if bundled.is_symlink() or digest(bundled) != ARCHIVE_SHA256:
            raise ValueError('Repository archive differs from its pinned hash')
        return bundled
    target = cache / ARCHIVE
    if target.exists() or target.is_symlink():
        if target.is_symlink() or digest(target) != ARCHIVE_SHA256:
            raise ValueError('Cached archive differs; refusing to overwrite it')
        return target
    url = 'https://raw.githubusercontent.com/' + REPOSITORY + '/main/' + ARCHIVE
    fd, temp = tempfile.mkstemp(prefix='.download-', dir=cache)
    try:
        with os.fdopen(fd, 'wb') as out, urllib.request.urlopen(url, timeout=120) as response:
            shutil.copyfileobj(response, out)
        temp = Path(temp)
        if digest(temp) != ARCHIVE_SHA256:
            raise ValueError('Downloaded archive SHA-256 mismatch')
        temp.rename(target)
    finally:
        Path(temp).unlink(missing_ok=True)
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog='Other arguments are forwarded to the workflow, including --wandb-entity, --through and --only. No full93942 training is launched.')
    p.add_argument('--directory', type=Path, required=True, help='Persistent source and run directory')
    p.add_argument('--source-only', action='store_true', help='Verify source only; no install, model or data download, or GPU execution')
    args, forwarded = p.parse_known_args(argv)
    if not args.source_only:
        if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
            p.error('Execution requires Linux x86_64 and Python 3.12; --source-only works without GPU dependencies.')
    root = checked_directory(args.directory)
    lock_path = root / 'BOOTSTRAP.lock'
    if lock_path.is_symlink():
        raise ValueError('Bootstrap lock must not be a symlink')
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = checked_directory(root / '.sam-source')
        source = extract_verified(obtain_archive(cache), cache / ARCHIVE_SHA256[:16])
        receipt = dict(status='SOURCE_VERIFIED', archive_sha256=ARCHIVE_SHA256,
                       source_manifest_sha256=MANIFEST_SHA256, source_files=len(verify_source(source)),
                       source=str(source), executed_workflow=False)
        report = root / 'BOOTSTRAP_SOURCE.json'
        if report.is_symlink():
            raise ValueError('Bootstrap receipt must not be a symlink')
        report.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
        if args.source_only:
            return 0
        env = os.environ.copy()
        env['PYTHON_BIN'] = sys.executable
        return subprocess.run(['bash', str(source / 'start.sh'), '--directory',
                               str(root / 'workspace'), *forwarded], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
