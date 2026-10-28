"""Isolated study controls. Never imported by the production trainer."""
import os
from pathlib import Path
from unique_scatter_round3 import extension as production_extension

MODE = 'baseline'
_extension = None
_cache = None


def clear_cache():
    global _cache
    _cache = None


def decode(packed):
    global _cache
    if MODE != 'decode_reuse':
        return {n: p.decode() for n, p in packed.items()}
    if _cache is None or _cache[0] is not packed:
        _cache = None
        _cache = (packed, {n: p.decode() for n, p in packed.items()})
    return _cache[1]


def extension():
    global _extension
    if MODE != 'bulk_gather':
        return production_extension()
    if _extension is None:
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent
        build = here / 'bulk-gather-round6-build'
        build.mkdir(exist_ok=True)
        os.environ['MAX_JOBS'] = '1'
        _extension = load(name='solos_qwen_bulk_gather_round6',
                          sources=[str(here/'expert_bulk_gather_round6.cpp')],
                          build_directory=str(build), extra_cflags=['-O2', '-g0'],
                          with_cuda=False, verbose=True)
    return _extension
