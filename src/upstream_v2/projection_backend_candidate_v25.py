"""Isolated static-row32 per-projection dispatch; not installed in any worker.

Install after directional/pipeline/row-tile wrappers and before memo. Other
backend modules (including dynamic/long-input fallback) are never modified.
CPU launch and routing tests do not establish GPU numerical/speed acceptance.
"""
from pipeline_launch_candidate_v12 import launch_config
from projection_launch_candidate_v25 import ROLES, select, validate


def packed_roles(blocks):
    """Retain exact objects so recycled Python IDs cannot authorize a weight."""
    mapped = {}
    for block in blocks:
        packed = block._packed_nf4
        if not all(role in packed for role in ROLES):
            raise ValueError('Incomplete packed projection mapping')
        for role in ROLES:
            value = packed[role]
            previous = mapped.get(id(value))
            if previous is not None and previous[1] != role:
                raise ValueError('One packed object has conflicting roles')
            mapped[id(value)] = (value, role)
    if not mapped:
        raise ValueError('No packed blocks')
    return mapped


def install_backend(module, choose_config, blocks):
    if module.BM != 32:
        raise ValueError('Per-projection candidate requires original row32 backend')
    original = module.grouped_mm
    roles = packed_roles(blocks)

    def grouped_mm(a, p, ends, rows, transposed):
        cfg = choose_config()
        overrides = validate(cfg)
        if not overrides:
            return original(a, p, ends, rows, transposed)
        entry = roles.get(id(p))
        if entry is None or entry[0] is not p:
            raise ValueError('Unknown packed projection identity')
        v = select(cfg, entry[1], transposed)
        if v == launch_config(cfg, transposed):
            return original(a, p, ends, rows, transposed)
        assert a.is_contiguous() and a.dtype == module.torch.bfloat16 and p.sources is None
        n, k = p.shape if transposed else p.shape[::-1]
        assert a.shape[1] == k and rows > 0 and rows % 32 == 0 and a.shape[0] >= rows
        output = a.new_zeros(a.shape[0], n)
        ctas = min(rows // 32 * module.triton.cdiv(n, v['bn']),
                   module.torch.cuda.get_device_properties(a.device).multi_processor_count * v['waves'])
        module._mm[(ctas,)](
            a, p.data, p.absmax, p.scale_state.absmax, p.scale_state.code,
            p.offsets, p.code, ends, output, n, k, p.experts, rows,
            p.shape[0] * p.shape[1], transposed, p.blocksize, p.scale_state.blocksize,
            32, v['bn'], v['bk'], ctas, rows // 32, v['group'],
            num_warps=v['warps'], num_stages=v['stages'], enable_fp_fusion=False)
        return output

    module.grouped_mm = grouped_mm

    def undo():
        if module.grouped_mm is not grouped_mm:
            raise RuntimeError('Remove outer wrappers before projection dispatch')
        module.grouped_mm = original
    return undo
