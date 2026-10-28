"""Isolated, NOT installed in v9/v11: tune stages of existing NF4 arithmetic.

Stages=1 delegates to the known implementation. More stages are a compiler
request, not proof that asynchronous copies were emitted or speed improved.
Install after directional geometry and before per-checkpoint memo wrappers.
"""
def launch_config(cfg, transposed):
    values = dict(bn=64, bk=32, warps=4, waves=cfg['waves'],
                  group=cfg['group'], stages=1)
    values.update(cfg.get('forward_kernel' if transposed else 'backward_kernel', {}))
    assert set(values) == {'bn', 'bk', 'warps', 'waves', 'group', 'stages'}
    assert values['bn'] in (32, 64, 128) and values['bk'] in (32, 64)
    assert values['warps'] in (4, 8) and values['waves'] in (2, 4, 8)
    assert values['group'] in (1, 2, 4) and values['stages'] in (1, 2, 3)
    return values


def install(module, choose_config, *, static_capacity):
    original = module.grouped_mm

    def grouped_mm(a, p, ends, rows, transposed):
        v = launch_config(choose_config(), transposed)
        if v['stages'] == 1:
            return original(a, p, ends, rows, transposed)
        torch = module.torch
        assert a.is_contiguous() and a.dtype == torch.bfloat16 and p.sources is None
        n, k = p.shape if transposed else p.shape[::-1]
        assert a.shape[1] == k and rows > 0 and rows % module.BM == 0
        output = a.new_zeros(a.shape[0], n) if static_capacity else a.new_empty(a.shape[0], n)
        ctas = min(rows // module.BM * module.triton.cdiv(n, v['bn']),
                   torch.cuda.get_device_properties(a.device).multi_processor_count * v['waves'])
        module._mm[(ctas,)](
            a, p.data, p.absmax, p.scale_state.absmax, p.scale_state.code,
            p.offsets, p.code, ends, output, n, k, p.experts, rows,
            p.shape[0] * p.shape[1], transposed, p.blocksize, p.scale_state.blocksize,
            module.BM, v['bn'], v['bk'], ctas, rows // module.BM, v['group'],
            num_warps=v['warps'], num_stages=v['stages'], enable_fp_fusion=False)
        if not static_capacity:
            output[rows:] = 0
        return output

    module.grouped_mm = grouped_mm
    return grouped_mm


def candidates(incumbent):
    """All direction combinations survive irrespective of standalone speed."""
    result = []
    for stages in (2, 3):
        for direction in ('forward', 'backward', 'both'):
            for retained in (16, 24):
                cfg = dict(incumbent, name=f'pipeline_s{stages}_{direction}_r{retained}',
                           retained=retained)
                cfg.pop('factors', None)
                cfg.pop('forward_kernel', None)
                cfg.pop('backward_kernel', None)
                for side in ('forward', 'backward'):
                    if direction in (side, 'both'):
                        cfg[side + '_kernel'] = {'stages': stages}
                result.append(cfg)
    return result
