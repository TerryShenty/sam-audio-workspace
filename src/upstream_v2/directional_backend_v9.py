"""Use existing arithmetic with independently selected forward/backward launch geometry.

The caller is single-threaded. Each Triton launch captures its constexpr values
before another call changes these Python globals. No packed weight is changed.
"""
KEYS={'bn':'BN','bk':'BK','warps':'WARPS','waves':'WAVES','group':'GROUP_M'}

def install(module, choose_config):
    original=module.grouped_mm
    def grouped_mm(a,p,ends,rows,transposed):
        cfg=choose_config()
        values=dict(bn=64,bk=32,warps=4,waves=cfg['waves'],group=cfg['group'])
        values.update(cfg.get('forward_kernel' if transposed else 'backward_kernel',{}))
        assert values['bn'] in (32,64,128) and values['bk'] in (32,64)
        assert values['warps'] in (4,8) and values['waves'] in (2,4,8)
        assert values['group'] in (1,2,4)
        for key,attr in KEYS.items():setattr(module,attr,values[key])
        return original(a,p,ends,rows,transposed)
    module.grouped_mm=grouped_mm
    return grouped_mm
