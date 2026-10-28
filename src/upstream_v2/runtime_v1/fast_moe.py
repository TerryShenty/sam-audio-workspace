"""Staged exact MoE unique-scatter candidate; not active in production.

This module is intended to replace the root fast_moe.py only in an explicitly
authorized controlled continuation. The narrow scatter fastpath relies on unique
indices per expert, retaining the original expert order and all other arithmetic.
"""
import hashlib,sys,types
from pathlib import Path
EXPECTED = {'unique_scatter_round3.py': '72179880311b7b98dda72cc0a8c13284da351efa776513779dce71678340b8b3', 'expert_unique_scatter_round3.cpp': '954f6cc0e139abded113100aa7f4b106e219c3af0527bc5168974f1d654919b8', 'grouped_moe.py': 'd4a99afcd004de6a3f91070c9a93de0ec4da01a91d6925e12b879ab39d76a8ec', 'route_moe.py': '84653c07341f83c60895a37d43b6123933f60055828920a98df22983d6bdb80d'}

EXPECTED['saved_tensor_moe_round5.py'] = '98fdd6fe551a9ae25b193be59707f31ff8eecf060dbbb3db2abc4f384d461658'

EXPECTED.update({'bulk_gather_support_round6.py': '58944f400e483882c7d02358807260e1ee4849c8338ea67e6bc3fdeadf7ea4a4', 'bulk_gather_moe_round6.py': '21d78ca286d60b3a9a79319db58d3f8e0c496e9b5d04377a976331c10afbcb26', 'expert_bulk_gather_round6.cpp': '6e249ca730b118ee0bb42dd39fcac24c68e41684cfe76bfc97884d889eee95db'})

EXPECTED.update({'fused_decode_round7.py': '3e68cb8f4af549acddbbc80f2a3cca93ae5a8699c6eea5e22372f0c61cc2523a', 'fused_moe_round7.py': '71a5e630ec7ec8d72158bfda6aa5b52a1eb44fd5a06a06f4580b254f791a82b2', 'profile_round7.py': '61e792553cb452827680c018478d172190e8c795b98538602bd725dc15a2f156'})

EXPECTED.update({'pointwise_moe_round8.py': '93340eeddbe66cf2cb6ee73e11031c7e005f9d60ff0b0d9ac9e7e4d4416c55ed', 'expert_pointwise_round8.cpp': 'abbfd0af2f88e248fc0370e50b268e5e1cee92d6e3b155b8ae82402a5ba91268', 'profile_round8.py': '618c7e6ea3f3b327fafb39072f5a96246c4eab9096fa93590087c6aa0c262aed'})

EXPECTED.update({'hybrid_moe_round16.py': 'c72573a963f3c8dc3d3e40b37fc7f472130a826083e4e01dae7c55df73790f2b', 'triton_moe_round12.py': '7ffdc541ad3d480ae3801f7cbc3517031e97e30dde7cca9577b07f5cbc63978b', 'triton_grouped_round12.py': '64b65e5a98003bfbb6bb3c49594d6e5bdd33434ccb36e804afa1e87026164425', 'buffers_moe_round9.py': 'fe562d43198f842299732fc54bd123ebf7c41fc11ae1f6f880f65a0dbd68fb39', 'expert_buffers_round9.cpp': 'ca4dbd1f4d0752d4d5ffde78f803a27284449fc59fa26d315d6b80648d83dc75'})

EXPECTED['checkpoint_policy_round14.py']='205e039f08aafc3088db90b5ff00ca5c1849b3ed084c578be2126b0cdda583f9'

EXPECTED.update({'hybrid_moe_round16.py': 'c72573a963f3c8dc3d3e40b37fc7f472130a826083e4e01dae7c55df73790f2b', 'triton_moe_round12.py': '7ffdc541ad3d480ae3801f7cbc3517031e97e30dde7cca9577b07f5cbc63978b', 'triton_grouped_round12.py': '64b65e5a98003bfbb6bb3c49594d6e5bdd33434ccb36e804afa1e87026164425', 'nf4_moe_round13b.py': 'aa2eec464ce49049e0dc6ea2483fbcb4743e789f4de56acf569cca4cf035566c', 'nf4_grouped_round13b.py': 'ceb9f0ba3b4868567ae46630b5ef0ea76299a5242b921b2db0ae21c5ea6de8d3', 'checkpoint_policy_round14.py': '205e039f08aafc3088db90b5ff00ca5c1849b3ed084c578be2126b0cdda583f9'})

EXPECTED['hybrid_moe_round16.py']='71975d3269f9077dd95c1a882aee05db2822a899c29fd2ec8c8393287d480c1d'
EXPECTED['update_replay_round17.py']='6bb84617dc776e1236b11f6fd6640fd6780324f6beaba5358c6a273cbbccf6d7'

EXPECTED.update({'sam_batch_round23.py': 'ea6a461b88c2342361493bda8135f26174a05730c35171dcf41781a476229096', 'update_replay_round23.py': '82e5f353df8af3d9a8e0acef821d0c75422d071c9f1084537a3dfa676922dec9'})

EXPECTED['update_replay_round24.py']='e5958918e612a232db6c19f89154838631803e6c19360fd95fed6b9dc2f6e836'

EXPECTED['compiled_sam_round26.py']='cc1168dfa22a621afffbfaf9b174bff1405c720f4f56c3b3b1377a1d37989081'
EXPECTED['update_replay_round26.py']='0965ac8e67100fb4d6c0754e6161c02644fefc1af70c62fc1383e4b39f787b3e'

def enable(qwen):
    perf=Path(__file__).resolve().parent/'perf'
    for name,digest in EXPECTED.items():
        assert hashlib.sha256((perf/name).read_bytes()).hexdigest()==digest, name
    if str(perf) not in sys.path:sys.path.insert(0,str(perf))
    from unique_scatter_round3 import install
    blocks=install(qwen,share_storage=True,probe_offload=False)
    import bulk_gather_support_round6 as support
    support.MODE = "bulk_gather"
    from hybrid_moe_round16 import forward
    from buffers_moe_round9 import extension
    extension()
    import fused_decode_round7 as fused
    fused.MODE = "fused_reuse"
    for block in blocks:
        fused.alias_storage(block)
        block.forward=types.MethodType(forward,block)
    import checkpoint_policy_round14 as policy
    import json
    proof=json.loads((perf/'deployment-round17/characterization.json').read_text())
    assert proof['status']=='QUALIFIED_FOR_REAL_UPDATE_COMPARISON_NOT_ACCEPTED_FOR_PRODUCTION'
    assert hashlib.sha256((perf/'hybrid-exclusive-round16.json').read_bytes()).hexdigest()==proof['raw_proof_sha256']
    import hybrid_moe_round16 as hybrid
    hybrid.PREALLOC_ONLY=False
    policy.BYPASS=0;policy.ACTIVE=True
    policy.install(qwen.model)
    return blocks
