"""Fresh pretrained joint construction. No benchmark checkpoints or preloading.

Caller must install production_bootstrap and execution policy before CUDA,
validate completed input/source admission, and supply a rank-local CUDA device.
This module does not authorize a launch or assert target-device readiness.
"""
import gc
import torch
from bridge import QwenBridge,load_sam
from common import key
from sam_batch_round23 import SAMBatchJoint
from production_data import install_qwen_cache

EXPECTED_GROUPS={'qwen':194,'bridge':6,'sam_lora':484,'sam_conditions':15}

def training_groups(joint):
    groups={name:[] for name in EXPECTED_GROUPS}
    for name,param in joint.named_parameters():
        if not param.requires_grad:continue
        group=('qwen' if name.startswith('vlm.') else 'bridge' if name.startswith('bridge.')
               else 'sam_lora' if '.lora_' in name else 'sam_conditions')
        groups[group].append((name,param))
    assert {k:len(v) for k,v in groups.items()}==EXPECTED_GROUPS
    return groups

def construct(device,*,cache_root,media,cache_identity,processor_bytes,av_bytes):
    from execution_speed_v29 import assert_policy
    assert_policy()
    assert device.type=='cuda' and device.index==torch.cuda.current_device()
    assert len(media)==12161 and processor_bytes>0 and av_bytes>0
    torch.manual_seed(314158)
    qwen=QwenBridge(device)
    assert qwen.identity==cache_identity,'Frozen AV/model identity mismatch'
    caches=install_qwen_cache(qwen,cache_root,media,cache_identity,key,
        processor_bytes=processor_bytes,av_bytes=av_bytes,pinned=True)
    from native_extension_loader import install as load_native_extensions
    load_native_extensions()
    from fast_moe import enable
    enable(qwen.qwen)
    from alias_nf4_metadata_v1 import install as alias_metadata
    alias=alias_metadata(qwen.qwen)
    gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache()
    import nf4_grouped_round13b as kernel,dynamic_nf4 as dynamic
    dynamic.WAVES=4;dynamic.GROUP_M=1;kernel.grouped_mm=dynamic.grouped_mm
    sam,config,tuning=load_sam(device)
    torch.manual_seed(314160)
    joint=SAMBatchJoint(sam,qwen,device)
    groups=training_groups(joint)
    assert all(not p.requires_grad for p in qwen.qwen.visual.parameters())
    assert all(not p.requires_grad for p in qwen.qwen.audio_tower.parameters())
    evidence=dict(initialization='original_pretrained_fresh_adapters',checkpoint_loaded=False,
        groups={k:len(v) for k,v in groups.items()},alias=alias,
        qwen_cache_identity=qwen.identity,processor_cache_bytes=processor_bytes,av_cache_bytes=av_bytes,
        allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
        free=torch.cuda.mem_get_info(device)[0])
    return joint,tuning,caches,evidence

def optimizer(joint):
    groups=training_groups(joint)
    rates={'qwen':2e-5,'bridge':1e-4,'sam_lora':1e-4,'sam_conditions':1e-5}
    return torch.optim.AdamW([dict(params=[p for _,p in values],lr=rates[name],name=name)
        for name,values in groups.items()],betas=(.9,.95),eps=1e-8,weight_decay=.01,fused=True)
