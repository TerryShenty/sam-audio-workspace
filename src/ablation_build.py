"""Compose the V6 runtime with the ablation data, sampler and objective.

Mirrors production_build.build / production_models.construct; differences:
  - data contract from ablation_data (no fixed 12,161 / 111,068 / 817 counts),
  - AblationJoint (arm-selected objective) instead of SAMBatchJoint,
  - PairSampler (one A/B mixture per rank per update) for every arm.
Model construction, kernels, optimizer groups/learning rates, scheduler, seeds,
DDP wrapper, trainer and checkpoint format are the V6 ones.
"""
import gc, hashlib, random


def construct(device, *, cache_root, media, cache_identity, arm, contrastive_weight, processor_bytes=128 * 2**20, av_bytes=512 * 2**20):
    import torch
    from execution_speed_v29 import assert_policy
    from bridge import QwenBridge, load_sam
    from common import key
    from production_data import install_qwen_cache
    from production_models import training_groups
    from ablation_joint import AblationJoint
    assert_policy()
    assert device.type == 'cuda' and device.index == torch.cuda.current_device() and media
    torch.manual_seed(314158)
    qwen = QwenBridge(device)
    assert qwen.identity == cache_identity, 'Frozen AV/model identity mismatch'
    caches = install_qwen_cache(qwen, cache_root, media, cache_identity, key, processor_bytes=processor_bytes, av_bytes=av_bytes, pinned=True)
    from native_extension_loader import install as load_native_extensions
    load_native_extensions()
    from fast_moe import enable
    enable(qwen.qwen)
    from alias_nf4_metadata_v1 import install as alias_metadata
    alias = alias_metadata(qwen.qwen)
    gc.collect(); torch.cuda.synchronize(device); torch.cuda.empty_cache()
    import nf4_grouped_round13b as kernel, dynamic_nf4 as dynamic
    dynamic.WAVES = 4; dynamic.GROUP_M = 1; kernel.grouped_mm = dynamic.grouped_mm
    sam, config, tuning = load_sam(device)
    torch.manual_seed(314160)
    joint = AblationJoint(sam, qwen, device).configure_arm(arm, contrastive_weight)
    groups = training_groups(joint)
    assert all(not p.requires_grad for p in qwen.qwen.visual.parameters())
    assert all(not p.requires_grad for p in qwen.qwen.audio_tower.parameters())
    evidence = dict(initialization='original_pretrained_fresh_adapters', checkpoint_loaded=False, arm=arm,
                    contrastive_weight=contrastive_weight, groups={k: len(v) for k, v in groups.items()}, alias=alias,
                    qwen_cache_identity=qwen.identity, allocated=torch.cuda.memory_allocated(device),
                    free=torch.cuda.mem_get_info(device)[0])
    return joint, tuning, caches, evidence


def build(*, root, device, contract_path, immutable_identity, sam_namespace, configuration, arm, contrastive_weight):
    from deployment_guard import configuration as check_configuration, GROUPS
    check_configuration(contract_path, configuration, GROUPS)
    world = configuration['world_size']
    assert 1 <= world <= 8 and configuration['per_rank_batch'] == 2 and configuration['global_batch'] == 2 * world
    from ablation_data import load, AblationBatch
    data = load(root)
    import numpy as np, torch, torch.distributed as dist
    assert dist.is_initialized() and dist.get_world_size() == world and dist.get_backend() == 'nccl'
    from production_models import optimizer
    from production_runtime import ProductionRuntime
    from joint_ddp import wrap
    from production_data import ProductionStore
    from distributed_trainer import DistributedTrainer
    from production_session import ProductionSession
    from ablation_sampler import PairSampler
    joint, tuning, caches, evidence = construct(device, cache_root=root / 'qwen_cache', media=data['media'],
                                                cache_identity=data['processor']['identity'], arm=arm,
                                                contrastive_weight=contrastive_weight)
    check_configuration(contract_path, configuration, evidence['groups'])
    runtime = ProductionRuntime(joint, max_tokens=data['processor']['max_actual_tokens'], contract_path=contract_path)
    wrapped, ddp = wrap(joint, device=device, immutable_identity=immutable_identity)
    opt = optimizer(joint)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: min((n + 1) / 50, 1.))
    rank = dist.get_rank(); seed = 20261027 + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    sampler = PairSampler(data['train'], rank=rank, seed=20261110, data_identity=data['identity'], world_size=world)
    identity = hashlib.sha256((immutable_identity + data['identity'] + arm + repr(contrastive_weight)).encode()).hexdigest()
    trainer = DistributedTrainer(wrapped, opt, scheduler, sampler, identity=identity, configuration=configuration)
    runtime.activate_after_training_state_initialized()
    store = ProductionStore(sam_namespace, {}, limit=512 * 2**20, pinned=True)
    batch = AblationBatch(data['train'], store, device)
    session = ProductionSession(trainer, runtime, batch)
    return session, dict(model=evidence, ddp=ddp, data=data, tuning=tuning, processor_cache=caches[0], av_cache=caches[1], sam_cache=store.cache)
