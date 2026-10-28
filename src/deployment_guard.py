"""Fail closed before launch/resume. A version label never proves activation."""
import hashlib, json
from pathlib import Path

BASELINE_SHA='266b60de355a03a401147ed812b09369689cffb411935a480bbdb8dd1fb600ce'
GROUPS={'qwen':194,'bridge':6,'sam_lora':484,'sam_conditions':15}
FORBIDDEN=('atomic_route_replay_enabled','constant_codebook_enabled','forward_same_expert64_enabled','sam_wo_int8_enabled','scaled_gate_up_enabled','activation_offload_enabled')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def configuration(contract_path,effective,groups):
    assert sha(contract_path)==BASELINE_SHA,'UNACCEPTED_BASELINE_CONTRACT'
    contract=json.loads(Path(contract_path).read_text())
    for k,v in contract['configuration'].items():
        if k!='name':assert effective.get(k)==v,('BASELINE_CONFIG_MISMATCH',k)
    assert groups==GROUPS,('TRAINING_GROUPS_CHANGED',groups)
    assert effective.get('lora_intermediate_cache_cap_bytes')==0
    for k in FORBIDDEN:assert effective.get(k) is False,('UNACCEPTED_OPTION',k)
    assert effective.get('qwen_precision')=='nf4_doublequant_bf16'
    assert effective.get('qwen_lora_rank')==8 and effective.get('qwen_lora_projections')==['q_proj','v_proj']
    assert effective.get('language_hidden_cached') is False
    return contract

def source_binding(receipt,loaded_modules,entry):
    assert {'bp_backend','bp_dispatch','bp_live','bp_registry','bp_lifecycle','bytepair_launch','bytepair_kernel','word8_launch','word8_decode_kernel','word4_launch','word4_decode_kernel','nf4_attention_policy','nf4_attention_backward_policy','a30_adapter','int8_fused_rowquant_linear','int8_fused_dynamic_kernel','int8_rowquant_kernel'}<=set(receipt['modules']),'INCOMPLETE_V2_SOURCE_BINDING'
    assert receipt['contract_sha256']==BASELINE_SHA
    assert str(Path(entry).resolve())==receipt['entry']
    for name,expected in receipt['modules'].items():
        assert name in loaded_modules,('MODULE_NOT_LOADED',name)
        actual=Path(loaded_modules[name].__file__).resolve()
        assert str(actual)==expected['path'],('RESOLVED_IMPORT_CHANGED',name,str(actual))
        assert sha(actual)==expected['sha256'],('SOURCE_CHANGED',name)
    assert sha(entry)==receipt['entry_sha256']

def actual_activation(before,after,steps,max_tokens,long_input_receipt=None):
    assert steps>0
    assert before['attention']['enabled'] and after['attention']['enabled']
    assert after['attention']['bases']==192 and after['attention']['calls']>before['attention']['calls'],'ATTENTION_FASTPATH_NOT_EXECUTED'
    assert before['attention']['backward_int8_enabled'] and after['attention']['backward_int8_enabled']
    assert after['attention']['backward_dispatch_forward_calls']>before['attention']['backward_dispatch_forward_calls'],'BACKWARD_DISPATCH_NOT_EXECUTED'
    bytepair_activation(before['bytepair'],after['bytepair'])
    bytepair_activation(before['word4'],after['word4'])
    bytepair_activation(before['word8'],after['word8'])
    assert after['retained_layers']==23 and after['memo_cap_bytes']==3623878656
    assert after['language_updates']-before['language_updates']==steps
    assert after['optimizer_updates']-before['optimizer_updates']==steps
    assert after['retention_fallback_calls']-before['retention_fallback_calls']==0,'SILENT_LONG_INPUT_RETENTION_FALLBACK'
    for k in ['native_rms_graph_executions','sam_compiled_calls','qwen_compiled_calls','route_input_calls','route_backward_calls','route_aggregate_calls','pointwise_calls','memo_hits']:
        assert after[k]>before[k],('FASTPATH_NOT_ACTIVATED',k)
    assert set(after['finite_updated_groups'])==set(GROUPS)
    if max_tokens>1024:
        assert long_input_receipt and long_input_receipt['status']=='PASS_TARGET_COMPLETE_UPDATES'
        assert long_input_receipt['max_tokens']>=max_tokens
        assert long_input_receipt['retained']==23 and long_input_receipt['memo_active']
        assert long_input_receipt['physical_headroom_pass'] and long_input_receipt['all_ranks_compute']

def full_restore(manifest,world_size):
    assert world_size==4 and manifest['world_size']==4
    assert manifest['complete'] is True and manifest['contract_sha256']==BASELINE_SHA
    required={'model','optimizer','scheduler','ema','rng','data_position','sampler','config','identity'}
    assert required<=set(manifest['components'])
    assert sorted(manifest['rank_ids'])==list(range(4))
    assert manifest['effective_continuation_updates']>0 and manifest['all_groups_updated']

def bytepair_activation(before,after):
    """Require every accepted NF4 forward role to execute in the measured window."""
    assert before['enabled'] is True and after['enabled'] is True,'BYTEPAIR_DISABLED'
    roles={'gate_proj_forward','up_proj_forward','down_proj_forward'}
    assert set(before['calls'])==roles and set(after['calls'])==roles,'BYTEPAIR_ROLE_MISSING'
    for role in roles:
        first,last=before['calls'][role],after['calls'][role]
        assert type(first) is int and type(last) is int and 0<=first<last,('BYTEPAIR_NOT_EXECUTED',role)
