"""Fixed accepted joint runtime, isolated long-input selectors; GPU gate pending.

Construct after original pretrained model setup. Activate only after optimizer,
scheduler, EMA and RNG initialization. Wrap the whole trainer step with guard_step.
"""
import copy,hashlib,json,types
import torch

class ProductionRuntime:
    def __init__(self,joint,*,max_tokens,contract_path):
        from deployment_guard import BASELINE_SHA
        assert hashlib.sha256(contract_path.read_bytes()).hexdigest()==BASELINE_SHA
        self.config=copy.deepcopy(json.loads(contract_path.read_text())['configuration'])
        self.joint=joint;self.qwen=joint.vlm
        from singleton_attention_v2 import install
        self.cross=[layer.cross_attention for layer in joint.sam.transformer.layers]
        for module in self.cross:install(module);module.singleton_enabled=False
        from native_sam_candidate_v2.native_sam_policy import NativeSAMPolicy
        from sam_audio.model.transformer import Attention,RMSNorm
        self.native=NativeSAMPolicy(joint.sam.transformer,Attention,RMSNorm)
        from long_input_policy import policy_classes,guard_model_body
        Retention,Registry=policy_classes()
        self.retention=Retention(self.qwen.qwen.model,max_tokens=max_tokens)
        self.original_body=guard_model_body(self.qwen.qwen.model,max_tokens)
        import compiled_sam_round26 as compiled
        from sam_b1_policy import B1Policy
        compiled.enable(joint.sam.transformer)
        self.b1=B1Policy(joint.sam.transformer);self.b1.set(True)
        import dynamic_nf4 as dynamic
        import dynamic_nf4_static_v1 as static
        import routing_replay_candidate_v58 as routing
        from route_input_runtime_v57 import RouteInputRuntime
        from route_backward_runtime_v55 import RouteBackwardRuntime
        from route_aggregate_runtime_v53 import RouteAggregateRuntime
        from pointwise_runtime_v50 import PointwiseRuntime
        self.runtimes=dict(input=RouteInputRuntime(routing),backward=RouteBackwardRuntime(routing),
                           aggregate=RouteAggregateRuntime(routing),pointwise=PointwiseRuntime(routing))
        self.blocks=[m for m in self.qwen.qwen.modules() if hasattr(m,'_packed_nf4')]
        assert len(self.blocks)==48
        self.originals=[m.forward for m in self.blocks]
        import nf4_grouped_round13b as kernel
        from replay_observation_v58h import ObservedLayerReplayMemo
        self.memo=ObservedLayerReplayMemo(max_bytes=0);routing.replay_memo=self.memo
        self.registry=Registry(self.qwen.qwen.model,self.memo,max_tokens=max_tokens)
        from bp_backend import BackendStack
        base=dict(name='matched_fused_pinned',singleton=False,cache='pinned',retained=0,waves=4,
                  group=1,static=False,native_sam_mode='existing',compiler_mode='existing')
        self.stack=BackendStack(dynamic,static,routing,kernel,self.blocks,self.memo,base)
        from matched_route_control_v58i import MatchedRouteControl
        import routing_input_prepare_v57 as accepted_routing
        self.route_control=MatchedRouteControl(self.stack,accepted_routing)
        from production_regional_compile import RegionalCompiler
        from compiler_origin_install_v58c import origin_compiler_class
        from transformers.utils.deprecation import deprecate_kwarg
        import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as installed_qwen
        self.compiler=origin_compiler_class(RegionalCompiler)(joint.sam.transformer.layers,
            self.qwen.qwen.model.layers,deprecate_kwarg,vars(installed_qwen))
        from memory_policy_v13 import RetainedOffloadPolicy
        self.offload=RetainedOffloadPolicy(self.qwen.qwen.model.layers,self.blocks)
        from bp_lifecycle import WorkerLifecycle
        self.lifecycle=WorkerLifecycle(self.registry,self.stack,self.compiler,self.runtimes,self.offload)
        from a30_adapter import AttentionRuntime
        self.attn=AttentionRuntime(self.qwen.qwen.model.layers,self.lifecycle)
        self.activated=False

    def activate_after_training_state_initialized(self):
        assert not self.activated
        from bp_identity import normalize
        from lora_policy_v21 import JointCheckpointPolicy
        config=normalize(self.config)
        assert not config.get('checkpoint_atomic_route_replay',False)
        assert config['retained']==23 and config['memo_bytes']==3623878656
        def mutate(cfg):
            self.offload.remove()
            for runtime in self.runtimes.values():runtime.configure(cfg)
            self.attn.configure(cfg.get('qwen_nf4_attention_forward_int8_30b_v1',False),backward=cfg.get('qwen_nf4_attention_backward_int8_30b_v1',False))
            self.stack.configure(cfg);self.route_control.select(False)
            JointCheckpointPolicy.configure(self.registry,0);self.memo.set_enabled(False)
            self.registry.first=cfg['retained']
            self.registry.count=cfg.get('memo_layers',48-cfg['retained'])
            self.native.configure(cfg['native_sam_mode'])
            self.compiler.configure(cfg['compiler_mode'],cfg['native_sam_mode'])
            self.retention.configure(cfg)
            for module in self.cross:module.singleton_enabled=cfg['singleton']
            for module,original in zip(self.blocks,self.originals):
                module.forward=types.MethodType(self.stack.forward(original),module)
            self.offload.configure(0);self.joint.train();self.qwen.audit_flow_gradient=False
        self.lifecycle.initialized()
        self.lifecycle.configure(lambda:config,mutate)
        self.activated=True

    def guard_step(self,operation):
        assert self.activated
        return self.lifecycle.guard_step(operation)

    def statistics(self):
        return dict(attention=self.attn.statistics(),retention=self.retention.statistics(),memo=self.memo.statistics(),
                    routes={k:v.statistics() for k,v in self.runtimes.items()},
                    compiler=self.compiler.statistics(),native=self.native.statistics(),
                    bytepair=dict(enabled=self.stack.current.get('nf4_bytepair_decode_v1') is True,
                                  calls=dict(self.stack.row_reuse.bp_counts)),
                    word8=dict(enabled=self.stack.current.get('nf4_word8_decode_v1') is True,calls=dict(self.stack.row_reuse.eight_counts)),
                    word4=dict(enabled=self.stack.current.get('nf4_word4_decode_v1') is True,calls=dict(self.stack.row_reuse.word_counts)),
                    atomic_route_replay=self.memo.enabled,activated=self.activated)
