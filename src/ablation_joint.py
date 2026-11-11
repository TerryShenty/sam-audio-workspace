"""Four-arm objective on top of the accepted V6 SAMBatchJoint path.

Every arm keeps the V6 per-example loss  .5*FM_target + .5*FM_residual + .1*CE
and the V6 random-time flow state (t ~ U(0,1), independent noise per example).
Arms only ADD terms; nothing in the sealed V6 runtime is edited.

  baseline              V6 loss.
  dual_t0               FM terms are the equal average of the random-t state and a
                        second state at t = 0 (pure noise input) for the same
                        example: FM = .5*FM(t~U) + .5*FM(t=0).
  contrastive           V6 loss + lambda * D, with D the t = 0 A/B difference loss
                        || (v_A - v_B) - (c_A - c_B) ||^2 over all 256 channels.
                        A and B are the two targets of ONE mixture and get the SAME
                        noise, so at t = 0 their inputs are identical and only the
                        prompt can separate the predictions (no x_t shortcut).
  dual_t0_contrastive   dual_t0 FM average + lambda * D; the t = 0 forward (shared
                        noise) serves both terms.

Time convention follows SAM Audio: x_t = (1-(1-sigma)t)*noise + t*clean, so t = 0
is pure noise. Source of D: the seven-module A/B small test
(animal3000_s7_test_20261110/s7_test.py), lambda = 1.
"""
import torch
from sam_batch_round23 import SAMBatchJoint, compatible
from batch2_joint_round4 import condition_batch

ARMS = {
    'baseline': (False, False),
    'dual_t0': (True, False),
    'contrastive': (False, True),
    'dual_t0_contrastive': (True, True),
}
SIGMA_MIN = 1e-5


def fm_terms(pred, velocity, mask):
    from sam_audio.native_finetuning.benchmark_utils import per_target_mse
    t = per_target_mse(pred[..., :128], velocity[..., :128], mask)
    r = per_target_mse(pred[..., 128:], velocity[..., 128:], mask)
    return t, r


def pair_difference_loss(pred0, velocity0, mask):
    """Mean over the valid frames/channels of ((v_A-v_B)-(c_A-c_B))^2; rows 0/1 = A/B."""
    from sam_audio.native_finetuning.benchmark_utils import per_target_mse
    assert pred0.shape[0] == 2 and velocity0.shape[0] == 2
    assert bool((mask[0] == mask[1]).all()), 'A/B pair must share the mixture mask'
    return per_target_mse(pred0[0:1] - pred0[1:2], velocity0[0:1] - velocity0[1:2], mask[0:1]).mean()


def zero_time_states(clean, shared_noise):
    from sam_audio.native_finetuning.objective import make_flow_matching_state
    if shared_noise:
        assert clean[0].shape == clean[1].shape
        noise = torch.randn_like(clean[0])
        noises = [noise, noise]
    else:
        noises = [torch.randn_like(x) for x in clean]
    zero = lambda x: torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    return [make_flow_matching_state(x, sigma_min=SIGMA_MIN, time=zero(x), noise=n) for x, n in zip(clean, noises)]


class AblationJoint(SAMBatchJoint):
    arm = 'baseline'
    contrastive_weight = 1.0

    def configure_arm(self, arm, contrastive_weight=1.0):
        assert arm in ARMS, arm
        assert contrastive_weight >= 0
        self.arm = arm
        self.contrastive_weight = float(contrastive_weight)
        self.last_terms = None
        return self

    def forward(self, item, args, clean, weight=1.):
        dual, contrast = ARMS[self.arm]
        if self.arm == 'baseline' or not isinstance(item, list):
            out = super().forward(item, args, clean, weight)
            self.last_terms = None
            return out
        from sam_audio.native_finetuning.objective import make_flow_matching_state
        assert len(item) == len(args) == len(clean) == len(weight) == 2
        assert compatible(args, clean), 'Ablation arms require a shape-compatible A/B pair'
        if contrast:
            assert item[0]['pair_id'] == item[1]['pair_id'], 'Contrastive pair must come from one mixture'
            assert [item[0]['role'], item[1]['role']] == ['A', 'B'], 'Pair order must be A then B'
        cond, ce, acc, lengths, _ = condition_batch(self, item)
        packed = {k: torch.cat([a[k] for a in args], 0) if torch.is_tensor(v) else v for k, v in args[0].items()}
        assert 'text_features' not in packed and 'text_mask' not in packed
        device = next(self.sam.parameters()).device
        mask = packed['audio_pad_mask']
        text_mask = torch.ones((2, 1), dtype=torch.bool, device=device)

        def run(states):
            pred = self.sam(noisy_audio=torch.cat([s.noisy_latent for s in states], 0),
                            time=torch.cat([s.time for s in states]),
                            text_features=cond.to(device), text_mask=text_mask, **packed)
            return pred, torch.cat([s.target_velocity for s in states], 0)

        random_states = [make_flow_matching_state(x, sigma_min=SIGMA_MIN) for x in clean]
        pred, velocity = run(random_states)
        t, r = fm_terms(pred, velocity, mask)
        t_random, r_random = t.detach(), r.detach()
        pred0, velocity0 = run(zero_time_states(clean, shared_noise=contrast))
        t0, r0 = fm_terms(pred0, velocity0, mask)
        if dual:
            t, r = .5 * (t + t0), .5 * (r + r0)
        difference = pair_difference_loss(pred0, velocity0, mask) if contrast else torch.zeros((), device=device)
        t, r, difference = t.to(ce.device), r.to(ce.device), difference.to(ce.device)
        w = torch.as_tensor(weight, device=ce.device, dtype=t.dtype)
        loss = ((.5 * t + .5 * r + .1 * ce) * w).sum() + self.contrastive_weight * difference
        self.sam_batch_used = True
        self.last_terms = dict(fm_target_random=t_random, fm_residual_random=r_random,
                               fm_target_t0=t0.detach(), fm_residual_t0=r0.detach(),
                               pair_difference=difference.detach())
        return loss, torch.stack([t.detach(), r.detach(), ce.detach(), acc.detach()], dim=1), lengths
