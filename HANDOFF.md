# Release acceptance

The single-server entry has completed new training and public285 feature
generation, independent export, nine real GT updates, three complete validations,
CPU checkpoint audit, independent cloud-metric readback and automatic archival.
CPU/transfer/GPU phases were separated; installed environment/model/raw bytes
were explicitly reused. This does not claim an empty ordinary server completed
one uninterrupted cold invocation.

The base multi-node runtime completed real SAM 0→2 training, checkpoint
save and fresh-process 2→3 resume on two different physical servers (one L40S
and one L40). Saved state and actual updates were audited. This is functional
evidence, not a fast-network or H200/large-cluster performance claim.

One/four/eight-card planning and strict guards have CPU coverage; those GPU
counts are not accepted hardware tests. The separate terminal model-path relocation
fix now passed actual two-A100 step9→10 and complete CPU state inspection. This
requires the same strict-loader model identities and explicit budget-extension
policy; it does not change world size or training inputs. See DELIVERY_STATUS
and the hardware/communication guides for the complete scope.

The package contains no credentials, model weights, checkpoints or private
source maps. Existing historical proof and immutable artifacts remain retained.

## Native two-L40S speed evidence

2026-10-20: repeated actual 112-update runs accepted, ~0.30–0.31 s/update after warmup; full state CPU checks passed. See PERFORMANCE.md/PERFORMANCE.json. The speed experiment used accepted152; default DDP initialization stays enabled. This package additionally contains the separately accepted153 relocation fix.
