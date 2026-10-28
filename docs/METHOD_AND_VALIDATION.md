# Joint fine-tuning and validation actually implemented

## Training
- 111,068 final prompt rows: music 90,065 + speech 21,003; 817 held-out rows are excluded from the training sampler.
- Qwen3-Omni-30B-A3B-Thinking: original packed NF4 double-quantized BF16 language base, trainable q/v LoRA rank8 and special-token parameters. Frozen audio/video towers are cached; language hidden states are recomputed with gradients.
- The causal hidden state at the position predicting `<AUDSEG>` is projected by a trainable bridge into one SAM conditioning token. SAM's old cached text embedding is removed. This is fixed-template token conditioning, not a free-text generation benchmark.
- SAM uses original pretrained initialization with trainable LoRA and conditioning modules. Optimizer groups contain 194 Qwen, 6 bridge, 484 SAM LoRA and 15 SAM conditioning tensors. These are tensor counts, not parameter counts. AdamW group learning rates are 2e-5, 1e-4, 1e-4 and 1e-5.
- Per-example loss: 0.5 target flow-matching MSE + 0.5 residual flow-matching MSE + 0.1 Qwen template cross-entropy. Each rank averages two examples; four-rank gradient synchronization gives global batch8.
- Speech training keeps the bound original span annotations. No research-trained checkpoint initializes production. Checkpoints preserve model, optimizer, scheduler, EMA, rank RNG, sampler/data position, configuration and identities.

## Validation
- Exactly URMP412, ChoralSynth120, EasyCom207, MSDWild78 = 817. Original sample IDs, prompts, scoring protocols and per-example noise seeds are bound.
- Uses live weights, evaluation mode and no gradients; restores RNG and module modes afterward. It computes the Qwen causal condition from the final prompt and cached audiovisual input.
- SAM integrates flow from fixed-seed noise with midpoint solver, 16 steps; codec decodes only each sample's actual latent length. Validation does not inject target training spans.
- Scores original GT and mixture waveforms: SI-SDR, aligned SI-SDR, SI-SDR improvement over mixture, activity F1 and silent failures. It preserves original true-GT/near-GT distinctions. This measures audio separation, not general Qwen language/video understanding.
- Frozen feature caches suffice for conditioning/generation. Original held-out GT and mixture audio remain necessary for scoring; original videos are not required for this cached validation path.
- The automatic handoff now calls all817 validation after qualification and after the final formal checkpoint. It first tests the longest actual token/video/audio cases with retained training state, codec and scoring; then enforces memory headroom throughout full coverage. Initial per-rank batch1 is conservative and explicitly differs from the unqualified batch8 preference. The final30 minutes are reserved for validation within the fixed walltime. No target four-GPU or complete817 result has been established by source/CPU checks.

## V6 acceleration
V5 mechanisms remain active. V6 adds dynamic INT8 gradient-row quantization and transposed INT8 weight caches for Qwen attention input gradients. This introduces an additional approximation; matching a few forward losses does not establish held-out quality equivalence. Single A100/global2/fixed64, two matched full32 pairs: control0.8228534477s/update; V6 0.8092989110s/update, mean gain1.64726%. No four-GPU speed claim. Long-input per-device physical memory, aggregate host memory and full-state continuation must be tested on the recipient cluster.
