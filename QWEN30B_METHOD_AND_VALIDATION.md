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
- The automatic handoff calls all817 validation after qualification, after every half epoch and at the final formal checkpoint when not already scored. It first tests the longest actual token/video/audio cases with retained training state, codec and scoring; then enforces memory headroom throughout full coverage. Initial per-rank batch1 is conservative and explicitly differs from the unqualified batch8 preference. The final30 minutes are reserved for validation within the fixed walltime. No target four-GPU or complete817 result has been established by source/CPU checks.

## V6 acceleration
V5 mechanisms remain active. V6 adds dynamic INT8 gradient-row quantization and transposed INT8 weight caches for Qwen attention input gradients. This introduces an additional approximation; matching a few forward losses does not establish held-out quality equivalence. Single A100/global2/fixed64, two matched full32 pairs: control0.8228534477s/update; V6 0.8092989110s/update, mean gain1.64726%. No four-GPU speed claim. Long-input per-device physical memory, aggregate host memory and full-state continuation must be tested on the recipient cluster.


## Output files and sharing results
Default formal output: `<root>/runs/train/` (override with `--output`).
- `training-rank0.csv` through `training-rank3.csv`: recorded local loss, gradient norm, step and approximate processed epochs. Warmup updates are included in step numbers but not CSV rows; resumed epochs are approximate.
- `VALIDATION817-summary.csv`: overall and per-dataset means, sample counts, training step, checkpoint identity and measured validation seconds (excluding the preliminary memory probe).
- `VALIDATION817-samples.csv`: all817 per-example scores and identities.
- `VALIDATION817.json`: authoritative complete-coverage validation receipt.
- `checkpoints/step-*/COMPLETE.json` and rank state files: complete resumable checkpoints. CSV files do not replace checkpoints.
Send the summary CSV and training-rank0.csv for a quick review; include all four training CSVs to investigate rank differences. Validation CSVs are generated only after complete817 coverage. Missing validation CSV does not mean a passing validation.
Validation now runs every half epoch based on committed global sampler progress (111,068 rows; global batch8), plus final validation when the final state has not already been scored. Boundaries occur after updates6,942,13,884,20,826,...; sample-based thresholds avoid cumulative rounding drift. Each boundary saves a complete checkpoint, runs all817 with the same scoring protocol, then resumes training. Fixed walltime takes priority near the deadline. Validation receipts are under `validation/step-*.json`, with matching `-summary.csv` and `-samples.csv`. The final summary remains `VALIDATION817-summary.csv`. A resumed checkpoint at an unfinished boundary is validated before further updates. No model, loss, optimizer, V6 kernel or validation solver change is made by this scheduling update. Target GPU execution remains pending.
