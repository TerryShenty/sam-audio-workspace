# Archived Qwen30B four-GPU V5 portable source

This archive preserves the previous V5 implementation as a debugging fallback. Prefer the V6 package and QWEN30B_V6_README.md at the repository root for the current handoff.

Included: selected transitive runtime sources, four native C++ extension sources, environment notes, official model download/hash checks, CPU preparation, four-GPU qualification/training/resume launchers and validation implementation. Data and model weights are separate.

Run verify_package.py, read ENVIRONMENT.md, install the required environment, download official models with download_models.py, and set data_root/sam_namespace/qwen_model/sam_model in your local paths configuration. Then run prepare.py --paths my_paths.json on CPU resources.

Export QWEN_PACKAGE, QWEN_PYTHON, QWEN_OUTPUT and QWEN_MODE=qualify; submit train.sbatch with your cluster account/partition. After actual successful qualification, set QWEN_MODE=train and QWEN_GATE to its PORTABLE_GATE.json, choose a new output directory and submit with the authorized wall time. QWEN_RESUME may reference a complete formal-training checkpoint of this same deployment. Do not use research state as production initialization.

Limitations: static checks only for this historical archive; target four-GPU execution and full817 validation were not established. The V5 export lacks the later V6 data downloader and automatic local validation-path binding. Its validation API requires target codec/memory integration, and the training CLI does not automatically run817 validation. Historical cache-identity path strings must not be globally replaced.

The V5 single-A100/global2/fixed64 mean0.841817s is not a four-GPU guarantee. No automatic speed-search sweep is launched. The repository-root SOLOs quickstart is unrelated. Never disable source, memory, or checkpoint guards to hide errors.
