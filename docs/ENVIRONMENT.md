# Automatic environment setup for one server with 1–8 approximately 48GB GPUs

Run `python3.12 scripts/setup_environment.py --venv /YOUR_STORAGE/environment` on Linux x86_64 CPU facilities with network access. This installs a private virtual environment, the original CUDA13.0 PyTorch2.13/torchvision0.28/torchaudio2.11 stack, selected runtime dependencies, immutable upstream source revisions, and FFmpeg7 shared libraries using conda-forge micromamba. It does not change system packages. A C++ compiler and git must be available through the site's modules. The NVIDIA driver must support the selected CUDA runtime.

The installer runs selected real imports and a TorchCodec WAV decode before writing ENVIRONMENT_READY.json, and records pip freeze. TorchAudio2.11 supports PyTorch2.11 and later through its stable ABI; TorchCodec0.16 supports PyTorch>=2.11. References: https://docs.pytorch.org/audio/stable/installation and https://github.com/meta-pytorch/torchcodec . Original source revisions were read from the running environment metadata. The private FFmpeg library path is exported automatically by the entry and Slurm script.

CPU6927574 exposed missing system FFmpeg shared libraries. The affected retry uses the known working FFmpeg prefix; neither receipt establishes a clean install or CUDA correctness on the recipient server. Full fresh-install and target GPU qualification are separate checks. Unsupported driver/ABI, unavailable pinned wheels or missing compiler errors stop setup; they never silently change the accepted training implementation.

## Bring your own environment

The automatic installer is optional. Use Linux/Python 3.12, a CUDA PyTorch stack appropriate for your driver, the packages in `requirements.txt`, and the additional packages and pinned upstream dependencies listed in `scripts/setup_environment.py`. That script is the complete reference recipe, not a claim that other version combinations are supported. Keep FFmpeg shared libraries visible for TorchCodec.

```bash
python3.12 qwen_quickstart.py setup --root /YOUR_STORAGE/qwen-sam --python /YOUR_ENV/bin/python
/YOUR_ENV/bin/hf auth login
python3.12 qwen_quickstart.py prepare --root /YOUR_STORAGE/qwen-sam --python /YOUR_ENV/bin/python
python3.12 qwen_quickstart.py train --root /YOUR_STORAGE/qwen-sam --python /YOUR_ENV/bin/python --account YOUR_ACCOUNT --partition YOUR_PARTITION --gpus 4
```

With `--python`, setup does not install packages; it exercises imports and waveform decoding. Fix missing libraries in your own environment, then rerun the affected step. No GPU job is submitted by setup or prepare.
