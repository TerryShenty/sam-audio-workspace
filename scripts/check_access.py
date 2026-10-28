"""Verify own-account access before downloading any large gated model files."""
from huggingface_hub import HfApi,hf_hub_download
api=HfApi();user=api.whoami()
for repo in ['Qwen/Qwen3-Omni-30B-A3B-Thinking','facebook/sam-audio-large-tv']:
 hf_hub_download(repo,'config.json')
print('PASS official model access for the signed-in account; credentials are not exported')
