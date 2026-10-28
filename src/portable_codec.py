"""Frozen original codec prepared on CPU and loaded without changing training RNG."""
import hashlib,json
from pathlib import Path

def prepare(model,output):
 import torch
 from sam_audio.model.config import SAMAudioConfig
 from sam_audio.model.codec import DACVAE
 model=Path(model);output=Path(output);output.mkdir(exist_ok=True)
 raw=(model/'config.json').read_bytes();source=model/'checkpoint.pt'
 state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
 state={k[len('audio_codec.'):]:v for k,v in state.items() if k.startswith('audio_codec.')}
 rng=torch.get_rng_state()
 try:
  with torch.device('meta'):codec=DACVAE(SAMAudioConfig(**json.loads(raw)).audio_codec)
  codec.load_state_dict(state,strict=True,assign=True);codec.eval().requires_grad_(False)
  with torch.inference_mode(),torch.jit.optimized_execution(False):result=codec.float().decode(torch.zeros(1,128,4))
  assert result.shape==(1,1,7680) and bool(torch.isfinite(result).all())
  torch.save(codec.state_dict(),output/'codec.pt');(output/'config.json').write_bytes(raw)
  receipt={n:{'sha256':hashlib.sha256((output/n).read_bytes()).hexdigest(),'bytes':(output/n).stat().st_size,'mtime_ns':(output/n).stat().st_mtime_ns} for n in ['codec.pt','config.json']}
  (output/'CODEC_READY.json').write_text(json.dumps(receipt,indent=2))
 finally:torch.set_rng_state(rng)

def load(output,device):
 import torch
 from sam_audio.model.config import SAMAudioConfig
 from sam_audio.model.codec import DACVAE
 output=Path(output);receipt=json.loads((output/'CODEC_READY.json').read_text())
 for n,r in receipt.items():
  st=(output/n).stat();assert (st.st_size,st.st_mtime_ns)==(r['bytes'],r['mtime_ns'])
 rng=torch.get_rng_state()
 try:
  with torch.device('meta'):codec=DACVAE(SAMAudioConfig(**json.loads((output/'config.json').read_text())).audio_codec)
  codec.load_state_dict(torch.load(output/'codec.pt',map_location='cpu',weights_only=True,mmap=True),strict=True,assign=True)
  return codec.eval().requires_grad_(False).to(device=device,dtype=torch.bfloat16)
 finally:torch.set_rng_state(rng)
