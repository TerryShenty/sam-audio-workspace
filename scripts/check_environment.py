"""Exercise the actual selected imports and a native audio decoder on CPU."""
import importlib,json,os,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent.parent;V=P/'src'
os.environ.setdefault('QWEN_DATA_ROOT',str(P/'unloaded_data'))
sys.path.insert(0,str(V))
from production_bootstrap import PATHS
sys.path[:0]=[str(p) for p in PATHS]
for n in ['torch','torchaudio','torchcodec','bitsandbytes','transformers','dacvae','imagebind','core','laion_clap','production_models','automatic_validation','portable_codec','validation_scoring']:importlib.import_module(n)
import numpy as np,soundfile as sf,torch
from torchcodec.decoders import AudioDecoder
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'smoke.wav';sf.write(p,np.zeros(4800,dtype=np.float32),48000)
 decoded=AudioDecoder(str(p)).get_all_samples().data
 assert decoded.shape[-1]==4800 and bool(torch.isfinite(decoded).all())
assert not torch.cuda.is_initialized()
print('PASS selected CPU imports and real audio decoding; target GPU validation still required')
