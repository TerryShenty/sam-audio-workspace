"""Create an isolated Linux environment and verify actual CPU imports/audio decode."""
import argparse,json,os,pathlib,platform,subprocess,sys,venv,tarfile,urllib.request
P=pathlib.Path(__file__).resolve().parent.parent
SOURCES=[
 'git+https://github.com/facebookresearch/dacvae.git@414c20785fc3a28373073ea8ef7a1316eeeaca6e',
 'git+https://github.com/facebookresearch/ImageBind.git@53680b02d7e37b19b124fa37bae4b6c98c38f5be',
 'git+https://github.com/facebookresearch/perception_models.git@e72b6810b1133e1c879f2cc965d276eb73803f1f',
 'git+https://github.com/lematt1991/CLAP.git@b85cb7ecac0af7d064cbf03ebd990e709e959fe7',
 'git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f']
def main():
 p=argparse.ArgumentParser();p.add_argument('--venv',type=pathlib.Path,required=True);p.add_argument('--torch-index',default='https://download.pytorch.org/whl/cu130');a=p.parse_args()
 assert platform.system()=='Linux' and sys.version_info[:2]==(3,12),'Run on Linux with Python 3.12'
 assert not os.environ.get('SLURM_JOB_GPUS'),'Install on CPU/transfer facilities'
 target=a.venv.expanduser().resolve();venv.EnvBuilder(with_pip=True).create(target);py=target/'bin/python'
 def pip(*args):subprocess.run([str(py),'-m','pip',*args],check=True)
 pip('install','--upgrade','pip','setuptools<81','wheel')
 constraints=target/'qwen-constraints.txt';constraints.write_text((P/'requirements.txt').read_text()+'\ntorch==2.13.0\ntorchvision==0.28.0\ntorchaudio==2.11.0\ntorchcodec==0.16.0\n')
 pip('install','torch==2.13.0','torchvision==0.28.0','torchaudio==2.11.0','--index-url',a.torch_index)
 pip('install','-c',str(constraints),'-r',str(P/'requirements.txt'),'torchcodec==0.16.0','peft','audiobox_aesthetics','pydub','argbind','descript-audiotools','timm','ftfy','iopath','types-regex','omegaconf','msgspec','sentencepiece','tiktoken','blobfile','orjson','webdataset','decord','opencv-python-headless','librosa','torchlibrosa','braceexpand','wget','wandb','scipy','scikit-learn','pandas','h5py','progressbar','av','fvcore','parameterized')
 # Omit unrelated upstream training/evaluation extras and xformers: this cached
 # joint path uses its bundled implementation/SDPA. Check actual imports below.
 pip('install','--no-deps',*SOURCES)
 # Install shared FFmpeg in the private prefix, without system changes.
 assert platform.machine()=='x86_64','This release targets Linux x86_64 single-node servers'
 archive=target/'micromamba.tar.bz2';urllib.request.urlretrieve('https://micro.mamba.pm/api/micromamba/linux-64/latest',archive)
 with tarfile.open(archive) as tf:
  member=tf.getmember('bin/micromamba');assert member.isfile();tf.extract(member,target,filter='data')
 subprocess.run([str(target/'bin/micromamba'),'create','--yes','--no-rc','--override-channels','-c','conda-forge','-p',str(target/'media'),'ffmpeg=7'],check=True,env={**os.environ,'MAMBA_ROOT_PREFIX':str(target/'mamba-root')})
 env={**os.environ,'LD_LIBRARY_PATH':str(target/'media/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),'PATH':str(target/'media/bin')+':'+os.environ['PATH']}
 subprocess.run([str(py),str(P/'scripts/check_environment.py')],check=True,env=env)
 (target/'ENVIRONMENT_READY.json').write_text(json.dumps({'python':str(py),'media':str(target/'media'),'status':'CPU_IMPORT_AUDIO_PASS_GPU_UNTESTED'},indent=2))
 freeze=subprocess.check_output([str(py),'-m','pip','freeze'],text=True)
 (target/'INSTALLED_FREEZE.txt').write_text(freeze)
 print('Environment ready:',py)
if __name__=='__main__':main()
