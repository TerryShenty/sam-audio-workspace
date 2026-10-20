"""Run with python3 -m unittest -v test_preparation.py; no GPU or external network."""
import contextlib, io, json, os, shutil, subprocess, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import bootstrap
from types import SimpleNamespace
class PreparationTests(unittest.TestCase):
 def run_entry(self,root,payload):
  def fetch(url):
   if url.endswith(bootstrap.ARCHIVE):return payload
   if '/api/datasets/' in url:return b'{"sha":"test-revision"}'
   return b'{"state":"UPLOADING"}'
  with patch.object(bootstrap,'fetch',side_effect=fetch),patch.object(bootstrap.shutil,'which',return_value=None),patch('sys.argv',['bootstrap.py','--directory',str(root)]),contextlib.redirect_stdout(io.StringIO()):bootstrap.main()
 def test_repeat_and_preserve_environment(self):
  payload=Path(bootstrap.ARCHIVE).read_bytes()
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'run';self.run_entry(root,payload)
   marker=root/'training/.venv/keep';marker.parent.mkdir();marker.write_text('preserve')
   self.run_entry(root,payload);self.assertEqual(marker.read_text(),'preserve')
   self.assertFalse(json.loads((root/'MACHINE_REPORT.json').read_text())['training_ready'])
 def test_changed_source_not_overwritten(self):
  payload=Path(bootstrap.ARCHIVE).read_bytes()
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'run';self.run_entry(root,payload);p=root/'training/run.sh';p.write_text('custom')
   with self.assertRaises(SystemExit):self.run_entry(root,payload)
   self.assertEqual(p.read_text(),'custom')
 def test_bad_hash(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):bootstrap.unpack(b'bad',Path(d))
 def test_validation_preparation_independent_of_training_lookup(self):
  payload=Path(bootstrap.ARCHIVE).read_bytes()
  def fetch(url):
   if url.endswith(bootstrap.ARCHIVE):return payload
   raise OSError('Training release temporarily unavailable')
  with tempfile.TemporaryDirectory() as d:
   with patch.object(bootstrap,'fetch',side_effect=fetch),patch.object(bootstrap.shutil,'which',return_value=None),patch.object(bootstrap.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run,patch('sys.argv',['bootstrap.py','--directory',d,'--prepare-validation']),contextlib.redirect_stdout(io.StringIO()):
    bootstrap.main()
   self.assertTrue(run.call_args.args[0][1].endswith('tools/download_validation_data.py'))
   self.assertIn('dataset_lookup_error',json.loads((Path(d)/'MACHINE_REPORT.json').read_text()))
 def test_avspeech_follows_successful_environment_check(self):
  payload=Path(bootstrap.ARCHIVE).read_bytes()
  def fetch(url):
   if url.endswith(bootstrap.ARCHIVE):return payload
   raise OSError('Optional current-release lookup unavailable')
  with tempfile.TemporaryDirectory() as d,patch.object(bootstrap,'fetch',side_effect=fetch),patch.object(bootstrap.shutil,'which',side_effect=lambda n:'/mock/'+n if n in ['python3.12','git'] else None),patch.object(bootstrap.platform,'platform',return_value='Linux-test'),patch.object(bootstrap.platform,'system',return_value='Linux'),patch.object(bootstrap.platform,'machine',return_value='x86_64'),patch.object(bootstrap.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run,patch('sys.argv',['bootstrap.py','--directory',d,'--install','--prepare-avspeech','--avspeech-revision','a'*40,'--avspeech-ids','clip']),contextlib.redirect_stdout(io.StringIO()):
   bootstrap.main()
   calls=[c.args[0] for c in run.call_args_list]
   self.assertEqual(calls[0],['bash','setup.sh'])
   self.assertIn('preflight.py',calls[1][2])
   self.assertIn('prepare_avspeech_workflow.py',calls[2][2])
   self.assertEqual(calls[2][-4:],['--revision','a'*40,'--ids','clip'])
 def test_invalid_avspeech_selection_stops_before_fetch(self):
  with tempfile.TemporaryDirectory() as d,patch.object(bootstrap,'fetch') as fetch,patch('sys.argv',['bootstrap.py','--directory',d,'--install','--prepare-avspeech','--avspeech-revision','main','--avspeech-all']),contextlib.redirect_stderr(io.StringIO()):
   with self.assertRaises(SystemExit):bootstrap.main()
   fetch.assert_not_called()
 def test_environment_failure_prevents_download_and_audio_preparation(self):
  payload=Path(bootstrap.ARCHIVE).read_bytes()
  def fetch(url):
   if url.endswith(bootstrap.ARCHIVE):return payload
   raise OSError('Optional metadata unavailable')
  with tempfile.TemporaryDirectory() as d,patch.object(bootstrap,'fetch',side_effect=fetch),patch.object(bootstrap.shutil,'which',side_effect=lambda n:'/mock/'+n if n in ['python3.12','git'] else None),patch.object(bootstrap.platform,'platform',return_value='Linux-test'),patch.object(bootstrap.platform,'system',return_value='Linux'),patch.object(bootstrap.platform,'machine',return_value='x86_64'),patch.object(bootstrap.subprocess,'run',return_value=SimpleNamespace(returncode=7)) as run,patch('sys.argv',['bootstrap.py','--directory',d,'--install','--prepare-avspeech','--avspeech-revision','a'*40,'--avspeech-all']),contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaises(SystemExit):bootstrap.main()
   self.assertEqual(run.call_count,1)
class ShellEntryTests(unittest.TestCase):
 """Real shell control flow with stub network/host checks; not a Linux install."""
 def run_shell(self, directory, *, missing_git=False, bootstrap_exit=0, options=()):
  root=Path(directory);bin_dir=root/'bin';bin_dir.mkdir()
  for name in ['mkdir','mktemp','rm','tee']:
   (bin_dir/name).symlink_to(shutil.which(name))
  scripts={
   'git':'#!/bin/bash\nexit 0\n',
   'curl':'#!/bin/bash\nwhile (( $# )); do\n if [[ "$1" == "-o" ]]; then printf "# stub bootstrap\\n" > "$2"; exit 0; fi\n shift\ndone\nexit 2\n',
   'python3.12':'#!/bin/bash\nif [[ "$1" == "-c" ]]; then exit 0; fi\nprintf "%s\\n" "$@" > "$ENTRY_TEST_RECEIPT"\nexit "$ENTRY_TEST_BOOTSTRAP_EXIT"\n',
  }
  for name,script in scripts.items():
   if name=='git' and missing_git:continue
   target=bin_dir/name;target.write_text(script);target.chmod(0o755)
  env=dict(os.environ,PATH=str(bin_dir),ENTRY_TEST_RECEIPT=str(root/'called.txt'),
           ENTRY_TEST_BOOTSTRAP_EXIT=str(bootstrap_exit))
  result=subprocess.run(['/bin/bash',str(Path(__file__).with_name('start.sh').resolve()),
                         str(root/'workspace'),*options],env=env,text=True,capture_output=True,timeout=10)
  return result,root/'called.txt'
 def test_no_system_ffmpeg_reaches_automatic_preparation(self):
  with tempfile.TemporaryDirectory() as d:
   result,receipt=self.run_shell(d)
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)
   self.assertIn('--prepare-validation',receipt.read_text())
   self.assertIn('--install',receipt.read_text())
 def test_missing_git_stops_before_bootstrap(self):
  with tempfile.TemporaryDirectory() as d:
   result,receipt=self.run_shell(d,missing_git=True)
   self.assertEqual(result.returncode,2)
   self.assertIn('Server prerequisite missing: git',result.stderr)
   self.assertFalse(receipt.exists())
   self.assertFalse((Path(d)/'workspace').exists())
 def test_bootstrap_failure_propagates_and_never_reports_ready(self):
  with tempfile.TemporaryDirectory() as d:
   result,receipt=self.run_shell(d,bootstrap_exit=23)
   self.assertTrue(receipt.exists(),result.stderr)
   self.assertEqual(result.returncode,23)
   self.assertIn('Preparation stopped.',result.stdout)
   self.assertNotIn('Validation media and environment prepared.',result.stdout)
 def test_selection_forwarded_without_removing_validation_gate(self):
  options=['--prepare-avspeech','--avspeech-revision','a'*40,'--avspeech-ids','clip']
  with tempfile.TemporaryDirectory() as d:
   result,receipt=self.run_shell(d,options=options)
   self.assertEqual(result.returncode,0,result.stderr)
   args=receipt.read_text().splitlines()
   self.assertIn('--prepare-validation',args)
   self.assertIn('--install',args)
   self.assertEqual(args[-len(options):],options)
if __name__=='__main__':unittest.main()
