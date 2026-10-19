"""Run with python3 -m unittest -v test_preparation.py; no GPU or external network."""
import contextlib, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import bootstrap
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
if __name__=='__main__':unittest.main()
