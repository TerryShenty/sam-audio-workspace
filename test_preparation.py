"""Root bootstrap tests; no model download, installation or GPU required."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('bootstrap',Path(__file__).with_name('bootstrap.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.archive=b.ROOT/b.ARCHIVE
    def extract(self):
        return b.extract_verified(self.archive,self.root/'source')
    def test_all_real_release_files(self):
        self.assertEqual(len(b.verify_source(self.extract())),154)
    def test_repeat_and_changed_source_rejected(self):
        source=self.extract();self.assertEqual(source,self.extract())
        (source/'start.sh').write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'):self.extract()
    def test_corrupt_archive_rejected(self):
        p=self.root/'bad.tar.gz';p.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'SHA-256'):b.extract_verified(p,self.root/'out')
    def test_symlink_source_rejected(self):
        source=self.extract();link=self.root/'link';link.symlink_to(source)
        with self.assertRaisesRegex(ValueError,'symlink'):b.extract_verified(self.archive,link)
    def test_symlink_parent_rejected(self):
        link=self.root/'link';link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlink'):b.checked_directory(link/'out')
    def malicious(self,kind):
        p=self.root/(kind+'.tar.gz')
        with tarfile.open(p,'w:gz') as tf:
            names=['training/../../escape'] if kind=='traversal' else ['training/a']
            if kind=='duplicate':names*=2
            for name in names:
                i=tarfile.TarInfo(name)
                if kind=='symlink':i.type=tarfile.SYMTYPE;i.linkname='/tmp/elsewhere';tf.addfile(i)
                else:i.size=1;tf.addfile(i,io.BytesIO(b'x'))
        with patch.object(b,'ARCHIVE_SHA256',b.digest(p)):
            with self.assertRaisesRegex(ValueError,'Unsafe'):b.extract_verified(p,self.root/'out')
    def test_traversal_rejected(self):self.malicious('traversal')
    def test_link_rejected(self):self.malicious('symlink')
    def test_duplicate_rejected(self):self.malicious('duplicate')
    def test_source_only_never_executes(self):
        with patch.object(b.subprocess,'run',side_effect=AssertionError('execution forbidden')):
            self.assertEqual(b.main(['--directory',str(self.root/'checked'),'--source-only']),0)
    def test_real_shell_help_and_source_only(self):
        for args in [['--help'],['--directory',str(self.root/'shell'),'--source-only']]:
            r=subprocess.run(['bash',str(b.ROOT/'start.sh'),*args],text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)
    def test_execution_arguments_and_python(self):
        with patch.object(b.sys,'version_info',(3,12,2)),patch.object(b.platform,'system',return_value='Linux'),patch.object(b.platform,'machine',return_value='x86_64'),patch.object(b.subprocess,'run') as run:
            run.return_value.returncode=7
            self.assertEqual(b.main(['--directory',str(self.root/'run'),'--wandb-entity','example','--only','train']),7)
            cmd=run.call_args.args[0]
            self.assertEqual(cmd[2:],['--directory',str(self.root/'run/workspace'),'--wandb-entity','example','--only','train'])
            self.assertEqual(run.call_args.kwargs['env']['PYTHON_BIN'],b.sys.executable)
    def test_wrong_platform_before_workspace(self):
        with patch.object(b.platform,'system',return_value='Other'),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):b.main(['--directory',str(self.root/'absent')])
        self.assertFalse((self.root/'absent').exists())

if __name__=='__main__':unittest.main()
