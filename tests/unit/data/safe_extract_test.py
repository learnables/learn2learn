#!/usr/bin/env python3

import os
import shutil
import tarfile
import tempfile
import unittest
import zipfile

from learn2learn.data.utils import safe_extract


class SafeExtractTests(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.source = os.path.join(self.tmpdir, 'source.txt')
        with open(self.source, 'w') as f:
            f.write('learn2learn')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def make_dir(self, name):
        path = os.path.join(self.tmpdir, name)
        os.makedirs(path)
        return path

    def test_benign_tar_extracts(self):
        archive_path = os.path.join(self.tmpdir, 'benign.tar')
        with tarfile.open(archive_path, 'w') as tar:
            tar.add(self.source, arcname='sub/data.txt')
        destination = self.make_dir('tar_out')
        with tarfile.open(archive_path) as tar:
            safe_extract(tar, destination)
        self.assertTrue(
            os.path.exists(os.path.join(destination, 'sub', 'data.txt'))
        )

    def test_benign_zip_extracts(self):
        archive_path = os.path.join(self.tmpdir, 'benign.zip')
        with zipfile.ZipFile(archive_path, 'w') as zf:
            zf.writestr('sub/data.txt', 'learn2learn')
        destination = self.make_dir('zip_out')
        with zipfile.ZipFile(archive_path) as zf:
            safe_extract(zf, destination)
        self.assertTrue(
            os.path.exists(os.path.join(destination, 'sub', 'data.txt'))
        )

    def test_malicious_tar_blocked(self):
        archive_path = os.path.join(self.tmpdir, 'malicious.tar')
        with tarfile.open(archive_path, 'w') as tar:
            tar.add(self.source, arcname='../escaped.txt')
        destination = self.make_dir('tar_mal_out')
        with tarfile.open(archive_path) as tar:
            self.assertRaises(Exception, safe_extract, tar, destination)
        escaped = os.path.join(self.tmpdir, 'escaped.txt')
        self.assertFalse(os.path.exists(escaped))

    def test_malicious_zip_blocked(self):
        archive_path = os.path.join(self.tmpdir, 'malicious.zip')
        with zipfile.ZipFile(archive_path, 'w') as zf:
            zf.writestr('../escaped.txt', 'pwned')
        destination = self.make_dir('zip_mal_out')
        with zipfile.ZipFile(archive_path) as zf:
            self.assertRaises(Exception, safe_extract, zf, destination)
        escaped = os.path.join(self.tmpdir, 'escaped.txt')
        self.assertFalse(os.path.exists(escaped))

    def test_rejects_unsupported_archive(self):
        self.assertRaises(TypeError, safe_extract, object(), self.tmpdir)


if __name__ == "__main__":
    unittest.main()
