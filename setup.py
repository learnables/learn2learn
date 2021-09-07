#!/usr/bin/env python3

import sys
import re

from distutils.core import setup
from setuptools import (
    setup as install,
    find_packages,
    Extension
)

# Parses version number: https://stackoverflow.com/a/7071358
VERSIONFILE = 'learn2learn/_version.py'
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    VERSION = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

# Compile with Cython
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
# https://github.com/FedericoStra/cython-package-example/blob/master/setup.py
extension_type = '.c'
cmd_class = {}
use_cython = 'develop' in sys.argv or 'build_ext' in sys.argv
if use_cython:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    extension_type = '.pyx'
    cmd_class = {'build_ext': build_ext}

extensions = [
    Extension(name='learn2learn.data.meta_dataset',
              sources=['learn2learn/data/meta_dataset' + extension_type]), 
    Extension(name='learn2learn.data.task_dataset',
              sources=['learn2learn/data/task_dataset' + extension_type]), 
    Extension(name='learn2learn.data.transforms',
              sources=['learn2learn/data/transforms' + extension_type]), 
]

if use_cython:
    compiler_directives = {
        'language_level': 3,
        'embedsignature': True,
        #  'profile': True,
        #  'binding': True,
    }
    extensions = cythonize(extensions, compiler_directives=compiler_directives)

# Installs the package
install(
    name='learn2learn',
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass=cmd_class,
    zip_safe=False,  # as per Cython docs
    version=VERSION,
    description='PyTorch Library for Meta-Learning Research',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    author='Debajyoti Datta, Ian bunner, Seb Arnold, Praateek Mahajan',
    author_email='smr.arnold@gmail.com',
    url='https://github.com/learnables/learn2learn',
    download_url='https://github.com/learnables/learn2learn/archive/' + str(VERSION) + '.zip',
    license='MIT',
    classifiers=[],
    scripts=[],
    setup_requires=['cython>=0.28.5', ],
    install_requires=[
        'numpy>=1.15.4',
        'gym>=0.14.0',
        'torch>=1.1.0',
        'torchvision>=0.3.0',
        'scipy',
        'requests',
        'gsutil',
        'tqdm',
        'qpth>=0.0.15',
        #  'pytorch_lightning>=1.0.2',
    ],
)
