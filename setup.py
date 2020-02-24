#!/usr/bin/env python3

import re

from distutils.core import setup
from setuptools import (
    setup as install,
    find_packages,
    Extension
)
from Cython.Build import cythonize

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
include_dirs = []
compiler_directives = {"language_level": 3, "embedsignature": True}
extensions = [
    Extension(name='learn2learn.data',
              sources=['learn2learn/data/*.pyx']), 
]
setup(
      name='learn2learn',
      ext_modules=cythonize(extensions),
      include_dirs=include_dirs,
)

# Installs the package
install(
    name='learn2learn',
    packages=find_packages(),
    version=VERSION,
    description='PyTorch Meta-Learning Framework for Researchers',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Debajyoti Datta, Ian bunner, Seb Arnold, Praateek Mahajan',
    author_email='smr.arnold@gmail.com, praateekm@gmail.com',
    url='https://github.com/learnables/learn2learn',
    download_url='https://github.com/learnables/learn2learn/archive/' + str(VERSION) + '.zip',
    license='MIT',
    classifiers=[],
    scripts=[],
    install_requires=[
        'numpy>=1.15.4',
        'gym>=0.14.0',
        'torch>=1.1.0',
        'torchvision>=0.3.0',
        'pandas',
        'requests',
        'Cython>=0.28.5',
    ],
)
