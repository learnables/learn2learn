
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

* clone_module supports non-Module objects.
* VGG flowers now relies on tarfile.open() instead of tarfile.TarFile().


## v0.1.1

### Added

* New tutorial: 'Feature Reuse with ANIL'. (@ewinapun)

### Changed

* Mujoco imports optional for docs: the import error is postponed to first method call.

### Fixed

* `MAML()` and `clone_module` support for RNN modules.


## v0.1.0.1

### Fixed

* Remove Cython dependency when installing from PyPI and clean up package distribution.


## v0.1.0

### Added

* A CHANGELOG.md file.
* New vision datasets: FC100, tiered-Imagenet, FGVCAircraft, VGGFlowers102.
* New vision examples: Reptile & ANIL.
* Extensive benchmarks of all vision examples.

### Changed

* Re-wrote TaskDataset and task transforms in Cython, for a 20x speed-up.
* Travis testing with different versions of Python (3.6, 3.7), torch (1.1, 1.2, 1.3, 1.4), and torchvision (0.3, 0.4, 0.5).
* New Material doc theme with links to changelog and examples.

### Fixed

* Support for `RandomClassRotation` with newer versions of torchvision.
* Various minor fixes in the examples.
* Add Dropbox download if GDrive fails for FC100.
