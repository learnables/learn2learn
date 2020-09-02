
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* `l2l.vision.datasets.DescribableTextures`
* `l2l.vision.datasets.Quickdraw`

### Changed

* Updated reference for citations.

### Fixed


## v0.1.3

### Added

* `l2l.vision.datasets.CUBirds200`.

### Changed

* Optimization transforms can be accessed directly through `l2l.optim`, e.g. `l2l.optim.KroneckerTransform`.
* All vision models adhere to the `.features` and `.classifier` interface.

### Fixed

* Fix `clone_module` for Modules whose submodules share parameters.


## v0.1.2

### Added

* New example: [Meta-World](https://github.com/rlworkgroup/metaworld) example with MAML-TRPO with it's own env wrapper. (@[Kostis-S-Z](https://github.com/Kostis-S-Z))
* `l2l.vision.benchmarks` interface.
* Differentiable optimization utilities in `l2l.optim`. (including `l2l.optim.LearnableOptimizer` for meta-descent)
* General gradient-based meta-learning wrapper in `l2l.algorithms.GBML`.
* Various `nn.Modules` in `l2l.nn`.
* `l2l.update_module` as a more general alternative to `l2l.algorithms.maml_update`.

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
