
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* A CHANGELOG.md file.
* New vision datasets: FC100, tiered-Imagenet, FGVCAircraft, VGGFlowers102.

### Changed

* Re-wrote TaskDataset and task transforms in Cython, for a 20x speed-up.
* Travis testing with different versions of Python (3.6, 3.7), torch (1.1, 1.2, 1.3, 1.4), and torchvision (0.3, 0.4, 0.5).
* New Material doc theme with links to changelog and examples.

### Fixed

* Support for `RandomClassRotation` with newer versions of torchvision.
* Various minor fixes in the examples.
