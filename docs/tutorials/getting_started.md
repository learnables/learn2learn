# Getting Started

learn2learn is a meta-learning library providing three levels of functionality for users.
At a high level, there are many examples using meta-learning algorithms to train
on a myriad of datasets/environments. At a mid level, it provides a functional
interface for several popular meta-learning algorithms as well as a data loader
to make it easier to import other data sets. At a low level, it provides extended
functionality for modules.

## Installing

A pip package is available, updated periodically. Use the command:

```pip install -U learn2learn```

For the most update-to-date version clone the [repository](https://github.com/learnables/learn2learn) and use:

```pip install -e .```

When installing from sources, make sure that Cython is installed: `pip install cython`.

!!! info
    While learn2learn is actively used in current research projects, it is still in development.
    Breaking changes might occur.

## Development

To simplify the development process, the following commands can be executed from the cloned sources:

* `make build` - Builds learn2learn in place.
* `make clean` - Cleans previous installation.
* `make lint` - Runs linting on the codebase.
* `make lint-examples` - Runs linting on the examples.
* `make tests` - Runs a light testing suite. (i.e. the Travis one)
* `make alltests` - Runs an extensive testing suite. (much longer)
* `make docs` - Builds the documentation and serves the website locally.

!!! tip
    If you encounter a problem, feel free to an open an [issue](https://github.com/learnables/learn2learn/issues) 
    and we'll look into it.
