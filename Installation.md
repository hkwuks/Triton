# Installation

Triton is not support Windows right now. Please install it on Linux.

## Binary Distributions

You can install the latest stable release of Trition from pip:

`pip install triton`

Binary wheels are available for CPython 3.9-3.13.

## From Source

You can install the Python package from source by running the following commands:

```sh
git clone https://github.com/triton-lang/triton.git;
cd triton/python;
pip install ninja cmake wheel; # build-time dependencies
pip install -e .
```

Note that, if llvm is not present on your system, the setup.py script will download the official LLVM static libraries and link against that.

You can then test your installation by running the unit tests:
```sh
pip install -e '.[tests]'
pytest -vs test/unit/
```

and the benchmarks

```sh
cd bench
python -m run --with-plits --result-dir /tmp/triton-bench
```