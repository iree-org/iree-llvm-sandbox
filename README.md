# IREE LLVM Sandbox

DISCLAIMER: This is not an officially-supported Google project. It is a sandbox
for quick iteration and experimentation on projects related to the IREE project,
MLIR, and LLVM.

This repository contains experimental work by the IREE team closely related to
LLVM and MLIR, usually with the aim of upstreaming in some form. The main
project is at https://github.com/google/iree.

As an experimental project, build greenness, documentation, and polish are
likely to be minimal, as it instead prioritizes easy experimentation.

# License

Licensed under the Apache license with LLVM Exceptions. See [LICENSE](LICENSE)
for more information.

# Build instructions

This project builds as part of the LLVM External Projects facility (see
documentation for the `LLVM_EXTERNAL_PROJECTS` config setting). There are many
ways to set this up. We recommend using our `configure.py` script below.

It is left to the reader to adapt paths if deviating. We assume below that
projects are checked out to `$HOME/src`.

## Check out projects

TODO: Simplify instructions.

In your `$HOME/src` directory, check out each project:

Required:

*   `git clone https://github.com/google/iree-llvm-sandbox`

We use the following environment variables defaults in these instructions:

*   `IREE_LLVM_SANDBOX_SOURCE_DIR`: $HOME/src/iree-llvm-sandbox
*   `IREE_LLVM_SANDBOX_BUILD_DIR`: ${IREE_LLVM_SANDBOX_SOURCE_DIR}/build

## Python prerequisites (if using Python)

Follow the instructions for
[MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/):

```
which python
python -m venv ~/.venv/mlirdev
source ~/.venv/mlirdev/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure and build.

The sandbox can be optionally built with or without IREE integration (for
accessing IREE specific IR and evaluating on IREE compatible targets):

### Building with IREE.

Checkout the [IREE](https://github.com/google/iree) GitHub repo next to this
directory and initialize submodules:

```
(cd .. && git clone https://github.com/google/iree --recurse-submodules=third_party/llvm-project)
```

And configure/build the project:

```
python configure.py --iree-path=../iree
```

Note that the `third_party/llvm-project` bundled with IREE will be used.

### Building without IREE.

You must checkout [llvm-project](https://github.com/llvm/llvm-project) at a
compatible commit.

```
(cd .. && git clone https://github.com/llvm/llvm-project.git)
```

And configure/build the project. By default the `configure.py` script will look in `${IREE_LLVM_SANDBOX_SOURCE_DIR}/../llvm-project` (this can also
be overridden with `--llvm-path`):

```
python configure.py
```

## Using the Python API

```
source .env && export PYTHONPATH

# Sanity check (should not error).
python -c "import mlir.iree_sandbox"

# Run a matmul.
export MLIR_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_runner_utils.so; \
cd ${IREE_LLVM_SANDBOX_SOURCE_DIR}; \
python -m python.examples.matmul.test
```

## Using mlir-proto-opt

```
"${IREE_LLVM_SANDBOX_BUILD_DIR}"/bin/mlir-proto-opt \
  ../iree-llvm-sandbox/test/test_constant.mlir \
  -linalg-comprehensive-module-bufferize
```

TODOs:

1.  hook up a lit test target.
2.  re-add npcomp instructions once it is upgraded to use the same build setup.


## Running tests

The following commands either run the lit tests only or all tests:
```
# Run lit tests
lit -v test
# Run python and lit tests
python ./run_tests
```
The lit configuration file `test/lit.cfg.py` contains a list of excluded tests.