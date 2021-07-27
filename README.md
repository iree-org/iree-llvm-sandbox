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
ways to set this up but the following is recommended. It is left to the reader
to adapt paths if deviating. We assume below that projects are checked out to
`$HOME/src`.

Patches required in LLVM:

*   https://reviews.llvm.org/D106520

## Check out projects

In your `$HOME/src` directory, check out each project:

Required:

*   `git clone https://github.com/llvm/llvm-project.git`
*   `git clone https://github.com/google/iree-llvm-sandbox`

Optional: * `git clone https://github.com/llvm/mlir-npcomp.git`

We use the following environment variables in these instructions:

*   `IREE_LLVM_SANDBOX_SOURCE_DIR`: $HOME/src/iree-llvm-sandbox
*   `IREE_LLVM_SANDBOX_BUILD_DIR`: $HOME/src/sandbox_build

## Python prerequisites (if using Python)

Follow the instructions for
[MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/):

```
which python
python -m venv ~/.venv/mlirdev
source ~/.venv/mlirdev/bin/activate
python -m pip install --upgrade pip
python -m pip install -r $HOME/src/llvm-project/mlir/python/requirements.txt
```

Optionally, install pytorch nightly:

```
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

## CMake configure and build

The following assumes that you will be building into `$HOME/src/sandbox_build`:

```
cd $HOME/src
cmake -GNinja -Bsandbox_build llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=iree_llvm_sandbox \
  -DLLVM_EXTERNAL_IREE_LLVM_SANDBOX_SOURCE_DIR=$PWD/iree-llvm-sandbox \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python) \
  -DCMAKE_BUILD_TYPE=Release
```

The following CMake options are recommended for efficiency, if you have the
corresponding tools installed:

*   `-DLLVM_ENABLE_LLD=ON`
*   `-DLLVM_CCACHE_BUILD=ON`

Useful build commands (from the `sandbox_build` directory):

*   `ninja check-mlir`: Run MLIR tests.
*   `ninja all`: Build everything.
*   `ninja IREELLVMSandboxPythonModules`: Just build the python packages
    (smaller/faster than everything).

## Using the Python API

```
cd "${IREE_LLVM_SANDBOX_BUILD_DIR}"
export PYTHONPATH=$PWD/tools/iree_llvm_sandbox/python_package

# Sanity check (should not error).
python -c "import mlir.iree_sandbox"

# Run a matmul.
python ../iree-llvm-sandbox/runners/test/python/linalg_matmul.py
```

## Using mlir-proto-opt

```
cd "${IREE_LLVM_SANDBOX_BUILD_DIR}"

./bin/mlir-proto-opt \
  ../iree-llvm-sandbox/runners/test/test_constant.mlir \
  -linalg-comprehensive-module-bufferize
```

TODOs:

1.  hook up a lit test target.
2.  re-add npcomp instructions once it is upgraded to use the same build setup.

# Python-driven parameter search

Python tests come with a tool to perform as simple randomized search. The search
is going to randomly instantiate a given op to some cocnrete dimensions and type
variables and try to compile it using mlir.

The results are persisted in the `output/` folder by default in a structure that
includes a name of the expert compiler, the name of the op and the
success/failure/timeout status code. The results contain the full program output
(including potential compilation errors) and an accompanying `.sh` file that can
be used to re-run the same configuration again.

## Collecting random measurements

To run the search with default settings:

```
export PATH="${IREE_LLVM_SANDBOX_BUILD_DIR}/bin:$PATH"
alias search_cli="python ${IREE_LLVM_SANDBOX_SOURCE_DIR}/runners/test/python/search_cli.py"
search_cli
```

To run with a different linalg op, use `--op` flag:

```
search_cli --op matvec
```

To specify the name of the expert compilers, use `--expert` (see `experts.py`
for all available expert definitions):

```
search_cli --experts ExpertCompiler1
```

To specify the possible types, use `--types` flag:

```
search_cli --types f32,f64
```

Alternatively, one can also force some variables to concrete values, while
others will ramain random using `--assign`:

```
search_cli --assign M=16 N=32 K=64
```

To specify range of possible values for dimensions, use `--range` flag (where
numbers correspond to arguments of the corresponding `range` function in
Python):

```
search_cli --range 128,256,8
```

The search can be run using multiple processes at once, via `--par` flag:

```
search_cli --par 72
```

Each process collects the fixed number of random samples, customized via
`--samples` flag:

```
search_cli --samples 100
```

## Showing ranked results

One can see a ranked list, based on llvm-mca performance estimates:

```
alias rank_cli="${IREE_LLVM_SANDBOX_SOURCE_DIR}/runners/test/python/rank_mca_cli.py"
rank_cli
```

You can customize the `--op`, the number of the output results (`--limit`) and
the metric used for ranking (`--by`) through additional command-line flags.

The metrics are coming from either `runtime` or `mca` input files that can be
specified using `--input` flag. By default results are ranked by the measured
runtime.
