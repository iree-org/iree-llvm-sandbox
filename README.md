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

The following environment variables must be set:

*   `IREE_LLVM_SANDBOX_SOURCE_DIR`: path to the root of the sandbox source directory
*   `IREE_LLVM_SANDBOX_BUILD_DIR`: path to the directory where all binaries and libraries are built (under the `bin` and `lib` subdirectories respectively)

In these instructions, we chose:

*   `IREE_LLVM_SANDBOX_SOURCE_DIR`: $HOME/src/iree-llvm-sandbox
*   `IREE_LLVM_SANDBOX_BUILD_DIR`: $HOME/src/iree-llvm-sandbox/build

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

You **must** checkout the [IREE](https://github.com/google/iree) GitHub repo next 
to this directory and initialize submodules:

```
(cd .. && git clone https://github.com/google/iree --recurse-submodules=third_party/llvm-project)
```

And configure/build the project:

```
python configure.py --use-iree=True
```

Note that the `third_party/llvm-project` bundled with IREE will be used.

### Building without IREE.

You **must** checkout [llvm-project](https://github.com/llvm/llvm-project) at a
compatible commit next to this directory.

```
(cd .. && git clone https://github.com/llvm/llvm-project.git)
```

And configure/build the project:

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
python -m python.matmul.test
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
cd ${IREE_LLVM_SANDBOX_SOURCE_DIR}
alias search_cli="python -m python.local_search.search_cli"
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
alias rank_cli="python -m python.local_search.rank_cli"
rank_cli
```

You can customize the `--op`, the number of the output results (`--limit`) and
the metric used for ranking (`--by`) through additional command-line flags.

The metrics are coming from either `runtime` or `mca` input files that can be
specified using `--input` flag. By default results are ranked by the measured
runtime.
