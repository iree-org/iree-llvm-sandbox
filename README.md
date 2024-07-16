# IREE LLVM Sandbox

DISCLAIMER: This is not an officially-supported Google project. It is a sandbox
for quick iteration and experimentation on projects related to the IREE project,
MLIR, and LLVM.

This repository contains experimental work by the IREE team closely related to
LLVM and MLIR, usually with the aim of upstreaming in some form. The main
project is at <https://github.com/google/iree>.

As an experimental project, build greenness, documentation, and polish are
likely to be minimal, as it instead prioritizes easy experimentation.

## License

Licensed under the Apache license with LLVM Exceptions. See [LICENSE](LICENSE)
for more information.

## Subprojects

The repository currently houses the following projects:

* The [Iterators](README-Iterators.md) dialect: database-style iterators for
  expressing computations on streams of data.
* The [Substrait](README-Substrait.md) dialect: an input/output dialect for
  [Substrait](https://substrait.io/), the cross-language serialization format
  of database query plans.
* The [Tuple](include/structured/Dialect/Tuple/): ops for manipulation of
  built-in tuples (used by the Iterators dialect).

## Build Instructions

This project builds as part of the LLVM External Projects facility (see
documentation for the `LLVM_EXTERNAL_PROJECTS` config setting).

It is left to the reader to adapt paths if deviating. We assume below that
projects are checked out to `$HOME/src`.

### Check out Project

In your `$HOME/src` directory, clone this project recursively:

```bash
git clone --recursive https://github.com/google/iree-llvm-sandbox
```

If you have cloned non-recursively already and every time a submodule is
updated, run the following command inside the cloned repository instead:

```bash
git submodule update --recursive --init
```

Define the following environment variables (adapted to your situation), ideally
making them permanent in your `$HOME/.bashrc` or in the `activate` script of
your Python virtual environment (see below):

```bash
export IREE_LLVM_SANDBOX_SOURCE_DIR=$HOME/src/iree-llvm-sandbox
export IREE_LLVM_SANDBOX_BUILD_DIR=${IREE_LLVM_SANDBOX_SOURCE_DIR}/build
```

### Python prerequisites

Create a virtual environment, activate it, and install the dependencies from
[`requirements.txt`](requirements.txt):

```bash
python -m venv ~/.venv/mlirdev
source ~/.venv/mlirdev/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ${IREE_LLVM_SANDBOX_SOURCE_DIR}/requirements.txt
```

For details, see the documentation of the
[MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/).

### Configure and build main project

Run the command below to set up the build system, possibly adapting it to your
needs. For example, you may choose not to compile `clang`, `clang-tools-extra`,
`lld`, and/or the examples to save compilation time, or use a different variant
than `Debug`.

```bash
cmake \
  -DPython3_EXECUTABLE=$(which python) \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_EXTERNAL_PROJECTS=structured \
  -DLLVM_EXTERNAL_STRUCTURED_SOURCE_DIR=${IREE_LLVM_SANDBOX_SOURCE_DIR} \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_ENABLE_PYTHON_BENCHMARKS=ON \
  -S${IREE_LLVM_SANDBOX_SOURCE_DIR}/third_party/llvm-project/llvm \
  -B${IREE_LLVM_SANDBOX_BUILD_DIR} \
  -G Ninja
```

To build, run:

```bash
cd ${IREE_LLVM_SANDBOX_BUILD_DIR} && ninja
```

## Using structured-opt

```bash
"${IREE_LLVM_SANDBOX_BUILD_DIR}"/bin/structured-opt --help
```

## Running tests

You can run all tests with the following command:

```bash
cd ${IREE_LLVM_SANDBOX_BUILD_DIR} && ninja
```

You may also use `lit` to run a subset of the tests. You may

```bash
lit -v ${IREE_LLVM_SANDBOX_BUILD_DIR}/test
lit -v ${IREE_LLVM_SANDBOX_BUILD_DIR}/test/Integration
lit -v ${IREE_LLVM_SANDBOX_BUILD_DIR}/test/Dialect/Iterators/map.mlir
```

## Diagnostics via LSP servers

The [MLIR LSP Servers](https://mlir.llvm.org/docs/Tools/MLIRLSP/) allows editors
to display as-you-type diagnostics, code navigation, and similar features. In
order to extend this functionality to the dialects from this repository, use
the following LSP server binaries:

```bash
${IREE_LLVM_SANDBOX_BUILD_DIR}/bin/mlir-proto-lsp-server
${IREE_LLVM_SANDBOX_BUILD_DIR}/bin/tblgen-lsp-server",
${IREE_LLVM_SANDBOX_BUILD_DIR}/bin/mlir-pdll-lsp-server
```

In VS Code, this is done via the `mlir.server_path`, `mlir.pdll_server_path`,
and `mlir.tablegen_server_path` properties in `settings.json`.

## Hashes of Interest

Repro for experimental results described in the
[arxiv paper](https://arxiv.org/abs/2202.03293):

```bash
git checkout 680c8160edb7aa13b621b28c221288624ebc37e4
echo Please update LLVM to $(cat pinned-llvm-version)
```

Hash before transitioning to schedule dialect only:

```bash
git checkout ea0e5ec37a4d73808e16926c0335cc21fde0286c
echo Please update LLVM to $(cat pinned-llvm-version)
```
