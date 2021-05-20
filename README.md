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

# Prerequisites

Export some useful environment variables (add them to your ~/.bashrc) and
`mkdir` the directories:

```
export LLVM_SOURCE_DIR=${HOME}/github/llvm-project && \
export LLVM_BUILD_DIR=${HOME}/github/builds/llvm && \
export LLVM_INSTALL_DIR=${HOME}/github/install/ && \
export IREE_LLVM_SANDBOX_SOURCE_DIR=${HOME}/github/iree_llvm_sandbox && \
export IREE_LLVM_SANDBOX_BUILD_DIR=${HOME}/github/builds/iree_llvm_sandbox && \
export NPCOMP_SOURCE_DIR=${HOME}/github/mlir-npcomp && \
export NPCOMP_BUILD_DIR=${HOME}/github/builds/npcomp
```

# Python prerequisites (if using Python)

Follow the instructions for
[MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/):

```
which python
python -m venv ~/.venv/mlirdev
source ~/.venv/mlirdev/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ${LLVM_SOURCE_DIR}/mlir/lib/Bindings/Python/requirements.txt
```

Optionally, install pytorch nightly:

```
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

# Build instructions

Get LLVM, for instance:

```
git clone git@github.com:llvm/llvm-project.git ${LLVM_SOURCE_DIR}
```

Build and install LLVM + MLIR with python bindings (also see the
[mlir getting started doc](https://mlir.llvm.org/getting_started/)):

```
(cd ${LLVM_SOURCE_DIR} && \
\
cmake -G Ninja llvm \
-Dpybind11_DIR=${HOME}/.venv/mlirdev/lib/python3.9/site-packages/pybind11/share/cmake/pybind11/ \
-DLLVM_ENABLE_PROJECTS="mlir" \
-DBUILD_SHARED_LIBS=ON \
-DLLVM_BUILD_LLVM_DYLIB=ON \
-DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="X86" \
-DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
-DCMAKE_BUILD_TYPE=Release \
-DMLIR_BINDINGS_PYTHON_ENABLED=ON \
-DPython3_EXECUTABLE=$(which python) \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR} \
-DLLVM_INCLUDE_UTILS=ON \
-DLLVM_INSTALL_UTILS=ON \
-B ${LLVM_BUILD_DIR} && \
\
cmake --build ${LLVM_BUILD_DIR} --target check-mlir; \
\
cmake --build ${LLVM_BUILD_DIR} --target install)
```

Verify the MLIR cmake has been properly installed:

```
find ${LLVM_INSTALL_DIR} -name MLIRConfig.cmake
```

This should print: `${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake`

Get iree-llvm-sandbox:

```
git clone git@github.com:google/iree-llvm-sandbox.git ${IREE_LLVM_SANDBOX_SOURCE_DIR}
```

Build iree-llvm-sandbox:

```
(cd ${IREE_LLVM_SANDBOX_SOURCE_DIR} && \
\
cmake -GNinja \
-DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir \
-DCMAKE_BUILD_TYPE=Debug \
-B ${IREE_LLVM_SANDBOX_BUILD_DIR} && \
\
cmake --build ${IREE_LLVM_SANDBOX_BUILD_DIR} --target all)
```

Run a simple sanity check:

```
LD_LIBRARY_PATH=${IREE_LLVM_SANDBOX_BUILD_DIR}/runners/lib \
${IREE_LLVM_SANDBOX_BUILD_DIR}/runners/mlir-proto-opt \
${IREE_LLVM_SANDBOX_SOURCE_DIR}/runners/test/test_constant.mlir \
-linalg-comprehensive-bufferize-inplace
```

# Test and run python

Set up you PYTHONPATH properly:

```
export PYTHONPATH=${PYTHONPATH}:$LLVM_INSTALL_DIR/python:${IREE_LLVM_SANDBOX_BUILD_DIR}:${IREE_LLVM_SANDBOX_BUILD_DIR}/runners/lib; \
export PYTHONPATH=${PYTHONPATH}:${NPCOMP_BUILD_DIR}:${NPCOMP_BUILD_DIR}/lib:${NPCOMP_BUILD_DIR}/python
```

Run a simple python sanity check:

```
python ${IREE_LLVM_SANDBOX_SOURCE_DIR}/runners/test/python/linalg_matmul.py
```

Optionally, get npcomp-mlir:

```
git clone git@github.com:llvm/mlir-npcomp.git ${NPCOMP_SOURCE_DIR}
```

Optionally build npcomp-mlir:

```
(cd ${NPCOMP_SOURCE_DIR} && \
\
cmake -GNinja \
-Dpybind11_DIR=${HOME}/.venv/mlirdev/lib/python3.9/site-packages/pybind11/share/cmake/pybind11/ \
-DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir \
-DCMAKE_BUILD_TYPE=Debug \
-DPYTHON_EXECUTABLE=$(which python) \
-DPython3_EXECUTABLE=$(which python) \
-DCMAKE_BUILD_TYPE=Debug \
-DNPCOMP_USE_SPLIT_DWARF=ON \
-DCMAKE_CXX_FLAGS_DEBUG=$DEBUG_FLAGS \
-DLLVM_ENABLE_WARNINGS=ON \
-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
-B ${NPCOMP_BUILD_DIR} && \
\
cmake --build ${NPCOMP_BUILD_DIR} --target all)
```

TODOs:

1.  hook up a lit test target.

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
search_cli=${IREE_LLVM_SANDBOX_SOURCE_DIR}/runners/test/python/search_cli.py
python3 $search_cli
```

To run with a different linalg op, use `--op` flag:

```
python3 $search_cli --op matvec
```

To specify the name of the expert compiler, use `--expert` (see `experts.py` for
all available expert definitions):

```
python3 $search_cli --expert ExpertCompiler1
```

To specify the possible types, use `--types` flag:

```
python3 $search_cli --types f32,f64
```

Alternatively, one can also force some variables to concrete values, while
others will ramain random using `--assign`:

```
python3 $search_cli --assign M=16 N=32 K=64
```

To specify range of possible values for dimensions, use `--range` flag (where
numbers correspond to arguments of the corresponding `range` function in
Python):

```
python3 $search_cli --range 128,256,8
```

The search can be run using multiple processes at once, via `--par` flag:

```
python3 $search_cli --par 72
```

Each process collects the fixed number of random samples, customized via
`--samples` flag:

```
python3 $search_cli --samples 100
```

## Showing ranked results

One can see a ranked list, based on llvm-mca performance estimates:

```
rank_cli=${IREE_LLVM_SANDBOX_SOURCE_DIR}/runners/test/python/rank_mca_cli.py
python3 $rank_cli
```

You can customize the `--op`, the number of the output results (`--limit`) and
the metric used for ranking (`--by`) through additional command-line flags.
