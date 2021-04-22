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

Export some useful environment variables (add them to your ~/.bashrc) and
`mkdir` the directories: `export LLVM_SOURCE_DIR=${HOME}/github/llvm-project
export LLVM_BUILD_DIR=${HOME}/github/builds/llvm export
LLVM_INSTALL_DIR=${HOME}/github/install/ export
IREE_LLVM_SANDBOX_SOURCE_DIR=${HOME}/github/iree_llvm_sandbox export
IREE_LLVM_SANDBOX_BUILD_DIR=${HOME}/github/builds/iree_llvm_sandbox export
NPCOMP_SOURCE_DIR=${HOME}/github/mlir-npcomp export
NPCOMP_BUILD_DIR=${HOME}/github/builds/npcomp`

Get LLVM, for instance:

```
git clone git@github.com:llvm/llvm-project.git ${LLVM_SOURCE_DIR}
```

Build and install LLVM + MLIR with python bindings (also see the
[mlir getting started doc](https://mlir.llvm.org/getting_started/)): `(cd
${LLVM_SOURCE_DIR} && \ \ cmake -G Ninja llvm
-Dpybind11_DIR=${HOME}/.venv/mlirdev/lib/python3.9/site-packages/pybind11/share/cmake/pybind11/
\ -DLLVM_ENABLE_PROJECTS="mlir" -DBUILD_SHARED_LIBS=ON -DLLVM_BUILD_EXAMPLES=ON
\ -DLLVM_TARGETS_TO_BUILD="X86" -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
-DCMAKE_BUILD_TYPE=Release \ -DMLIR_BINDINGS_PYTHON_ENABLED=ON
-DPython3_EXECUTABLE=/usr/bin/python3 \ -DLLVM_ENABLE_ASSERTIONS=ON
-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR} -B ${LLVM_BUILD_DIR} \ cmake --build
build --target check-mlir && cmake --build build --target install)`

Verify the MLIR cmake has been properly installed:

```
find ${LLVM_INSTALL_DIR} -name MLIRConfig.cmake
```

This should print: `${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake`

Optionally, get npcomp-mlir:

```
git clone git@github.com:llvm/mlir-npcomp.git ${NPCOMP_SOURCE_DIR}
```

Optionally build npcomp-mlir: `(cd ${NPCOMP_SOURCE_DIR} && \ \ cmake -GNinja
-Dpybind11_DIR=${HOME}/.venv/mlirdev/lib/python3.9/site-packages/pybind11/share/cmake/pybind11/
\ -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Debug -B
${NPCOMP_BUILD_DIR} \ -DPython3_EXECUTABLE=/usr/bin/python3
-DCMAKE_BUILD_TYPE=Debug -DNPCOMP_USE_SPLIT_DWARF=ON \
-DCMAKE_CXX_FLAGS_DEBUG=$DEBUG_FLAGS -DLLVM_ENABLE_WARNINGS=ON
-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE && \ \ cmake --build ${NPCOMP_BUILD_DIR}
--target all)`

Get iree-llvm-sandbox:

```
git clone git@github.com:google/iree-llvm-sandbox.git ${IREE_LLVM_SANDBOX_SOURCE_DIR}
```

Build iree-llvm-sandbox: `(cd ${IREE_LLVM_SANDBOX_SOURCE_DIR} && \ \ cmake
-GNinja -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Debug
-B ${IREE_LLVM_SANDBOX_BUILD_DIR} && \ \ cmake --build
${IREE_LLVM_SANDBOX_BUILD_DIR} --target all)`

Run a simple sanity check:

```
LD_LIBRARY_PATH=${IREE_LLVM_SANDBOX_BUILD_DIR}/runners/lib \
${IREE_LLVM_SANDBOX_BUILD_DIR}/runners/mlir-proto-opt \
${IREE_LLVM_SANDBOX_SOURCE_DIR}/runners/test/test_constant.mlir \
-linalg-comprehensive-bufferize-inplace
```

# Experimental python support

```
export PYTHONPATH=$LLVM_INSTALL_DIR/python:${IREE_LLVM_SANDBOX_BUILD_DIR}:${IREE_LLVM_SANDBOX_BUILD_DIR}/runners/lib:${NPCOMP_BUILD_DIR}
```

Run a simple python sanity check: `python
${IREE_LLVM_SANDBOX_SOURCE_DIR}/runners/test/python/linalg_matmul.py`

TODOs:

1.  hook up a lit test target.
