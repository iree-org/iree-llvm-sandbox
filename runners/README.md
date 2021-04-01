# Sandbox for Linalg on Tensors

The purpose of this directory is to expose a minimal end-to-end path for
targetable codegen with linalg on tensors. This acts as a staging area for
experimenting with new concepts until they mature and graduate to core.

# Build instructions

Get LLVM, compile and install it (also see the
[mlir getting started doc](https://mlir.llvm.org/getting_started/)):

```
git clone git@github.com:llvm/llvm-project.git && cd llvm-project && mkdir build && \
\
cmake -G Ninja llvm -DLLVM_ENABLE_PROJECTS="mlir" -DBUILD_SHARED_LIBS=ON -DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="X86" -DMLIR_INCLUDE_INTEGRATION_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
-DMLIR_BINDINGS_PYTHON_ENABLED=ON -DPython3_EXECUTABLE=/usr/bin/python3  -DLLVM_ENABLE_ASSERTIONS=ON \
-DCMAKE_INSTALL_PREFIX=<llvm_install_path> -B build && \
\
cmake --build build --target check-mlir && cmake --build build --target install
```

Verify the MLIR cmake has been properly installed:

```
find <llvm_install_path> -name MLIRConfig.cmake
```

This should print: `<llvm_install_path>/lib/cmake/mlir/MLIRConfig.cmake`

Get iree-llvm-sandbox and compile it; e.g.:

```
git clone git@github.com:google/iree-llvm-sandbox.git && \
\
cmake -GNinja -DMLIR_DIR=<llvm_install_path>/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Debug -B build && \
\
cmake --build build --target all
```

TODO: hook up a lit test target.

TODO: hook up python support and a python example based on the example in this
[commit](https://reviews.llvm.org/D99430) which can call the
[LinalgTensorCodegenStrategy](https://github.com/google/iree-llvm-sandbox/blob/main/runners/LinalgTensorCodegenStrategy.cpp)
from python.
