add_llvm_library(MLIRLinalgTensorSandbox
  ConvertToAsyncDialect.cpp
  ConvertToGPUDialect.cpp
  LinalgComprehensiveBufferizePass.cpp
  LinalgDistributePass.cpp
  LinalgTensorCodegenStrategy.cpp
  LinalgTileAndFuse.cpp
  LinalgTiledLoopToSCFPass.cpp
  Registration.cpp
)

target_link_libraries(
  MLIRLinalgTensorSandbox

  MLIRAsync
  MLIRGPU
  MLIRLinalgTransforms
)

set(LINALG_TENSOR_SANDBOX_LIB "libMLIRLinalgTensorSandbox.so" PARENT_SCOPE)
