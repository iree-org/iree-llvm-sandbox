add_llvm_library(MLIRLinalgTensorSandbox
  ConvertToAsyncDialect.cpp
  ConvertToGPUDialect.cpp
  LinalgTensorCodegenStrategy.cpp
  LinalgTileAndFuse.cpp
  Registration.cpp
)

target_link_libraries(
  MLIRLinalgTensorSandbox

  MLIRAsync
  MLIRGPUOps
  MLIRLinalgTransforms
)

set(LINALG_TENSOR_SANDBOX_LIB "libMLIRLinalgTensorSandbox.so" PARENT_SCOPE)
