include(AddMLIR)

add_mlir_library(IREELinalgTensorSandbox
  ConvertToAsyncDialect.cpp
  ConvertToGPUDialect.cpp
  LinalgTensorCodegenStrategy.cpp
  LinalgTileAndFuse.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  MLIRAsync
  MLIRGPUOps
  MLIRLinalgTransforms
)

add_mlir_public_c_api_library(IREELinalgTensorSandboxCAPI
  CAPI.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  IREELinalgTensorSandbox
)
