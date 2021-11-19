add_subdirectory(LinalgExt)

add_mlir_library(IREELinalgTensorSandbox
  ConvertToAsyncDialect.cpp
  ConvertToGPUDialect.cpp
  FuseFillIntoReduction.cpp
  LinalgTensorCodegenDriver.cpp
  LinalgTileAndFuse.cpp
  VectorDistribution.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  MLIRAsync
  MLIRGPUOps
  MLIRLinalgTransforms

  DEPENDS
  RunnersPassIncGen
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtPassIncGen
)

add_mlir_public_c_api_library(IREELinalgTensorSandboxCAPI
  CAPI.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  IREELinalgTensorSandbox
)
