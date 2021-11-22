
add_mlir_library(IREELinalgTensorSandboxTransforms
  ConvertToAsyncDialect.cpp
  ConvertToGPUDialect.cpp
  FuseFillIntoReduction.cpp
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
