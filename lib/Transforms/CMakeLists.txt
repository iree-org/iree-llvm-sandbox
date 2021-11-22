
add_mlir_library(IREELinalgTensorSandboxTransforms
  ConvertToAsyncDialect.cpp
  ConvertToGPUDialect.cpp
  FuseFillIntoReduction.cpp
  LinalgTileAndFuse.cpp
  VectorDistribution.cpp

  LINK_LIBS PRIVATE
  MLIRAsync
  MLIRGPUOps
  MLIRLinalg
  MLIRLinalgTransforms

  DEPENDS
  RunnersPassIncGen
  MLIRLinalg
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtPassIncGen
)
