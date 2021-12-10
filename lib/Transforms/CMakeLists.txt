
add_mlir_library(IREELinalgTensorSandboxTransforms
  ConvertToAsyncDialect.cpp
  ConvertToGPUDialect.cpp
  FuseFillIntoReduction.cpp
  LinalgTileAndFuse.cpp
  StagedPatternRewriteDriver.cpp
  VectorDistribution.cpp

  LINK_LIBS PRIVATE
  MLIRAsync
  MLIRGPUOps
  MLIRLinalg
  MLIRLinalgTransforms

  DEPENDS
  RunnersPassIncGen
  MLIRLinalg  # TODO: Why needed here?
  MLIRLinalgTransforms  # TODO: Why needed here?
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtPassIncGen
)
