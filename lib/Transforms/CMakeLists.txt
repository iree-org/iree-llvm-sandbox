
add_mlir_library(IREELinalgTensorSandboxTransforms
  ConvertToAsyncDialect.cpp
  ConvertToGPUDialect.cpp
  FuseFillIntoReduction.cpp
  LinalgTileAndFuse.cpp
  Listener.cpp
  ListenerCSE.cpp
  ListenerGreedyPatternRewriteDriver.cpp
  StagedPatternRewriteDriver.cpp
  VectorDistribution.cpp

  LINK_LIBS PRIVATE
  MLIRAsync
  MLIRGPUOps
  MLIRLinalg
  MLIRLinalgTransforms

  DEPENDS
  mlir-headers
  DriverPassIncGen
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtPassIncGen
)
