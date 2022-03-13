
add_mlir_library(IREESandboxTransforms
  FuseFillIntoReduction.cpp
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
  IREELinalgExtIncGen
  IREELinalgExtInterfacesIncGen
)
