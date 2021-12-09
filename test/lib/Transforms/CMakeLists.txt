add_mlir_library(MLIRTransformsExtTestPasses
  TestStagedRewriteDriver.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTestDialect
  IREELinalgTensorSandboxTransforms
  )
