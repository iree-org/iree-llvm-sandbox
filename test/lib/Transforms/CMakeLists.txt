add_mlir_library(MLIRTransformsExtTestPasses
  TestListenerPasses.cpp
  TestStagedRewriteDriver.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTestDialect
  IREELinalgTensorSandboxTransforms
  )
