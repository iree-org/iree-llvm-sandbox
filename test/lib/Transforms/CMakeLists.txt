add_mlir_library(MLIRTransformsExtTestPasses
  TestListenerPasses.cpp
  TestStagedRewriteDriver.cpp

  DEPENDS
  mlir-headers

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTestDialect
  IREELinalgTensorSandboxTransforms
  )
