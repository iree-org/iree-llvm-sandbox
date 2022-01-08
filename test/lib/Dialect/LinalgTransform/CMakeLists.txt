add_mlir_library(MLIRLinalgTransformTestPasses
  TestScopedTransform.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRLinalgTransformTransforms
  )
