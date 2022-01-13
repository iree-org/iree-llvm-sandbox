add_mlir_library(MLIRLinalgTransformTestPasses
  TestScopedTransform.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRLinalgTransformTransforms
  )
