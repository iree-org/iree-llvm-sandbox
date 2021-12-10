add_mlir_library(MLIRVectorExtTestPasses
  TestVectorMaskingUtils.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRVector
  MLIRVectorExt
  MLIRVectorExtTransform
  )
