add_mlir_library(MLIRVectorExtTestPasses
  TestVectorMaskingUtils.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRRewrite
  MLIRVector
  MLIRVectorExt
  MLIRVectorExtTransforms
  )
