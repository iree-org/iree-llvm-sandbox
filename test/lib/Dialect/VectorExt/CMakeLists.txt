add_mlir_library(MLIRVectorExtTestPasses
  TestVectorMaskingUtils.cpp
  TestVectorWarp.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRVector
  MLIRVectorExt
  MLIRVectorExtTransform
  )
