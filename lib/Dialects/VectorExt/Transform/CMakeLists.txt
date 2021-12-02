add_mlir_library(MLIRVectorExtTransform
  VectorMaskingUtils.cpp

  DEPENDS
  MLIRVectorExt

  LINK_LIBS PUBLIC
  MLIRIR
)
