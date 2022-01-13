add_mlir_library(MLIRVectorExtTransform
  VectorMaskingUtils.cpp
  VisitorUtils.cpp

  DEPENDS
  mlir-headers
  MLIRVectorExt

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRIR
  MLIRLinalg
  MLIRVector
  MLIRVectorExt
)
