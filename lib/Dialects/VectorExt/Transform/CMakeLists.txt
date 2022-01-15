set(LLVM_OPTIONAL_SOURCES
  VisitorUtils.cpp
)

add_mlir_library(MLIRVectorExtTransform
  VectorMaskingUtils.cpp
  VectorWarpUtils.cpp

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
