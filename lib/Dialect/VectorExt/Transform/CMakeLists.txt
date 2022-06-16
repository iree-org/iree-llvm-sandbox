add_mlir_library(MLIRVectorExtTransforms
  VectorMaskingUtils.cpp

  DEPENDS
  mlir-headers
  MLIRVectorExt

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRGPUOps
  MLIRIR
  MLIRLinalgDialect
  MLIRVectorDialect
  MLIRVectorExt
  MLIRVectorTransforms
  MLIRVectorUtils
)
