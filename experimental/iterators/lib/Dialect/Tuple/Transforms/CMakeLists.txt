add_mlir_dialect_library(MLIRTupleTransforms
  DecomposeTuples.cpp

  DEPENDS
  MLIRTuplePassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
