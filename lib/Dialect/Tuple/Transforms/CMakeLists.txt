add_mlir_dialect_library(MLIRTupleTransforms
  DecomposeTuples.cpp

  DEPENDS
  MLIRTuplePassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransformUtils
  MLIRTupleDialect
)
