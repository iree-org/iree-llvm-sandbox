add_mlir_dialect_library(MLIRIteratorsTransforms
  DecomposeIteratorStates.cpp

  DEPENDS
  MLIRIteratorsPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRIterators
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransformUtils
)
