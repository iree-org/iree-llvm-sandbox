add_mlir_dialect_library(MLIRIteratorsTransforms
  DecomposeIteratorStates.cpp

  DEPENDS
  MLIRIteratorsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
