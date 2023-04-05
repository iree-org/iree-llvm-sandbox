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

# Make headers from test pass available here.
target_include_directories(MLIRIteratorsTransforms
  PRIVATE
  ${ITERATORS_MAIN_SRC_DIR}/test/lib/Conversion/OneToNTypeConversion/
)
