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

# Make headers from test pass available here.
target_include_directories(MLIRTupleTransforms
  PRIVATE
  ${ITERATORS_MAIN_SRC_DIR}/test/lib/Conversion/OneToNTypeConversion/
)
