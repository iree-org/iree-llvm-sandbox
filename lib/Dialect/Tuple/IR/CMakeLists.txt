add_mlir_dialect_library(MLIRTupleDialect
  Tuple.cpp

  LINK_LIBS PUBLIC
  MLIRIR

  DEPENDS
  MLIRTupleOpsIncGen
)
