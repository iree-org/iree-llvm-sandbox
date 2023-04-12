add_mlir_library(MLIRTupleDialect
  Tuple.cpp

  LINK_LIBS PUBLIC
  MLIRIR

  DEPENDS
  MLIRTupleOpsIncGen
)
