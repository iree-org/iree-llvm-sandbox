add_mlir_library(MLIRVectorExt
  VectorExtDialect.cpp
  VectorExtOps.cpp

  DEPENDS
  MLIRVectorExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRIR
)
