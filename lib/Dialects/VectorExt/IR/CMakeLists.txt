add_mlir_library(MLIRVectorExt
  VectorExtDialect.cpp
  VectorExtOps.cpp

  DEPENDS
  mlir-headers
  MLIRVectorExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRIR
)
