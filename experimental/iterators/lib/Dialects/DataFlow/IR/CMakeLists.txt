add_mlir_library(MLIRDataFlow
  DataFlow.cpp

  LINK_LIBS PUBLIC
  MLIRIR

  DEPENDS
  MLIRDataFlowOpsIncGen
)
