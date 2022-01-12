add_mlir_library(MLIRLinalgTransformOps
  LinalgTransformOps.cpp

  DEPENDS
  MLIRLinalgTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDL
  MLIRControlFlowInterfaces
)
