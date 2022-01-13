add_mlir_library(MLIRLinalgTransformOps
  LinalgTransformOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRPDL
  MLIRControlFlowInterfaces
)
