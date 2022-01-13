add_mlir_library(MLIRLinalgTransformOps
  LinalgTransformOps.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRPDL
  MLIRControlFlowInterfaces
)
