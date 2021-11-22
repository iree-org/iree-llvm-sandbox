add_mlir_library(MLIRLinalgExtTransforms
  LowerToSCF.cpp
  Tiling.cpp

  DEPENDS
  MLIRLinalgExt

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCF
)
