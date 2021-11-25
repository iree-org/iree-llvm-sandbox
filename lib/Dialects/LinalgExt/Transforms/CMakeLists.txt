add_mlir_library(MLIRLinalgExtTransforms
  TileToInParallel.cpp
  LowerToSCF.cpp
  Tiling.cpp

  DEPENDS
  MLIRLinalgExt

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRSCFToStandard
  MLIRLinalgToLLVM
  MLIRVectorToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRIR
  MLIRMath
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRLinalgExt
  MLIRPass
  MLIRSCF
  MLIRTransforms
)
