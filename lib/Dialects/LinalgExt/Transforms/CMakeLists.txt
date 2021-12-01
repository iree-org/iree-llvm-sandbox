add_mlir_library(MLIRLinalgExtTransforms
  InParallelToAsync.cpp
  InParallelToSequentialFor.cpp
  Tiling.cpp
  TileToSequentialFor.cpp
  TileToInParallel.cpp

  DEPENDS
  MLIRLinalgExt

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRAsync
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
