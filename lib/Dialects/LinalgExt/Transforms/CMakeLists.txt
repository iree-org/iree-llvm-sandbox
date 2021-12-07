add_mlir_library(MLIRLinalgExtTransforms
  InParallelToAsync.cpp
  InParallelToSequentialFor.cpp
  TilingExternalModels.cpp
  TileToSequentialFor.cpp
  TileToInParallel.cpp
  Tiling.cpp
  TilingToTileOp.cpp

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
