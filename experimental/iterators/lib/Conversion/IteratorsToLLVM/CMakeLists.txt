add_mlir_conversion_library(MLIRIteratorsToLLVM
  IteratorsToLLVM.cpp
  IteratorAnalysis.cpp

  DEPENDS
  MLIRIteratorsConversionIncGen

  LINK_LIBS PUBLIC
  MLIRFunc
  MLIRIterators
  MLIRLLVMIR
  MLIRPass
  MLIRSCF
  MLIRTransforms
  )
