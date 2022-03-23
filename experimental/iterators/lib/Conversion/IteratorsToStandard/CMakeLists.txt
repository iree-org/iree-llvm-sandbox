add_mlir_conversion_library(MLIRIteratorsToStandard
  IteratorsToStandard.cpp

  DEPENDS
  MLIRIteratorsConversionIncGen

  LINK_LIBS PUBLIC
  MLIRFunc
  MLIRIterators
  MLIRLLVMIR
  MLIRMemRef
  MLIRPass
  MLIRTransforms
  )
