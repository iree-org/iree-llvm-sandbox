add_mlir_conversion_library(MLIRDataFlowToIterators
  DataFlowToIterators.cpp

  DEPENDS
  MLIRIteratorsConversionIncGen

  LINK_LIBS PUBLIC
  MLIRDataFlow
  MLIRFunc
  MLIRIterators
  MLIRLLVMIR
  MLIRPass
  MLIRTransforms
  )
