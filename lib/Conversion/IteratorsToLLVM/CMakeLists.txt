add_mlir_conversion_library(MLIRIteratorsToLLVM
  IteratorsToLLVM.cpp
  IteratorAnalysis.cpp

  DEPENDS
  MLIRStructuredConversionIncGen

  LINK_LIBS PUBLIC
  IteratorsUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIterators
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTabularToLLVM
  MLIRTupleDialect
  MLIRTransforms
  MLIRTupleDialect
)
