add_mlir_conversion_library(MLIRStatesToLLVM
  StatesToLLVM.cpp

  DEPENDS
  MLIRIteratorsConversionIncGen

  LINK_LIBS PUBLIC
  IteratorsUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIterators
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFTransforms
)
