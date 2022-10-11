add_mlir_conversion_library(MLIRTabularToLLVM
  TabularToLLVM.cpp

  DEPENDS
  MLIRIteratorsConversionIncGen

  LINK_LIBS PUBLIC
  IteratorsUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRTabular
)
