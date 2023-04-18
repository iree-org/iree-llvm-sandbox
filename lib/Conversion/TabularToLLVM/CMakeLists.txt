add_mlir_conversion_library(MLIRTabularToLLVM
  TabularToLLVM.cpp

  DEPENDS
  MLIRStructuredConversionIncGen

  LINK_LIBS PUBLIC
  IteratorsUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFTransforms
  MLIRTabular
)
