add_mlir_conversion_library(MLIRTritonToLLVM
  TritonToLLVM.cpp

  DEPENDS
  MLIRStructuredConversionIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
