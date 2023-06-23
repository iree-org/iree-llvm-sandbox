add_mlir_conversion_library(MLIRTritonPtrToLLVM
  TritonPtrToLLVM.cpp

  DEPENDS
  MLIRStructuredConversionIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFTransforms
  MLIRTransformUtils
  TritonIR
)
