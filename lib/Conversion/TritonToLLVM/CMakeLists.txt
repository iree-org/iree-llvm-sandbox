add_mlir_conversion_library(MLIRTritonToLLVM
  TritonToLLVM.cpp

  DEPENDS
  MLIRStructuredConversionIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRFuncTransforms
  MLIRLinalgDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
