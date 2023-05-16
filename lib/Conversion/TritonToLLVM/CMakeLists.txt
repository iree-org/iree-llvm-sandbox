add_mlir_conversion_library(MLIRTritonToLLVM
  TritonToLLVM.cpp

  DEPENDS
  MLIRStructuredConversionIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRPass
  MLIRTransformUtils
)
