add_mlir_conversion_library(MLIRTritonSPMDToFuncArgs
  TritonSPMDToFuncArgs.cpp

  DEPENDS
  MLIRStructuredConversionIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
