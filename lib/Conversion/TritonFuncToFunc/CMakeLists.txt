add_mlir_conversion_library(MLIRTritonFuncToFunc
  TritonFuncToFunc.cpp

  DEPENDS
  MLIRStructuredConversionIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
