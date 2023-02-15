add_mlir_library(MLIRIterators
  ArrowUtils.cpp
  Iterators.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRLLVMDialect
  MLIRTabular

  DEPENDS
  MLIRIteratorsInterfacesIncGen
  MLIRIteratorsOpsIncGen
)
