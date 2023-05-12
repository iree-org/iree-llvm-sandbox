add_mlir_dialect_library(MLIRIterators
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
