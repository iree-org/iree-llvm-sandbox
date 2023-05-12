add_mlir_dialect_library(MLIRTabular
  Tabular.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRLLVMDialect

  DEPENDS
  MLIRTabularInterfacesIncGen
  MLIRTabularOpsIncGen
)
