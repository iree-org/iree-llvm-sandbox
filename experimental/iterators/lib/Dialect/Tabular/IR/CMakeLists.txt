add_mlir_library(MLIRTabular
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
