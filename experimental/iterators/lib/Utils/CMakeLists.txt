add_mlir_library(IteratorsUtils
  MLIRSupport.cpp
  NameAssigner.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
)
