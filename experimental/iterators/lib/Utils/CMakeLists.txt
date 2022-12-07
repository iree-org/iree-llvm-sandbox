add_mlir_library(IteratorsUtils
  NameAssigner.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
)
