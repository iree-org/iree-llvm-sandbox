add_mlir_library(IteratorsUtils
  MLIRSupport.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCF
)
