add_mlir_library(MLIRIterators
  Iterators.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface

  DEPENDS
  MLIRIteratorsOpsIncGen
  MLIRIteratorsInterfacesIncGen
)
