add_mlir_library(MLIRIterators
  Iterators.cpp

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRLLVMIR

  DEPENDS
  MLIRIteratorsInterfacesIncGen
  MLIRIteratorsOpsIncGen
)
