add_mlir_library(MLIRIterators
  Iterators.cpp

  LINK_LIBS PUBLIC
  MLIRFunc
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRLLVMIR

  DEPENDS
  MLIRIteratorsInterfacesIncGen
  MLIRIteratorsOpsIncGen
)
