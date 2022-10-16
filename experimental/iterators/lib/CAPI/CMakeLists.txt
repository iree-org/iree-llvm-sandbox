add_mlir_public_c_api_library(IteratorsCAPI
    Dialects.cpp
    Passes.cpp
  LINK_LIBS PUBLIC
    MLIRIterators
    MLIRIteratorsToLLVM
    MLIRTabular
    MLIRTabularToLLVM
    MLIRPass
    MLIRStatesToLLVM
)
