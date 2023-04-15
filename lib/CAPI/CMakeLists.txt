add_mlir_public_c_api_library(StructuredCAPI
    Dialects.cpp
    Passes.cpp
    Transforms.cpp
  LINK_LIBS PUBLIC
    MLIRIterators
    MLIRIteratorsToLLVM
    MLIRIteratorsTransforms
    MLIRTabular
    MLIRTabularToLLVM
    MLIRTupleDialect
    MLIRTupleTransforms
    MLIRPass
    MLIRStatesToLLVM
)
