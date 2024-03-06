add_mlir_public_c_api_library(StructuredCAPI
    Dialects.cpp
    Passes.cpp
    Transforms.cpp

  LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRIterators
    MLIRIteratorsToLLVM
    MLIRIteratorsTransforms
    MLIRSubstraitDialect
    MLIRTabular
    MLIRTabularToLLVM
    MLIRTupleDialect
    MLIRTupleTransforms
    MLIRPass
    MLIRStatesToLLVM
)
