add_mlir_public_c_api_library(StructuredCAPI
    Dialects.cpp
    Passes.cpp
    Transforms.cpp
    Triton.cpp

  DEPENDS
    MLIRTritonConversionIncGen
    MLIRTritonTransformIncGen
    MLIRTritonTransformsCAPIIncGen
    MLIRTritonGPUTransformsCAPIIncGen

  LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRIndexing
    MLIRIterators
    MLIRIteratorsToLLVM
    MLIRIteratorsTransforms
    MLIRTabular
    MLIRTabularToLLVM
    MLIRTritonFuncToFunc
    MLIRTritonToLLVM
    MLIRTupleDialect
    MLIRTupleTransforms
    MLIRPass
    MLIRStatesToLLVM
    TritonGPUIR
    TritonGPUToLLVM
    TritonGPUTransforms
    TritonIR
    TritonToTritonGPU
    TritonTransforms
)
