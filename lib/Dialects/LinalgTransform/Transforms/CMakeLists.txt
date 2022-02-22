add_mlir_library(MLIRLinalgTransformTransforms
  ExpertExpansion.cpp
  TrackingCSE.cpp
  TransformInterpreter.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRIR
  MLIRLinalg
  MLIRLinalgTransformOps
  MLIRLLVMIR
  MLIRMath
  MLIRMathToLLVM
  MLIRMemRef
  MLIRMemRefToLLVM
  MLIRPass
  MLIRTensor
  MLIRTransforms
  MLIRVector
  MLIRVectorToLLVM
)
