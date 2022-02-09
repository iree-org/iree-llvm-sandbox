add_mlir_library(MLIRLinalgTransformTransforms
  ExpertExpansion.cpp
  PDL.cpp
  ScopedTransform.cpp
  TrackingCSE.cpp
  TrackingRewriteDriver.cpp
  TransformInterpreter.cpp
  TrackingListener.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREELinalgTensorSandboxTransforms
  MLIRAffineToStandard
  MLIRBufferization
  MLIRIR
  MLIRLinalg
  MLIRLinalgToLLVM
  MLIRLinalgTransformOps
  MLIRLinalgTransforms
  MLIRLLVMIR
  MLIRMath
  MLIRMathToLLVM
  MLIRMemRef
  MLIRMemRefToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRTensor
  MLIRTransforms
  MLIRVector
  MLIRVectorToLLVM
)
