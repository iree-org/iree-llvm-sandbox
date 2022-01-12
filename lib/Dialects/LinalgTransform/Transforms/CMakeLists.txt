add_mlir_library(MLIRLinalgTransformTransforms
  ScopedTransform.cpp
  TrackingCSE.cpp
  TrackingRewriteDriver.cpp
  TransformInterpreter.cpp
  TrackingListener.cpp

  DEPENDS
  MLIRLinalgOpsIncGen
  MLIRLinalgInterfacesIncGen
  MLIRLinalgPassIncGen

  LINK_LIBS PUBLIC
  IREELinalgTensorSandboxTransforms
  MLIRAffineToStandard
  MLIRBufferization
  MLIRComprehensiveBufferize
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
