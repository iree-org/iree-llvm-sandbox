add_mlir_library(MLIRLinalgTransformOps
  LinalgTransformOps.cpp
  PDL.cpp
  ScopedTransform.cpp
  TransformOpInterface.cpp
  TrackingListener.cpp
  TrackingRewriteDriver.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREELinalgTensorSandboxTransforms
  MLIRIR
  MLIRLinalg
  MLIRPDL
  MLIRControlFlowInterfaces
  MLIRRewrite
  MLIRLinalgTransforms
  MLIRAffineToStandard
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRVectorToLLVM
  MLIRLinalgToLLVM
  MLIRSCFToControlFlow
)
