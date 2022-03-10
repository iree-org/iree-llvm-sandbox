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
  # Dialects
  MLIRAsync
  MLIRLinalg
  MLIRLinalgExt
  MLIRLinalgExtTransforms
  MLIRPDL
  MLIRControlFlowInterfaces
  MLIRRewrite
  # Transforms
  MLIRAsyncTransforms
  MLIRLinalgTransforms
  MLIRAffineToStandard
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  # Conversions
  MLIRAsyncToLLVM
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRVectorToLLVM
  MLIRLinalgToLLVM
  MLIRSCFToControlFlow
)
