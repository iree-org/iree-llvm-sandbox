add_mlir_library(MLIRLinalgTransformTransforms
  TrackingCSE.cpp
  TrackingRewriteDriver.cpp
  TransformInterpreter.cpp

  DEPENDS
  MLIRLinalgOpsIncGen
  MLIRLinalgInterfacesIncGen
  MLIRLinalgPassIncGen

  LINK_LIBS PUBLIC
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
