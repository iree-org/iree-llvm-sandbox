
add_mlir_library(IREESandboxTransforms
  FuseFillIntoReduction.cpp

  LINK_LIBS PRIVATE
  MLIRGPUOps
  MLIRLinalg
  MLIRLinalgTransforms

  DEPENDS
  mlir-headers
  DriverPassIncGen
  IREELinalgExtIncGen
  IREELinalgExtInterfacesIncGen
)
