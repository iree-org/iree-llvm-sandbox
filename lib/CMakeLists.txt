add_subdirectory(Dialects)
add_subdirectory(Transforms)

add_mlir_public_c_api_library(IREELinalgTensorSandboxCAPI
  CAPI.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  IREELinalgTensorSandbox
)

add_mlir_library(IREELinalgTensorSandbox
  LinalgTensorCodegenDriver.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  MLIRAsync
  MLIRGPUOps
  MLIRLinalgTransforms

  DEPENDS
  IREELinalgTensorSandboxTransforms
  RunnersPassIncGen
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtPassIncGen
)
