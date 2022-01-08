get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(SANDBOX_ENABLE_IREE_DIALECTS)
  list(APPEND dialect_libs
    IREEInputDialect
    IREELinalgExtDialect
    IREELinalgExtPasses
  )
endif()

add_subdirectory(Dialects)
add_subdirectory(Transforms)

set(IREE_LINALG_TENSOR_SANDBOX_LIBRARIES
  MLIRLinalgExt
  MLIRLinalgExtTransforms
  MLIRLinalgTransformOps
  MLIRLinalgTransformTransforms
  MLIRVectorExt
  MLIRVectorExtTransform
  IREELinalgTensorSandboxTransforms
)

if (SANDBOX_ENABLE_ALP)
  list(APPEND IREE_LINALG_TENSOR_SANDBOX_LIBRARIES
    ExperimentalAlpTransforms
  )
endif()


set(IREE_LINALG_TENSOR_SANDBOX_DEPENDS
  DriverPassIncGen
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtPassIncGen
)

add_mlir_library(IREELinalgTensorSandboxDriver
  LinalgTensorCodegenDriver.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  MLIRAsync
  MLIRGPUOps
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRAffineToStandard
  MLIRMemRefTransforms
  MLIRSCFToStandard
  MLIRLinalgToLLVM
  MLIRVectorToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM

  # Sandbox libraries
  ${IREE_LINALG_TENSOR_SANDBOX_LIBRARIES}

  DEPENDS
  ${IREE_LINALG_TENSOR_SANDBOX_DEPENDS}
  MLIRLinalgOpsIncGen
)

add_mlir_library(IREELinalgTensorSandboxRegistration
  Registration.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRPass

  # Sandbox libraries
  IREELinalgTensorSandboxDriver
  ${IREE_LINALG_TENSOR_SANDBOX_LIBRARIES}

  # Tests.
  MLIRLinalgTransformTestPasses
  MLIRTransformsExtTestPasses
  MLIRVectorExtTestPasses

  DEPENDS
  ${IREE_LINALG_TENSOR_SANDBOX_DEPENDS}
  MLIRLinalgOpsIncGen
  MLIRLinalgTransformOpsIncGen
)

add_mlir_public_c_api_library(IREELinalgTensorSandboxCAPI
  CAPI.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  MLIRCAPIRegistration
  MLIRPass

  # Sandbox libraries
  IREELinalgTensorSandboxDriver
  IREELinalgTensorSandboxRegistration
  ${IREE_LINALG_TENSOR_SANDBOX_LIBRARIES}
)
