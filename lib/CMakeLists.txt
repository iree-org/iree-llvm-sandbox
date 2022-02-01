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
add_subdirectory(ExecutionEngine)
add_subdirectory(Transforms)

set(IREE_LINALG_TENSOR_SANDBOX_LIBRARIES
  MLIRLinalgExt
  MLIRLinalgExtOpInterfaceImpl
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
  # Dialects
  MLIRAsync
  MLIRGPUOps
  MLIRLinalg
  MLIRLinalgExtOpInterfaceImpl
  # Transforms
  MLIRAsyncTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  # Conversions
  MLIRAsyncToLLVM
  MLIRAffineToStandard
  MLIRLinalgToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToStandard
  MLIRVectorToLLVM

  # Sandbox libraries
  ${IREE_LINALG_TENSOR_SANDBOX_LIBRARIES}

  DEPENDS
  mlir-headers
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
  mlir-headers
  ${IREE_LINALG_TENSOR_SANDBOX_DEPENDS}
  MLIRLinalgOpsIncGen
  MLIRLinalgTransformOpsIncGen
)

add_mlir_public_c_api_library(IREELinalgTensorSandboxCAPI
  CAPI.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  mlir-headers

  LINK_LIBS PRIVATE
  MLIRCAPIRegistration
  MLIRIR
  MLIRPass

  # Sandbox libraries
  IREELinalgTensorSandboxDriver
  IREELinalgTensorSandboxRegistration
  ${IREE_LINALG_TENSOR_SANDBOX_LIBRARIES}
)
