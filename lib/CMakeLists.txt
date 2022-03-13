get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(SANDBOX_ENABLE_IREE_DIALECTS)
  list(APPEND dialect_libs
    IREEInputDialect
    IREELinalgExtDialect
    IREELinalgExtPasses
    IREELinalgExtOpInterfaceImpl
    IREESandboxTransforms
  )
endif()

add_subdirectory(Dialects)
add_subdirectory(ExecutionEngine)
add_subdirectory(Transforms)

set(IREE_LINALG_TENSOR_SANDBOX_LIBRARIES
  MLIRLinalgTransformOps
  MLIRLinalgTransformTransforms
  MLIRVectorExt
  MLIRVectorExtTransform
)

if (SANDBOX_ENABLE_ALP)
  list(APPEND IREE_LINALG_TENSOR_SANDBOX_LIBRARIES
    ExperimentalAlpTransforms
  )
endif()


set(IREE_LINALG_TENSOR_SANDBOX_DEPENDS
  DriverPassIncGen
  IREELinalgExtInterfacesIncGen
  IREELinalgExtIncGen
)

add_mlir_library(IREESandboxDriver
  LinalgTensorCodegenDriver.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  IREELinalgExtOpInterfaceImpl
  # Dialects
  MLIRAsync
  MLIRGPUOps
  MLIRLinalg
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
  MLIRSCFToControlFlow
  MLIRVectorToLLVM

  # Sandbox libraries
  ${IREE_LINALG_TENSOR_SANDBOX_LIBRARIES}

  DEPENDS
  mlir-headers
  ${IREE_LINALG_TENSOR_SANDBOX_DEPENDS}
  MLIRLinalgOpsIncGen
)

add_mlir_library(IREESandboxRegistration
  Registration.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRPass

  # Sandbox libraries
  IREESandboxDriver
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

add_mlir_public_c_api_library(IREESandboxCAPI
  CAPI.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  mlir-headers

  LINK_LIBS PRIVATE
  MLIRCAPIRegistration
  MLIRIR
  MLIRPass

  # Sandbox libraries
  IREESandboxDriver
  IREESandboxRegistration
  ${IREE_LINALG_TENSOR_SANDBOX_LIBRARIES}
)
