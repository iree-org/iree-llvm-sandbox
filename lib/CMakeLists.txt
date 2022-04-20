get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Whether we are using IREE of the sandbox mirror, 
list(APPEND dialect_libs
  IREELinalgExtDialect
  IREELinalgExtOpInterfaceImpl
  IREELinalgTransformDialect
  IREELinalgTransformDialectTransforms
  MLIRVectorExt
  MLIRVectorExtTransforms
  IREESandboxTransforms
)

add_subdirectory(Dialect)
add_subdirectory(ExecutionEngine)
add_subdirectory(Passes)
if (NOT SANDBOX_ENABLE_IREE_DIALECTS)
  add_subdirectory(Transforms)
endif()

if (SANDBOX_ENABLE_ALP)
  list(APPEND dialect_libs
    ExperimentalAlpTransforms
  )
endif()

if (SANDBOX_ENABLE_ITERATORS)
  list(APPEND dialect_libs
    MLIRIterators
    MLIRIteratorsToLLVM
  )
endif()

set(IREE_DIALECTS_DEPENDS
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

  ${dialect_libs}

  DEPENDS
  mlir-headers
  ${IREE_DIALECTS_DEPENDS}
  MLIRLinalgOpsIncGen
)

add_mlir_library(IREESandboxRegistration
  Registration.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRPass
  IREESandboxDriver

  DEPENDS
  mlir-headers
  ${IREE_DIALECTS_DEPENDS}
  MLIRLinalgOpsIncGen
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

  ${dialect_libs}
  
  # Sandbox libraries
  IREESandboxDriver
  IREESandboxRegistration
)
