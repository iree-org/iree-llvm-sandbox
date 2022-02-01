get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if (SANDBOX_ENABLE_ALP)
  list(APPEND dialect_libs
    ExperimentalAlpTransforms
  )
endif()

add_llvm_executable(mlir-proto-opt
  mlir-proto-opt.cpp
)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
)

target_link_libraries(mlir-proto-opt
PRIVATE
  MLIROptLib

  ${LIBS}

  # Sandbox libs.
  IREELinalgTensorSandboxDriver
  IREELinalgTensorSandboxRegistration
)

mlir_check_all_link_libraries(mlir-proto-opt)
