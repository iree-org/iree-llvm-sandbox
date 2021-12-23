if(SANDBOX_ENABLE_IREE_DIALECTS)
  list(APPEND dialect_libs
    IREEInputDialect
    IREELinalgExtDialect
    IREELinalgExtPasses
  )
endif()

if (SANDBOX_ENABLE_ALP)
  list(APPEND dialect_libs
    ExperimentalAlpTransforms
  )
endif()

add_llvm_executable(mlir-proto-opt
  mlir-proto-opt.cpp
)

target_link_libraries(mlir-proto-opt
PRIVATE
  MLIROptLib

  # Sandbox libs.
  IREELinalgTensorSandboxDriver
  IREELinalgTensorSandboxRegistration
)

mlir_check_all_link_libraries(mlir-proto-opt)
