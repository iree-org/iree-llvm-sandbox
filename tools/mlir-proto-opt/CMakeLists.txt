get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(SANDBOX_ENABLE_IREE_DIALECTS)
  list(APPEND dialect_libs
    IREEInputDialect
    IREELinalgExtDialect
    IREELinalgExtPasses
  )
endif()

add_llvm_executable(mlir-proto-opt
  mlir-proto-opt.cpp
)
target_link_libraries(mlir-proto-opt
PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  IREELinalgTensorSandbox
  IREELinalgTensorSandboxTransforms
  IREELinalgTensorSandboxCAPI
  MLIRLinalgExt
  MLIRLinalgExtTransforms
  MLIROptLib
  MLIRVectorExt
  MLIRVectorExtTestPasses
  MLIRVectorExtTransform
)
if (SANDBOX_ENABLE_ALP)
  target_link_libraries(mlir-proto-opt
  PRIVATE
  ExperimentalAlpTransforms
  )
endif()

mlir_check_all_link_libraries(mlir-proto-opt)
