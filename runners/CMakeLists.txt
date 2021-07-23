add_subdirectory(lib)
add_subdirectory(python)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(mlir-proto-opt
mlir-proto-opt.cpp
)
target_link_libraries(mlir-proto-opt
PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  IREELinalgTensorSandbox
  IREELinalgTensorSandboxCAPI
)
mlir_check_all_link_libraries(mlir-proto-opt)
