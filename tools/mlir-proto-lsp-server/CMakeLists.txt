get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(mlir-proto-lsp-server
  mlir-proto-lsp-server.cpp
)

target_link_libraries(mlir-proto-lsp-server
PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRLspServerLib
  MLIRSupport
  MLIRIR

  # Sandbox libs.
  IREESandboxRegistration
)

mlir_check_all_link_libraries(mlir-proto-lsp-server)
