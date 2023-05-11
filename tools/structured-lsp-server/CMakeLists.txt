get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(structured-lsp-server
  structured-lsp-server.cpp
)
add_dependencies(structured-tools structured-lsp-server)

target_link_libraries(structured-lsp-server
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRLspServerLib
  MLIRSupport
  MLIRIR
)

mlir_check_all_link_libraries(structured-lsp-server)
