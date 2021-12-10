get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  OrcJIT
  )

include_directories(include/)
add_subdirectory(include)
add_subdirectory(lib)
