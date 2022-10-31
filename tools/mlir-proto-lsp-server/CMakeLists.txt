get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(mlir-proto-lsp-server
  mlir-proto-lsp-server.cpp
)

list(APPEND dialect_libs MLIRVectorExt MLIRVectorExtTransforms)

if(SANDBOX_ENABLE_ALP)
  list(APPEND dialect_libs ExperimentalAlpTransforms)
endif()

if(SANDBOX_ENABLE_ITERATORS)
  list(APPEND dialect_libs MLIRIterators)
  list(APPEND dialect_libs MLIRTabular)
  list(APPEND conversion_libs MLIRIteratorsToLLVM)
  list(APPEND conversion_libs MLIRTabularToLLVM)
  list(APPEND conversion_libs MLIRStatesToLLVM)
endif()

target_link_libraries(mlir-proto-lsp-server
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRLspServerLib
  MLIRSupport
  MLIRIR
)

mlir_check_all_link_libraries(mlir-proto-lsp-server)
