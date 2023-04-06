get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(iterators-lsp-server
  iterators-lsp-server.cpp
)
add_dependencies(iterators-tools iterators-lsp-server)

list(APPEND dialect_libs
  MLIRIterators
  MLIRTabular
  MLIRTupleDialect
  )
list(APPEND conversion_libs
  MLIRIteratorsToLLVM
  MLIRTabularToLLVM
  MLIRStatesToLLVM
  )
set(iterators_transform_libs
  MLIRIteratorsTransforms
  MLIRTupleTransforms
  )

target_link_libraries(iterators-lsp-server
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${iterators_transform_libs}
  MLIRLspServerLib
  MLIRSupport
  MLIRIR
)

mlir_check_all_link_libraries(iterators-lsp-server)
