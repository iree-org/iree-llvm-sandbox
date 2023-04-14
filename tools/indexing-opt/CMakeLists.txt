get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(indexing-opt
  indexing-opt.cpp
  )

list(APPEND dialect_libs
  MLIRIndexing
  )

target_link_libraries(indexing-opt
  PRIVATE
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

mlir_check_all_link_libraries(indexing-opt)
