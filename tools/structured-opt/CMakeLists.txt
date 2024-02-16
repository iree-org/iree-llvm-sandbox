get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(structured-opt
        structured-opt.cpp
  )
add_dependencies(structured-tools structured-opt)

set(test_libs
)

target_link_libraries(structured-opt
  PRIVATE
  ${conversion_libs}
  ${extension_libs}
  ${dialect_libs}
  ${test_libs}
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

mlir_check_all_link_libraries(structured-opt)
