get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(iterators-opt
  iterators-opt.cpp
  )
add_dependencies(iterators-tools iterators-opt)

list(APPEND dialect_libs
  MLIRIterators
  MLIRTabular
  )
list(APPEND conversion_libs
  MLIRIteratorsToLLVM
  MLIRTabularToLLVM
  MLIRStatesToLLVM
  )
set(iterators_transform_libs
  MLIRIteratorsTransforms
  )

target_link_libraries(iterators-opt
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

mlir_check_all_link_libraries(iterators-opt)
