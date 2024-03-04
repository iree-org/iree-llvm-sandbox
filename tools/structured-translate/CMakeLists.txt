set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS )
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(structured-translate
  structured-translate.cpp
  )
add_dependencies(structured-tools structured-translate)

target_link_libraries(structured-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTargetSubstraitPB
  MLIRTranslateLib
  MLIRSupport
  )

mlir_check_all_link_libraries(structured-translate)
