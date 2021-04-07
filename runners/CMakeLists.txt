add_subdirectory(lib)

set(MLIR_CORE_LIBS
${dialect_libs}
${conversion_libs}
)

add_llvm_executable(mlir-proto-opt
mlir-proto-opt.cpp
)
target_link_libraries(mlir-proto-opt
PRIVATE
  ${MLIR_CORE_LIBS}
  MLIROptLib
  MLIRLinalgTensorSandbox
)
mlir_check_all_link_libraries(mlir-proto-opt)
