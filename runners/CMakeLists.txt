set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
)

add_llvm_executable(mlir-proto-opt
  ConvertToGPUDialect.cpp
  mlir-proto-opt.cpp
  LinalgComprehensiveBufferizePass.cpp
  LinalgTensorCodegenStrategy.cpp
  LinalgTiledLoopToSCFPass.cpp
  LinalgTileToGeneric.cpp
)
target_link_libraries(mlir-proto-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(mlir-proto-opt)
