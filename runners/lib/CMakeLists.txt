add_llvm_library(MLIRLinalgTensorSandbox
  ConvertToGPUDialect.cpp
  LinalgComprehensiveBufferizePass.cpp
  LinalgTensorCodegenStrategy.cpp
  LinalgTileAndDistribute.cpp
  LinalgTiledLoopToSCFPass.cpp
  Registration.cpp
)

target_link_libraries(
  MLIRLinalgTensorSandbox
  MLIRLinalgTransforms
  MLIRGPU
)
