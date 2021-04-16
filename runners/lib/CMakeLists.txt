add_llvm_library(MLIRLinalgTensorSandbox
  ConvertToGPUDialect.cpp
  LinalgComprehensiveBufferizePass.cpp
  LinalgDistributePass.cpp
  LinalgTensorCodegenStrategy.cpp
  LinalgTileAndFuse.cpp
  LinalgTiledLoopToSCFPass.cpp
  Registration.cpp
)

target_link_libraries(
  MLIRLinalgTensorSandbox
  MLIRLinalgTransforms
  MLIRGPU
)
