add_subdirectory(CAPython)

add_llvm_library(MLIRLinalgTensorSandbox
  ConvertToGPUDialect.cpp
  LinalgComprehensiveBufferizePass.cpp
  LinalgTensorCodegenStrategy.cpp
  LinalgTileAndDistribute.cpp
  LinalgTiledLoopToSCFPass.cpp
)

target_link_libraries(MLIRLinalgTensorSandbox MLIRLinalgTransforms)
