include_directories(${triton_BINARY_DIR}/include)
include_directories(${LLVM_EXTERNAL_TRITON_SOURCE_DIR}/include)

add_mlir_public_c_api_library(TritonCAPI
  Dialects.cpp

  LINK_LIBS PUBLIC
  TritonIR
)