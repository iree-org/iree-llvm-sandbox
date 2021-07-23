if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "This project is intended to be built as part of LLVM via "
    "-DLLVM_EXTERNAL_PROJECTS=iree_llvm_sandbox "
    "-DLLVM_EXTERNAL_IREE_LLVM_SANDBOX_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(IREE_LLVM_SANDBOX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "iree_llvm_sandbox build directory: ${IREE_LLVM_SANDBOX_BINARY_DIR}")

set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(runners)
