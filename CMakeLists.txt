if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "This project is intended to be built as part of LLVM via "
    "-DLLVM_EXTERNAL_PROJECTS=iree_llvm_sandbox "
    "-DLLVM_EXTERNAL_IREE_LLVM_SANDBOX_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(IREE_LLVM_SANDBOX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "iree_llvm_sandbox build directory: ${IREE_LLVM_SANDBOX_BINARY_DIR}")

# Compile in IREE dialects.
set(LLVM_EXTERNAL_IREE_DIALECTS_SOURCE_DIR "" CACHE PATH "Path to iree-dialects project")
if(LLVM_EXTERNAL_IREE_DIALECTS_SOURCE_DIR)
  message(STATUS "Enabling iree-dialects in sandbox")
  add_compile_definitions("SANDBOX_ENABLE_IREE_DIALECTS")
  # TODO: For finding iree-dialects-c. Should not be necessary.
  include_directories(SYSTEM ${LLVM_EXTERNAL_IREE_DIALECTS_SOURCE_DIR}/include)
endif()

set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
set(MLIR_TABLEGEN_EXE mlir-tblgen)

include(TableGen)
include(AddLLVM)
include(AddMLIR)


include_directories(include)
include_directories(${IREE_LLVM_SANDBOX_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(tools)
