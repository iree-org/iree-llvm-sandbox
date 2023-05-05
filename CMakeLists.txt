cmake_minimum_required(VERSION 3.13.4)
project(structured LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
  set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
  list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
  set(MLIR_TABLEGEN_EXE mlir-tblgen)
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
endif()


################################################################################
# Set some variables
################################################################################

set(STRUCTURED_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STRUCTURED_INCLUDE_DIRS ${STRUCTURED_MAIN_SRC_DIR}/include)
set(STRUCTURED_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(STRUCTURED_TABLEGEN_OUTPUT_DIR ${STRUCTURED_BINARY_DIR}/include)
message(STATUS "Structured build directory: ${STRUCTURED_BINARY_DIR}")

################################################################################
# Set include paths
################################################################################
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${STRUCTURED_INCLUDE_DIRS})
include_directories(${STRUCTURED_TABLEGEN_OUTPUT_DIR})

################################################################################
# Enable python (assumes enabled MLIR bindings via MLIR_ENABLE_BINDINGS_PYTHON)
################################################################################
if(NOT DEFINED MLIR_ENABLE_BINDINGS_PYTHON)
  message(FATAL_ERROR
    "This project requires MLIR_ENABLE_BINDINGS_PYTHON=ON")
endif()

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

################################################################################
# Subdirs to recurse into
################################################################################
add_custom_target(structured-all)

add_subdirectory(examples)
add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(tools)
