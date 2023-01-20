if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "This project is intended to be built as part of LLVM via "
    "-DLLVM_EXTERNAL_PROJECTS=iterators "
    "-DLLVM_EXTERNAL_ITERATORS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

################################################################################
# Set some variables
################################################################################
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)

list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
set(MLIR_TABLEGEN_EXE mlir-tblgen)

set(ITERATORS_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ITERATORS_INCLUDE_DIRS ${ITERATORS_MAIN_SRC_DIR}/include)
set(ITERATORS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(ITERATORS_TABLEGEN_OUTPUT_DIR ${ITERATORS_BINARY_DIR}/include)
message(STATUS "Iterators build directory: ${ITERATORS_BINARY_DIR}")

################################################################################
# Set include paths
################################################################################
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

include_directories(${ITERATORS_INCLUDE_DIRS})
include_directories(${ITERATORS_TABLEGEN_OUTPUT_DIR})

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
# Enable LLVM stuff
################################################################################
include(TableGen)
include(AddLLVM)
include(AddMLIR)

################################################################################
# Subdirs to recurse into
################################################################################
add_custom_target(iterators-all)

add_subdirectory(examples)
add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(python)
add_subdirectory(tools)
