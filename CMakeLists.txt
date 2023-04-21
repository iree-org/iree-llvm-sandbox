if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "This project is intended to be built as part of LLVM via "
    "-DLLVM_EXTERNAL_PROJECTS=structured "
    "-DLLVM_EXTERNAL_STRUCTURED_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
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
# Enable LLVM stuff
################################################################################
include(TableGen)
include(AddLLVM)
include(AddMLIR)

################################################################################
# Add Triton as an external project
################################################################################

list(APPEND LLVM_EXTERNAL_PROJECTS triton)
list(REMOVE_DUPLICATES LLVM_EXTERNAL_PROJECTS)
set(LLVM_EXTERNAL_TRITON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton CACHE STRING "" FORCE)
set(LLVM_EXTERNAL_PROJECTS ${LLVM_EXTERNAL_PROJECTS} CACHE STRING "" FORCE)
message("-- Applying the patch ${CMAKE_CURRENT_SOURCE_DIR}/triton.patch")
execute_process(COMMAND
        git apply --verbose --allow-empty ${CMAKE_CURRENT_SOURCE_DIR}/triton.patch
        WORKING_DIRECTORY ${LLVM_EXTERNAL_TRITON_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_APPLY
        RESULT_VARIABLE GIT_APPLY
        ERROR_VARIABLE GIT_APPLY)
message(WARNING "GIT_APPLY ${GIT_APPLY}")
add_subdirectory(${LLVM_EXTERNAL_TRITON_SOURCE_DIR})

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
