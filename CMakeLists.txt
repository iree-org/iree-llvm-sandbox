cmake_minimum_required(VERSION 3.18)
project(iree_llvm_sandbox CXX)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR})
list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR}/../llvm)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPythonExtension)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(MLIR_BINDINGS_PYTHON_ENABLED)
  include(MLIRDetectPythonEnv)
  find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
    COMPONENTS Interpreter Development NumPy REQUIRED)
  message(STATUS "Found python include dirs: ${Python3_INCLUDE_DIRS}")
  message(STATUS "Found python libraries: ${Python3_LIBRARIES}")
  message(STATUS "Found numpy v${Python3_NumPy_VERSION}: ${Python3_NumPy_INCLUDE_DIRS}")
  mlir_detect_pybind11_install()
  find_package(pybind11 2.6 CONFIG REQUIRED)
  message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
  message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
                 "suffix = '${PYTHON_MODULE_SUFFIX}', "
                 "extension = '${PYTHON_MODULE_EXTENSION}")
endif()

add_subdirectory(runners)
