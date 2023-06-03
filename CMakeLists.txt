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
# Set up Triton
################################################################################

set(TRITON_ROOT_DIR ${STRUCTURED_MAIN_SRC_DIR}/third_party/triton)
set(TRITON_INCLUDE_DIR ${TRITON_ROOT_DIR}/include)
set(TRITON_BINARY_DIR ${STRUCTURED_BINARY_DIR}/third_party/triton)
set(TRITON_TABLEGEN_OUTPUT_DIR ${TRITON_BINARY_DIR}/include)

include_directories(${TRITON_INCLUDE_DIR})
include_directories(${TRITON_TABLEGEN_OUTPUT_DIR})

add_subdirectory(third_party/triton/include)
add_subdirectory(third_party/triton/lib)
add_subdirectory(third_party/triton/test/lib)

add_custom_target(triton-dependencies)
get_property(triton_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_dependencies(triton-dependencies
  ${triton_conversion_libs}
  ${triton_dialect_libs}
)
# Regenerate this list with the following command:
#   (cd ${IREE_LLVM_SANDBOX_SOURCE_DIR}/third_party/triton/ &&
#     git grep -oh -E '^ *(LLVM|MLIR)[a-zA-Z]*$' \
#       -- CMakeLists.txt "**/CMakeLists.txt" | tr -d " " | sort -u)
add_dependencies(triton-dependencies
  LLVMAsmParser
  LLVMCodeGen
  LLVMCore
  LLVMOption
  LLVMSupport
  MLIRAnalysis
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRExecutionEngineUtils
  MLIRGPUDialect
  MLIRGPUToNVVMTransforms
  MLIRGPUToROCDLTransforms
  MLIRGPUTransforms
  MLIRIndexToLLVM
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRNVVMToLLVMIRTranslation
  MLIROptLib
  MLIRPass
  MLIRReduceLib
  MLIRROCDLToLLVMIRTranslation
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRTransformUtils
)
# Regenerate this list with the following command:
#   (cd ${IREE_LLVM_SANDBOX_SOURCE_DIR}/third_party/triton/ &&
#     git grep -E -oh 'libLLVM.*\.a' -- CMakeLists.txt | \
#       sed "s/lib\(.*\)\.a/\1/" | sort -u)
add_dependencies(triton-dependencies
  LLVMAggressiveInstCombine
  LLVMAMDGPUAsmParser
  LLVMAMDGPUCodeGen
  LLVMAMDGPUDesc
  LLVMAMDGPUDisassembler
  LLVMAMDGPUInfo
  LLVMAMDGPUUtils
  LLVMAnalysis
  LLVMAsmParser
  LLVMAsmPrinter
  LLVMBinaryFormat
  LLVMBitReader
  LLVMBitstreamReader
  LLVMBitWriter
  LLVMCodeGen
  LLVMCore
  LLVMCoroutines
  LLVMDebugInfoCodeView
  LLVMDebugInfoDWARF
  LLVMDebugInfoMSF
  LLVMDemangle
  LLVMFrontendOpenMP
  LLVMGlobalISel
  LLVMInstCombine
  LLVMInstrumentation
  LLVMipo
  LLVMIRReader
  LLVMLinker
  LLVMMC
  LLVMMCDisassembler
  LLVMMCParser
  LLVMMIRParser
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo
  LLVMObjCARCOpts
  LLVMObject
  LLVMPasses
  LLVMProfileData
  LLVMRemarks
  LLVMScalarOpts
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  LLVMTextAPI
  LLVMTransformUtils
  LLVMVectorize
)

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
