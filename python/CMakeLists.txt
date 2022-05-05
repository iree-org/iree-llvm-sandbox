include(AddMLIRPython)

set(_addl_sources)
if(SANDBOX_ENABLE_IREE_DIALECTS)
  list(APPEND _addl_sources
    IREEDialectsPythonSources
    IREEDialectsPythonExtensions
  )
endif()

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(IREELLVMSandboxSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  SOURCES iree_sandbox.py
)
    
declare_mlir_python_extension(IREELLVMSandboxSources.API
  MODULE_NAME _ireeSandbox
  ADD_TO_PARENT IREELLVMSandboxSources
  SOURCES
    PythonAPI.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    IREESandboxRegistration
)

################################################################################
# Common CAPI
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(IREELLVMSandboxCAPI
  INSTALL_COMPONENT IREELLVMSandboxPythonModules
  INSTALL_DESTINATION python_packages/iree_llvm_sandbox/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    IREELLVMSandboxSources
    ${_addl_sources}
)

# If we are not reusing iree-dialects as is, export our own.
if (NOT SANDBOX_ENABLE_IREE_DIALECTS)
   declare_mlir_dialect_python_bindings(
 ADD_TO_PARENT IREELLVMSandboxSources
     ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sandbox"
     TD_FILE dialects/IreeLinalgExtBinding.td
     SOURCES
       dialects/iree_linalg_ext.py
     DIALECT_NAME iree_linalg_ext)
 
   declare_mlir_dialect_python_bindings(
     ADD_TO_PARENT IREELLVMSandboxSources
     ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sandbox"
     TD_FILE dialects/LinalgTransformBinding.td
     SOURCES
       dialects/iree_linalg_transform.py
       dialects/_iree_linalg_transform_ops_ext.py
     DIALECT_NAME iree_linalg_transform)
endif()

################################################################################
# Instantiation of all Python modules
################################################################################

# Include the main MLIR API in our python packages, linking against our CAPI
# library as above.
add_mlir_python_modules(IREELLVMSandboxPythonModules
  ROOT_PREFIX "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/mlir"
  INSTALL_PREFIX "python_packages/iree_llvm_sandbox/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    IREELLVMSandboxSources
    ${_addl_sources}
  COMMON_CAPI_LINK_LIBS
    IREELLVMSandboxCAPI
  )
