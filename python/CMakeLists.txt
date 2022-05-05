include(AddMLIRPython)

################################################################################
################################################################################
################################################################################
# Wrapping sandbox python. 
################################################################################
################################################################################
################################################################################

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
# CAPI
################################################################################

# Bundle our own, self-contained CAPI library.
add_mlir_python_common_capi_library(IREELLVMSandboxCAPI
  INSTALL_COMPONENT IREELLVMSandboxPythonModules
  INSTALL_DESTINATION python_packages/iree_llvm_sandbox/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    IREELLVMSandboxSources
)

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
  COMMON_CAPI_LINK_LIBS
    IREELLVMSandboxCAPI
  )


################################################################################
################################################################################
################################################################################
# Wrapping IREE dialects python. 
################################################################################
################################################################################
################################################################################


set(_addl_sources)
list(APPEND _addl_sources
  IREEDialectsPythonSources
  IREEDialectsPythonExtensions
)

################################################################################
# CAPI
################################################################################

# Bundle the IREE dialects CAPI, in a location compatible with just using IREE.
add_mlir_python_common_capi_library(IREECAPI
  INSTALL_COMPONENT IREEPythonModules
  INSTALL_DESTINATION python_packages/iree_llvm_sandbox/iree/compiler/
  OUTPUT_DIRECTORY "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/iree/compiler/"
  RELATIVE_INSTALL_ROOT "../../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    ${_addl_sources}
)

################################################################################
# Instantiation of all Python modules
################################################################################

# Include the main MLIR API in the IREE python packages, linking against the IREECAPI 
# library as above.
add_mlir_python_modules(IREEPythonModules
  ROOT_PREFIX "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/iree/compiler"
  INSTALL_PREFIX "python_packages/iree_llvm_sandboxiree/compiler"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    ${_addl_sources}
  COMMON_CAPI_LINK_LIBS
    IREECAPI
  )
