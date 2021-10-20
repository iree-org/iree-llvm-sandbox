include(AddMLIRPython)
include(MLIRDetectPythonEnv)

mlir_configure_python_dev_packages()

declare_mlir_python_sources(IREELLVMSandboxSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  SOURCES iree_sandbox.py
)

declare_mlir_python_extension(IREELLVMSandboxSources.API
  MODULE_NAME _ireeSandbox
  ADD_TO_PARENT IREELLVMSandboxSources
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/PythonAPI.cpp
  EMBED_CAPI_LINK_LIBS
    IREELinalgTensorSandboxCAPI
)

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(IREELLVMSandboxCAPI
  INSTALL_COMPONENT IREELLVMSandboxPythonModules
  INSTALL_DESTINATION python_packages/iree_llvm_sandbox/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_package/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    IREELLVMSandboxSources
)

# Include the main MLIR API in our python package, linking against our CAPI
# library as above.
add_mlir_python_modules(IREELLVMSandboxPythonModules
  ROOT_PREFIX "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_package/mlir"
  INSTALL_PREFIX "python_packages/iree_llvm_sandbox/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    IREELLVMSandboxSources
  COMMON_CAPI_LINK_LIBS
    IREELLVMSandboxCAPI
  )
