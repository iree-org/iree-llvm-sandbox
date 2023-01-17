include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under iree.compiler.
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=iree.compiler.")

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

declare_mlir_python_sources(SandboxSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  SOURCES
    sandbox/compilation.py
    sandbox/experts.py
    sandbox/harness.py
    sandbox/iree_sandbox.py
    sandbox/nevergrad_parallel_utils.py
    sandbox/nevergrad_searchable_strategies.py
    sandbox/nevergrad_tuner_utils.py
    sandbox/pdl_utils.py
    sandbox/plotting.py
    sandbox/problem_definition.py
    sandbox/transform.py
    sandbox/transforms.py
    sandbox/utils.py
    sandbox/variables.py
)
    
declare_mlir_python_extension(SandboxSources.API
  MODULE_NAME _ireeSandbox
  ADD_TO_PARENT SandboxSources
  SOURCES
    PythonAPI.cpp
)

################################################################################
# CAPI
################################################################################

# Bundle our own, self-contained CAPI library.
add_mlir_python_common_capi_library(SandboxSandboxCAPI
  INSTALL_COMPONENT SandboxSandboxPythonModules
  INSTALL_DESTINATION python_packages/iree_llvm_sandbox/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    SandboxSources
)

################################################################################
# Instantiation of all Python modules
################################################################################

# Include the main MLIR API in our python packages, linking against our CAPI
# library as above.
add_mlir_python_modules(SandboxSandboxPythonModules
  ROOT_PREFIX "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/mlir"
  INSTALL_PREFIX "python_packages/iree_llvm_sandbox/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    SandboxSources
  COMMON_CAPI_LINK_LIBS
    SandboxSandboxCAPI
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
    MLIRPythonExtension.RegisterEverything
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
    MLIRPythonExtension.RegisterEverything
    ${_addl_sources}
  COMMON_CAPI_LINK_LIBS
    IREECAPI
  )
