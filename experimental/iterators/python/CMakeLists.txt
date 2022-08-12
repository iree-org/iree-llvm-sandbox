include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_iterators`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_iterators.")

# ###############################################################################
# Sources
# ###############################################################################
declare_mlir_python_sources(IteratorsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IteratorsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_iterators"
  TD_FILE dialects/IteratorsOps.td
  SOURCES
  dialects/iterators.py
  DIALECT_NAME iterators
)

declare_mlir_python_extension(IteratorsPythonSources.DialectExtension
  MODULE_NAME _iteratorsDialects
  ADD_TO_PARENT IteratorsPythonSources
  SOURCES
  IteratorsDialects.cpp
  EMBED_CAPI_LINK_LIBS
  IteratorsCAPI
)

declare_mlir_python_extension(IteratorsPythonSources.PassesExtension
  MODULE_NAME _mlirIteratorsPasses
  ADD_TO_PARENT IteratorsPythonSources
  SOURCES
  IteratorsPasses.cpp
  EMBED_CAPI_LINK_LIBS
  IteratorsCAPI
)

declare_mlir_python_sources(IteratorsPythonSources.ExecutionEngine
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_iterators"
  ADD_TO_PARENT IteratorsPythonSources
  SOURCES_GLOB
    runtime/*.py
)

# ###############################################################################
# Common CAPI
# ###############################################################################
add_mlir_python_common_capi_library(IteratorsPythonCAPI
  INSTALL_COMPONENT IteratorsPythonModules
  INSTALL_DESTINATION python_packages/iterators/mlir_iterators/_mlir_libs
  OUTPUT_DIRECTORY "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/mlir_iterators/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
  IteratorsPythonSources
  MLIRPythonExtension.AllPassesRegistration
  MLIRPythonSources.Core
  MLIRPythonSources.ExecutionEngine
  MLIRPythonSources.Passes
)

# ###############################################################################
# Instantiation of all Python modules
# ###############################################################################
add_mlir_python_modules(IteratorsPythonModules
  ROOT_PREFIX "${IREE_LLVM_SANDBOX_BINARY_DIR}/python_packages/mlir_iterators"
  INSTALL_PREFIX "python_packages/mlir_iterators"
  DECLARED_SOURCES
  IteratorsPythonSources
  MLIRPythonExtension.AllPassesRegistration
  MLIRPythonSources.Core
  MLIRPythonSources.ExecutionEngine
  MLIRPythonSources.Passes
  COMMON_CAPI_LINK_LIBS
  IteratorsPythonCAPI
)
