include_directories(${LLVM_EXTERNAL_TRITON_SOURCE_DIR}/include)
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StructuredPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mlir_structured"
  TD_FILE dialects/TritonOps.td
  SOURCES
  dialects/triton.py
  DIALECT_NAME tt
)