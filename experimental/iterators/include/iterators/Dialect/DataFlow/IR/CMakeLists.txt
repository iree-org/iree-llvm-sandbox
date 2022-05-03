add_mlir_dialect(DataFlowOps dataflow)
add_dependencies(MLIRDataFlow MLIRDataFlowOpsIncGen)

add_dependencies(mlir-headers
  MLIRDataFlowOpsIncGen
)
