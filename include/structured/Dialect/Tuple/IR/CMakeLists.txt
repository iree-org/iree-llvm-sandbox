add_mlir_dialect(TupleOps tuple)
add_dependencies(MLIRTupleDialect MLIRTupleOpsIncGen)

add_dependencies(mlir-headers
  MLIRTupleOpsIncGen
)
