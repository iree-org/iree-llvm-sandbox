add_mlir_dialect(SubstraitOps substrait)
add_dependencies(MLIRSubstraitDialect MLIRSubstraitOpsIncGen)

add_dependencies(mlir-headers
  MLIRSubstraitOpsIncGen
)
