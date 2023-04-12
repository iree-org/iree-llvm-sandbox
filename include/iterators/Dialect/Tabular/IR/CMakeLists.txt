add_mlir_dialect(TabularOps tabular)
add_dependencies(MLIRTabular MLIRTabularOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TabularInterfaces.td)
mlir_tablegen(TabularOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TabularOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(TabularTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TabularTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRTabularInterfacesIncGen)
add_dependencies(MLIRTabular MLIRTabularInterfacesIncGen)

add_dependencies(mlir-headers
  MLIRTabularInterfacesIncGen
  MLIRTabularOpsIncGen
)
