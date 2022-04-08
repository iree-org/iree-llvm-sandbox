add_mlir_dialect(IteratorsOps iterators)

set(LLVM_TARGET_DEFINITIONS IteratorsInterfaces.td)
mlir_tablegen(IteratorsOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(IteratorsOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(IteratorsTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(IteratorsTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRIteratorsInterfacesIncGen)
