add_mlir_dialect(LinalgExtOps linalg_ext)
add_mlir_interface(LinalgExtInterfaces)
add_dependencies(mlir-headers MLIRLinalgExtInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LinalgExt)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix LinalgExt)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix LinalgExt)
add_public_tablegen_target(MLIRLinalgExtPassIncGen)
