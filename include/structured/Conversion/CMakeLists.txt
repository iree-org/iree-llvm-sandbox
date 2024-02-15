# This is last such that it (implicitly) depends on the previous passes.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StructuredConversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix StructuredConversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix StructuredConversion)
add_public_tablegen_target(MLIRStructuredConversionIncGen)
