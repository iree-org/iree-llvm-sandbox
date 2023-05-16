set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Triton)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Triton)
add_public_tablegen_target(MLIRTritonTransformsCAPIIncGen)
