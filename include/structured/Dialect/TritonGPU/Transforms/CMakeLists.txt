set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix TritonGPU)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix TritonGPU)
add_public_tablegen_target(MLIRTritonGPUTransformsCAPIIncGen)
