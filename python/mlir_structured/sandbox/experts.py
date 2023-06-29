from mlir_structured.sandbox.transforms import Bufferize, LowerToLLVM, LowerVectors

# TODO: After DecomposeToLowerDimensionalNamedOp the op_name to anchor on
# changes: we need a better control mechanism.

# Provide end-to-end batteries including bufferization, vector lowering and
# lowering to LLVM. All options must be passed inline, in particular it is
# not possible to deactivate bufferization or lowerings.
LoweringOnlyExpert = Bufferize.then(LowerVectors).then(LowerToLLVM)
