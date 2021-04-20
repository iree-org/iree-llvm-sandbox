//===- Registration.h - Register custom passes ----------------------------===//
//
// Convert from Linalg ops on tensors to Linalg ops on buffers in a single pass.
// Aggressively try to perform inPlace bufferization and fail if any allocation
// tries to cross function boundaries or if the pattern
// `tensor_load(tensor_memref(x))` is deemed unsafe (very conservative impl for
// now).
//
//===----------------------------------------------------------------------===//

// Defined directly in pass modules.
namespace mlir {
void registerConvertToAsyncPass();
namespace linalg {
void registerLinalgComprehensiveBufferizePass();
void registerLinalgTensorCodegenStrategyPass();
}  // namespace linalg
}  // namespace mlir

// C callable symbol to register everything.
extern "C" {
void ireeLlvmSandboxRegisterPasses();
}
