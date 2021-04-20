//===- Registration.cpp - Register custom passes --------------------------===//
//
// Convert from Linalg ops on tensors to Linalg ops on buffers in a single pass.
// Aggressively try to perform inPlace bufferization and fail if any allocation
// tries to cross function boundaries or if the pattern
// `tensor_load(tensor_memref(x))` is deemed unsafe (very conservative impl for
// now).
//
//===----------------------------------------------------------------------===//
#include "Registration.h"

using namespace mlir;
using namespace mlir::linalg;

void ireeLlvmSandboxRegisterPasses() {
  registerLinalgComprehensiveBufferizePass();
  registerLinalgDistributePass();
  registerLinalgTensorCodegenStrategyPass();
  registerConvertToGPUPass();
}
