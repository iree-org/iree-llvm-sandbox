//===- CAPI.cpp - CAPI implementation -------------------------------------===//
//
// Convert from Linalg ops on tensors to Linalg ops on buffers in a single pass.
// Aggressively try to perform inPlace bufferization and fail if any allocation
// tries to cross function boundaries or if the pattern
// `tensor_load(tensor_memref(x))` is deemed unsafe (very conservative impl for
// now).
//
//===----------------------------------------------------------------------===//

#include "CAPI.h"
#include "Dialects/LinalgTransform/LinalgTransformOps.h"
#include "Registration.h"

#include "mlir-c/Dialect/Linalg.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    LinalgTransform, linalg_transform,
    mlir::linalg::transform::LinalgTransformDialect)

void ireeLlvmSandboxRegisterAll(MlirContext context) {
  MlirDialectHandle linalgTransformDialect =
      mlirGetDialectHandle__linalg_transform__();
  mlirDialectHandleRegisterDialect(linalgTransformDialect, context);

  registerOutsideOfDialectRegistry();

  DialectRegistry registry;
  unwrap(context)->getDialectRegistry().appendTo(registry);
  registerIntoDialectRegistry(registry);
  unwrap(context)->appendDialectRegistry(registry);
}
