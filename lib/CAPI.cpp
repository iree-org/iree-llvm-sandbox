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
#include "Dialects/LinalgExt/LinalgExtDialect.h"
#include "Dialects/LinalgExt/Passes.h"

#include "Transforms/Passes.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LinalgExt, linalg_ext,
                                      mlir::linalg_ext::LinalgExtDialect)

void ireeLlvmSandboxRegisterPasses() {
  registerRunnersPasses();
  linalg_ext::registerLinalgExtPasses();
}

void ireeLlvmSandboxRegisterAll(MlirContext context) {
  MlirDialectHandle linalgExtDialect = mlirGetDialectHandle__linalg_ext__();
  mlirDialectHandleRegisterDialect(linalgExtDialect, context);

  ireeLlvmSandboxRegisterPasses();

  DialectRegistry registry;
  unwrap(context)->getDialectRegistry().appendTo(registry);
  linalg_ext::registerTilingInterfaceExternalModels(registry);
  unwrap(context)->appendDialectRegistry(registry);
}
