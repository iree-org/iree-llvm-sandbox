#ifndef THIRD_PARTY_MLIR_EDGE_JASC_TRANSFORM_OPS_DIALECT_EXTENSION_H_
#define THIRD_PARTY_MLIR_EDGE_JASC_TRANSFORM_OPS_DIALECT_EXTENSION_H_

#include "mlir/IR/DialectRegistry.h"

namespace jasc {
void registerTransformDialectExtension(mlir::DialectRegistry &registry);
}

#endif  // THIRD_PARTY_MLIR_EDGE_JASC_TRANSFORM_OPS_DIALECT_EXTENSION_H_
