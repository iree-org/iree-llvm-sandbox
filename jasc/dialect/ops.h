#ifndef THIRD_PARTY_MLIR_EDGE_JASC_DIALECT_OPS_H_
#define THIRD_PARTY_MLIR_EDGE_JASC_DIALECT_OPS_H_

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

#define GET_OP_CLASSES
#include "dialect/ops.h.inc"

#endif  // THIRD_PARTY_MLIR_EDGE_JASC_DIALECT_OPS_H_
