#ifndef THIRD_PARTY_MLIR_EDGE_JASC_TRANSFORMOPS_JASCTRANSFORMOPS_H_
#define THIRD_PARTY_MLIR_EDGE_JASC_TRANSFORMOPS_JASCTRANSFORMOPS_H_

#include <optional>

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "jasc_transform_ops.h.inc"

#endif  // THIRD_PARTY_MLIR_EDGE_JASC_TRANSFORMOPS_JASCTRANSFORMOPS_H_
