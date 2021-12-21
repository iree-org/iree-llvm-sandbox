#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H

#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;
class DominanceInfo;
class Value;

void eliminateCommonSubexpressionsWithTrackedOps(
    Operation *root, DenseMap<Value, SmallVector<Operation *, 4>> &trackedOps,
    DominanceInfo *domInfo = nullptr);
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H
