#include "dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "ops.h"

// Include code generated from dialect.td.
#include "dialect/dialect.cc.inc"

namespace jasc {

void JascDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialect/ops.cc.inc"
      >();
}

}  // namespace jasc
