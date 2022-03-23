//===- PassDetail.h - Iterators pass class details --------------*- C++ -*-===//

#ifndef EXPERIMENTAL_ITERATORS_LIB_CONVERSION_PASSDETAIL_H
#define EXPERIMENTAL_ITERATORS_LIB_CONVERSION_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "iterators/Conversion/Passes.h.inc"

} // namespace mlir

#endif // EXPERIMENTAL_ITERATORS_LIB_CONVERSION_PASSDETAIL_H
