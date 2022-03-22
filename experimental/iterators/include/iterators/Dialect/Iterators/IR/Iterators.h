//===-- IteratorsDialect.h - Iterators dialect ------------------*- C++ -*-===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"

#include "iterators/Dialect/Iterators/IR/IteratorsOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOps.h.inc"

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H
