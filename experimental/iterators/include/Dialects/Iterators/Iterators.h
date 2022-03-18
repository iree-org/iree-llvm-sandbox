//===-- IteratorsDialect.h - Iterators dialect ------------------*- C++ -*-===//

#ifndef DIALECTS_ITERATORS_ITERATORSDIALECT_H
#define DIALECTS_ITERATORS_ITERATORSDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"

#include "Dialects/Iterators/IteratorsOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialects/Iterators/IteratorsOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialects/Iterators/IteratorsOps.h.inc"

#endif // DIALECTS_ITERATORS_ITERATORSDIALECT_H
