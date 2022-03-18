//===-- IteratorsDialect.cpp - Iterators dialect ----------------*- C++ -*-===//

#include "Dialects/Iterators/Iterators.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iterators;

//===----------------------------------------------------------------------===//
// Iterators dialect
//===----------------------------------------------------------------------===//

#include "Dialects/Iterators/IteratorsOpsDialect.cpp.inc"

void IteratorsDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "Dialects/Iterators/IteratorsOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialects/Iterators/IteratorsOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Iterators operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialects/Iterators/IteratorsOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Iterators types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Dialects/Iterators/IteratorsOpsTypes.cpp.inc"
