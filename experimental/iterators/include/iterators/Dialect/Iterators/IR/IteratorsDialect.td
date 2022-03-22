//===-- IteratorsDialect.td - Iterators dialect ------------*- tablegen -*-===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSDIALECT
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Iterators_Dialect : Dialect {
  let name = "iterators";
  let cppNamespace = "::mlir::iterators";
  let summary = "Dialect implementing database-style iterators.";
  let useDefaultTypePrinterParser = 1;
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSDIALECT
