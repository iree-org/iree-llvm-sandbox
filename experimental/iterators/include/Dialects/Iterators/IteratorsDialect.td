//===-- IteratorsDialect.td - Iterators dialect ------------*- tablegen -*-===//

#ifndef DIALECTS_ITERATORS_ITERATORSDIALECT
#define DIALECTS_ITERATORS_ITERATORSDIALECT

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

#endif // DIALECTS_ITERATORS_ITERATORSDIALECT
