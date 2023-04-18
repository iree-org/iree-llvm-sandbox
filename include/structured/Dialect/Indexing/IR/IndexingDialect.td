//===-- IndexingDialect.td - Indexing dialect ------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_DIALECT_INDEXING_IR_INDEXINGDIALECT
#define STRUCTURED_DIALECT_INDEXING_IR_INDEXINGDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Indexing_Dialect
: Dialect {
  let name = "indexing";
  let cppNamespace = "::mlir::indexing";
  let useDefaultTypePrinterParser = 1;
}

#endif // STRUCTURED_DIALECT_INDEXING_IR_INDEXINGDIALECT
