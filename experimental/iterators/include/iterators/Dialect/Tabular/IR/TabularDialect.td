//===-- TabularDialect.td - Tabular dialect ----------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TABULAR_DIALECT_TABULAR_IR_TABULARDIALECT
#define TABULAR_DIALECT_TABULAR_IR_TABULARDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

// TODO(ingomueller): This dialect should probably not be lowered to LLVM
//                    directly. I think it would be a better design to treat
//                    tabular views (potentially with a different name) as a
//                    bundle of several tensors, which we can access (and maybe
//                    update) one at a time. It is then up to the consumer to
//                    extract rows or batches of rows from the tensors, for
//                    which they can use the usual means. The default lowering
//                    would then simple decompose the tabular view into its
//                    constituant tensors, which would not affect the accesses
//                    to rows or row batches. After that decomposition, normal
//                    bufferization can run without knowing the tabular
//                    provenance of the tensors.
def Tabular_Dialect : Dialect {
  let name = "tabular";
  let cppNamespace = "::mlir::iterators";
  let summary = "Dialect for dealing with tabular data.";
  let description = [{
    This dialect models data types and ops related to tabular data, i.e.,
    two-dimensional data where different values in the row dimension may have
    mixed types but all values in each column have the same type. It is largely
    orthogonal to the Iterators dialect; the connection consists in the fact
    that one may want to "iterate" over tabular data, i.e., convert tabular data
    to a stream of data. The lowering of the two dialects is currently somewhat
    intervowen but they should ideally be independent of each other.
  }];
  let useDefaultTypePrinterParser = 1;
}

#endif // TABULAR_DIALECT_TABULAR_IR_TABULARDIALECT
