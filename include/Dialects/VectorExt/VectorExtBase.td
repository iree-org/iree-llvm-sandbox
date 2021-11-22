//===-- VectorExtBase.td - Vector Extension dialect --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUNNERS_VECTOREXT_VECTOREXTBASE
#define RUNNERS_VECTOREXT_VECTOREXTBASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def VectorExt_Dialect : Dialect {
  let name = "vector_ext";
  let cppNamespace = "::mlir::vector_ext";
  let description = [{
    The `vector_ext` dialect is intended to experiment with new vector operations
    that could eventually be added to the Vector dialect.
  }];
}

#endif // RUNNERS_VECTOREXT_VECTOREXTBASE
