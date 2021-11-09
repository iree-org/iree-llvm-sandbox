//===-- LinalgExtBase.td - Linalg Extension dialect --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUNNERS_LINALGEXT_LINALGEXTBASE
#define RUNNERS_LINALGEXT_LINALGEXTBASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def LinalgExt_Dialect : Dialect {
  let name = "linalg_ext";
  let cppNamespace = "::mlir::linalg_ext";
  let description = [{
    The `linalg_ext` dialect is intended to experiment more support for
    non-structured operations, ie, can not be represented in Linalg operations.
  }];
}


#endif // RUNNERS_LINALGEXT_LINALGEXTBASE
