//===-- ops.cc - Jasc op implementations ------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ops.h"

#include "llvm/include/llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#define GET_OP_CLASSES
#include "dialect/ops.cc.inc"

namespace jasc {


}  // namespace jasc
