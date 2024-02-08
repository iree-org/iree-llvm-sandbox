//===-- jasc_transform_ops.h - Transform ops for Jasc dialect ---*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef JASC_TRANSFORMOPS_JASCTRANSFORMOPS_H_
#define JASC_TRANSFORMOPS_JASCTRANSFORMOPS_H_

#include <optional>

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "jasc_transform_ops.h.inc"

#endif  // JASC_TRANSFORMOPS_JASCTRANSFORMOPS_H_
