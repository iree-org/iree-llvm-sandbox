//===-- dialect_extension.h - TD extension for Jasc -------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef JASC_TRANSFORM_OPS_DIALECT_EXTENSION_H_
#define JASC_TRANSFORM_OPS_DIALECT_EXTENSION_H_

#include "mlir/IR/DialectRegistry.h"

namespace jasc {
void registerTransformDialectExtension(mlir::DialectRegistry &registry);
}

#endif  // JASC_TRANSFORM_OPS_DIALECT_EXTENSION_H_
