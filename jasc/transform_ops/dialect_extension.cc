//===-- dialect_extension.cc - TD extension for Jasc ------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dialect_extension.h"

#include "jasc_transform_ops.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/DialectRegistry.h"

namespace {
class JascTransformDialectExtension
    : public mlir::transform::TransformDialectExtension<
          JascTransformDialectExtension> {
 public:
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "jasc_transform_ops.cpp.inc"
        >();
  }
};
}  // namespace

void jasc::registerTransformDialectExtension(mlir::DialectRegistry &registry) {
  registry.addExtensions<JascTransformDialectExtension>();
}
