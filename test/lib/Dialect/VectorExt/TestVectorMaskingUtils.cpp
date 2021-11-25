//===- TestMaskingUtils.cpp - Utilities for vector masking ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Vector masking utilities.
//
//===----------------------------------------------------------------------===//

#include "Dialects/VectorExt/VectorExtOps.h"
#include "Dialects/VectorExt/VectorMaskingUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::vector;
using namespace mlir::vector_ext;

namespace {

struct TestVectorMaskingUtils
    : public PassWrapper<TestVectorMaskingUtils, FunctionPass> {
  StringRef getArgument() const final { return "test-vector-masking-utils"; }
  StringRef getDescription() const final {
    return "Test vector masking utilities";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LinalgDialect, VectorDialect, VectorExtDialect>();
  }

  void runOnFunction() override {
    getFunction().walk([](TiledLoopOp loopOp) {
      if (failed(predicateTiledLoop(loopOp)))
        loopOp.emitError("Predication of tiled loop failed");
    });
  }
};

} // namespace

namespace mlir {
namespace test_ext {
void registerTestVectorMaskingUtils() {
  PassRegistration<TestVectorMaskingUtils>();
}
} // namespace test_ext
} // namespace mlir
