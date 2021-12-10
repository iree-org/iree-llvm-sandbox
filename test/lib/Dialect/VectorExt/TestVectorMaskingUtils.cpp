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
    // Try different testing approaches until one triggers the predication
    // transformation for that particular function.
    bool predicationSucceeded = false;

    // Test tiled loop body predication.
    if (!predicationSucceeded) {
      getFunction().walk([&](TiledLoopOp loopOp) {
        predicationSucceeded = true;
        OpBuilder builder(loopOp);
        if (failed(predicateTiledLoop(builder, loopOp)))
          loopOp.emitError("Predication of tiled loop failed");
      });
    }

    // Test function body predication.
    if (!predicationSucceeded) {
      FuncOp funcOp = getFunction();
      predicationSucceeded = true;

      // Return the mask from the last argument position in the function, if
      // found. Otherwise, return a null value.
      auto createPredicateForFuncOp = [&](OpBuilder &builder) -> Value {
        Region *funcBody = &funcOp.body();
        if (funcBody->args_empty())
          return Value();

        Value mask = funcBody->getArguments().back();
        if (auto vecType = mask.getType().dyn_cast<VectorType>()) {
          Type elemType = vecType.getElementType();
          if (elemType.isInteger(1))
            return mask;
        }

        return Value();
      };

      OpBuilder builder(funcOp);
      if (!predicateOp(builder, funcOp, &funcOp.body(),
                       createPredicateForFuncOp))
        funcOp.emitError("Predication of function failed");
    }
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
