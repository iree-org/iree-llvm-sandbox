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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::vector;
using namespace mlir::vector_ext;

namespace {

struct TestVectorMaskingUtils
    : public PassWrapper<TestVectorMaskingUtils, OperationPass<FuncOp>> {

  TestVectorMaskingUtils() = default;
  TestVectorMaskingUtils(const TestVectorMaskingUtils &pass) {}

  StringRef getArgument() const final { return "test-vector-masking-utils"; }
  StringRef getDescription() const final {
    return "Test vector masking utilities";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LinalgDialect, VectorDialect, VectorExtDialect>();
  }

  Option<bool> predicationEnabled{*this, "predication",
                                  llvm::cl::desc("Test vector predication"),
                                  llvm::cl::init(false)};

  Option<bool> maskingEnabled{*this, "masking",
                              llvm::cl::desc("Test vector masking"),
                              llvm::cl::init(false)};

  void testPredication() {
    // Try different testing approaches until one triggers the predication
    // transformation for that particular function.
    bool predicationSucceeded = false;

    // Test tiled loop body predication.
    if (!predicationSucceeded) {
      getOperation().walk([&](TiledLoopOp loopOp) {
        predicationSucceeded = true;
        OpBuilder builder(loopOp);
        if (failed(predicateTiledLoop(builder, loopOp,
                                      /*incomingMask=*/llvm::None)))
          loopOp.emitError("Predication of tiled loop failed");
      });
    }

    // Test function body predication.
    if (!predicationSucceeded) {
      FuncOp funcOp = getOperation();
      ValueRange funcArgs = funcOp.body().getArguments();

      if (funcArgs.size() >= 3) {
        predicationSucceeded = true;

        // Return the mask from the third argument position starting from the
        // end, if found. Otherwise, return a null value.
        auto createPredicateMaskForFuncOp = [&](OpBuilder &builder) -> Value {
          if (funcArgs.size() < 3)
            return Value();

          // Predicate mask is the third argument starting from the end.
          Value mask = *std::prev(funcArgs.end(), 3);
          if (auto vecType = mask.getType().dyn_cast<VectorType>()) {
            Type elemType = vecType.getElementType();
            if (elemType.isInteger(1))
              return mask;
          }

          return Value();
        };

        OpBuilder builder(funcOp);
        Value idx = *std::prev(funcArgs.end(), 2);
        Value incoming = funcArgs.back();
        if (!predicateOp(builder, funcOp, &funcOp.body(),
                         createPredicateMaskForFuncOp, idx, incoming))
          funcOp.emitRemark("Predication of function failed");
      }
    }
  }

  void testMasking() {
    FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp);
    if (failed(maskVectorPredicateOps(builder, funcOp,
                                      maskGenericOpWithSideEffects)))
      funcOp.emitError("Masking of function failed");
  }

  void runOnOperation() override {
    if (predicationEnabled)
      testPredication();
    if (maskingEnabled)
      testMasking();
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
