//===- DecomposeIteratorStates.cpp - Pass Implementation --------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/Transforms/DecomposeIteratorStates.h"

#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Dialect/Iterators/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_CLASSES
#include "iterators/Dialect/Iterators/Transforms/Passes.h.inc"
} // namespace mlir

// using namespace iterators;
using namespace mlir;
using namespace mlir::iterators;

void iterators::populateDecomposeIteratorStatesPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      // clang-format on
      >(typeConverter, patterns.getContext());
}

struct DecomposeIteratorStatesPass
    : public DecomposeIteratorStatesBase<DecomposeIteratorStatesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
  };
};

std::unique_ptr<Pass> mlir::createDecomposeIteratorStatesPass() {
  return std::make_unique<DecomposeIteratorStatesPass>();
}
