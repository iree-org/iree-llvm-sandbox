//===- DecomposeTupless.cpp - Pass Implementation ----------------*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Tuple/Transforms/DecomposeTuples.h"

#include "iterators/Dialect/Tuple/IR/Tuple.h"
#include "iterators/Dialect/Tuple/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_CLASSES
#include "iterators/Dialect/Tuple/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::iterators;

void iterators::populateDecomposeTuplesPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      // clang-format on
      >(typeConverter, patterns.getContext());
}

struct DecomposeTuplesPass : public DecomposeTuplesBase<DecomposeTuplesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
  };
};

std::unique_ptr<Pass> mlir::createDecomposeTuplesPass() {
  return std::make_unique<DecomposeTuplesPass>();
}
