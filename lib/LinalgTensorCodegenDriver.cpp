//===- LinalgTensorCodegenDriver.cpp - Linalg transformation driver--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Passes/PassDetail.h"
#include "Passes/Passes.h"

#include "Passes/Transforms.h"

#include "Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "Dialect/LinalgExt/LinalgExtBufferization.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

struct UnrollOneVectorOpPass
    : public UnrollOneVectorOpBase<UnrollOneVectorOpPass> {
  UnrollOneVectorOpPass() = default;
  UnrollOneVectorOpPass(const UnrollOneVectorOpPass &pass) {}
  void runOnOperation() override;
};

} // namespace

void UnrollOneVectorOpPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName)
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShape(targetShape)
                    .setFilterConstraint([&](Operation *op) {
                      auto unrollInterface =
                          dyn_cast<VectorUnrollOpInterface>(op);
                      if (!unrollInterface ||
                          op->getName().getStringRef() != anchorOpName ||
                          !sourceShape.hasValue() ||
                          !unrollInterface.getShapeForUnroll().hasValue())
                        return failure();

                      ArrayRef<int64_t> sourceShapeToMatch{sourceShape};
                      auto shapeForUnroll =
                          unrollInterface.getShapeForUnroll().getValue();
                      ArrayRef<int64_t> actualSourceShape{
                          shapeForUnroll.begin(), shapeForUnroll.end()};
                      return success(sourceShapeToMatch == actualSourceShape);
                    }));
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

//===----------------------------------------------------------------------===//
// Pass creation entry points.
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createUnrollOneVectorOpPass() {
  return std::make_unique<UnrollOneVectorOpPass>();
}
