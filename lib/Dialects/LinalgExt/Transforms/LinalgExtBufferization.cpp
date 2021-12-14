//===-- LinalgExtBufferization.cpp - Linalg Extension bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mlir/IR/BuiltinOps.h>

#include "Dialects/LinalgExt/LinalgExtBufferization.h"
#include "Dialects/LinalgExt/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

using linalg::comprehensive_bufferize::BufferizableOpInterface;
using linalg::comprehensive_bufferize::BufferizationAliasInfo;
using linalg::comprehensive_bufferize::BufferizationState;
using linalg::comprehensive_bufferize::BufferRelation;

namespace linalg_ext {

static SmallVector<OpOperand *> getInsertionDest(InParallelOp inParallelOp) {
  Operation *terminator = inParallelOp.region().front().getTerminator();
  auto performConcOp = dyn_cast<PerformConcurrentlyOp>(terminator);
  assert(performConcOp && "expected PerformConcurrentlyOp as terminator");

  SmallVector<OpOperand *> result;
  performConcOp.walk([&](ParallelInsertSliceOp insertOp) {
    result.push_back(&insertOp->getOpOperand(1) /*dest*/);
  });

  return result;
}

struct InParallelOpInterface
    : public BufferizableOpInterface::ExternalModel<InParallelOpInterface,
                                                    InParallelOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult, BufferizationState &state) const {
    auto inParallelOp = cast<InParallelOp>(op);
    return {getInsertionDest(inParallelOp)[opResult.getResultNumber()]};
  }

  bool mustBufferizeInPlace(
      Operation *op, OpResult opResult, BufferizationState &state) const {
    return true;
  }

  bool isMemoryWrite(
      Operation *op, OpResult opResult, BufferizationState &state) const {
    // TODO: Return true only if there is actually a write inside the region.
    return true;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    auto inParallelOp = cast<InParallelOp>(op);

    // Create new InParallelOp.
    Block *body = &inParallelOp.region().front();
    SmallVector<Value> newResults;
    for (OpResult opResult : inParallelOp->getOpResults()) {
      Value buffer = state.getResultBuffer(opResult);
      newResults.push_back(buffer);

      SmallVector<OpOperand *> insertDestOperands =
          state.getAliasingOpOperand(opResult);
      assert(insertDestOperands.size() == 1 &&
             "expected exactly one aliasing OpOperand");
      Value destTensor = insertDestOperands.front()->get();

      // Replace all uses of the insert dest tensor inside the InParallelOp
      // with the result buffer.
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(body);
      Value toTensorOp = b.create<bufferization::ToTensorOp>(inParallelOp.getLoc(), buffer);
      for (OpOperand &use : destTensor.getUses())
        if (body->findAncestorOpInBlock(*use.getOwner()))
          // This is a use inside the InParallelOp.
          use.set(toTensorOp);
    }
    TypeRange newResultTypes;
    auto newInParallelOp = b.create<InParallelOp>(
        inParallelOp.getLoc(), newResultTypes, inParallelOp.num_threads());

    // Delete terminator.
    newInParallelOp.getBody()->getTerminator()->erase();

    // Move over block contents of the old op.
    IRRewriter rewriter(op->getContext());
    rewriter.mergeBlocks(inParallelOp.getBody(), newInParallelOp.getBody(),
                         {newInParallelOp.getBody()->getArgument(0)});

    // Replace the op.
    state.replaceOp(op, newResults);

    // Bufferize body of the op.
    if (failed(linalg::comprehensive_bufferize::bufferize(
            newInParallelOp.getBody(), state)))
      return failure();

    return success();
  }
};

struct PerformConcurrentlyOpInterface
    : public BufferizableOpInterface::ExternalModel<
          PerformConcurrentlyOpInterface, PerformConcurrentlyOp> {
  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto performConcurrentlyOp = cast<PerformConcurrentlyOp>(op);
    Location nestedTermLoc =
        performConcurrentlyOp.region().front().getTerminator()->getLoc();

    // Create a new PerformConcurrentlyOp.
    auto newOp = state.replaceOpWithNewOp<PerformConcurrentlyOp>(b, op);
    Block &block = newOp.region().emplaceBlock();

    // Create nested terminator.
    b.setInsertionPointToStart(&block);
    b.create<EndPerformConcurrentlyOp>(nestedTermLoc);

    return success();
  }
};

struct ParallelInsertSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ParallelInsertSliceOpInterface, ParallelInsertSliceOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult, BufferizationState &state) const {
    return {&op->getOpOperand(1) /*dest*/};
  }

  OpResult getAliasingOpResult(
      Operation *op, OpOperand &opOperand, BufferizationState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/
               // ParallelInsertSliceOp has not results, attempting to get the
               // OpResult form the parent.
               ? op->getParentOfType<InParallelOp>()->getResult(0)
               : OpResult();
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    llvm_unreachable("op is bufferized as part of InParallelOp");
    return failure();
  }
};
} // namespace linalg_ext
} // namespace mlir

void mlir::linalg_ext::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addOpInterface<InParallelOp, InParallelOpInterface>();
  registry.addOpInterface<PerformConcurrentlyOp, PerformConcurrentlyOpInterface>();
  registry.addOpInterface<linalg_ext::ParallelInsertSliceOp,
                          linalg_ext::ParallelInsertSliceOpInterface>();
}
