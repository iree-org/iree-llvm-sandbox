//===- FuseFillIntoReduction.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transforms/PassDetail.h"
#include "Transforms/Passes.h"
#include "Transforms/Transforms.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"

namespace mlir {
namespace linalg {
namespace {

using llvm::makeArrayRef;
using mlir::tensor::ExtractSliceOp;
using mlir::tensor::InsertSliceOp;

SmallVector<OpFoldResult> GetParallelDimStep(scf::LoopNest nest,
                                             GenericOp tiled_op) {
  assert(nest.loops.size() == 2 && "Expected a 2D loop");
  assert(tiled_op.getNumLoops() && "Expected a 2D tiled op");
  SmallVector<unsigned> parallelDims;
  tiled_op.getParallelDims(parallelDims);
  assert(parallelDims.size() == 1 && "Should be only 1 parallel loop");
  Value step = parallelDims.front() == 1 ? nest.loops.back().getStep()
                                         : nest.loops.front().getStep();
  if (auto constant = step.getDefiningOp<mlir::arith::ConstantOp>()) {
    return {constant.getValue()};
  }
  return {step};
}

mlir::FailureOr<Operation *> DetectCombiner(LinalgOp linalg_op) {
  mlir::SmallVector<Operation *, 4> combiners;
  if (!matchReduction(linalg_op.getRegionOutputArgs(), 0, combiners) ||
      combiners.size() != 1)
    return mlir::failure();
  return combiners.front();
}

// Fuses `linalg.fill` into a loop with a tiled reduction.
// Currently, only 2D case is supported. Fusion into a tiled 1D reduction is
// also possible.
struct FuseFillIntoTiledReductionPattern : public OpRewritePattern<GenericOp> {
  explicit FuseFillIntoTiledReductionPattern(MLIRContext *context,
                                             mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit) {}

  LogicalResult matchAndRewrite(GenericOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (linalg_op.getNumOutputs() != 1)
      return failure();
    if (linalg_op.getNumLoops() != 2)
      return failure();

    // Get immediate parent.
    auto inner_loop =
        dyn_cast<scf::ForOp>(linalg_op->getParentRegion()->getParentOp());
    if (!inner_loop)
      return failure();
    auto outer_loop =
        dyn_cast<scf::ForOp>(inner_loop->getParentRegion()->getParentOp());
    if (!outer_loop)
      return failure();
    scf::LoopNest nest{{outer_loop, inner_loop}};

    return RewriteTiledReduction(rewriter, nest, linalg_op);
  }

private:
  // Add a new output argument to the `scf.for` nest. It will be produced by
  // `init_tensor` op with the same shape of the tiled output argument.
  //
  // Rewrite
  //
  //   %init = linalg.init_tensor
  //   %fill = linalg.fill(%cst, %init)
  //   scf.for iter_args(%fill)
  //
  // into
  //
  //   %init = linalg.init_tensor
  //** %init_tile = linalg.init_tensor [%stride]
  //   %fill = linalg.fill(%cst, %init)
  //** scf.for iter_args(%fill, %init_tile)
  BlockArgument CloneAndAppendInitTensorToTiledLoop(PatternRewriter &rewriter,
                                                    FillOp fill,
                                                    scf::LoopNest nest,
                                                    GenericOp tiled_op) const {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(fill);

    auto fillType = fill.output().getType();

    auto loc = fill.getLoc();
    auto inner = nest.loops.back();
    auto outer = nest.loops.front();
    Value init_clone = rewriter.create<InitTensorOp>(
        fill.getLoc(), GetParallelDimStep(nest, tiled_op),
        fillType.cast<mlir::RankedTensorType>().getElementType());
    rewriter.updateRootInPlace(outer, [&]() {
      outer.getInitArgsMutable().append(init_clone);
      outer.getBody()->addArgument(init_clone.getType(), loc);
    });
    rewriter.updateRootInPlace(outer, [&]() {
      auto bbArg = outer.getBody()->getArguments().back();
      inner.getInitArgsMutable().append(bbArg);
      inner.getBody()->addArgument(bbArg.getType(), loc);
    });
    return inner.getBody()->getArguments().back();
  }

  // Fuse `fill` operation into the `scf.for`, rewire the `linalg.generic` to
  // use it as the output for the reduced tile. Also create an additional
  // `insert_slice` that updates the new output.
  //
  // Rewrite
  //
  // %init = linalg.init_tensor
  // %init_tile = linalg.init_tensor [%stride]
  // %fill = linalg.fill(%cst, %init)
  // scf.for iter_args(%fill, %init_tile) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //   %reduce = linalg.generic outs (%extract_output_slice)
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice
  // }
  //
  // into
  //
  // %init = linalg.init_tensor
  // %init_tile = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // scf.for iter_args(%fill, %init_tile) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //** %slice_of_output_tile = tensor.extract_slice %init
  //** %fill_of_output_tile = linalg.fill(%cst, %slice_of_output_tile)
  //** %reduce = linalg.generic outs (%fill_of_output_tile)
  //** %update_output_tile = tensor.insert_slice %reduce into %init_tile
  //
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice, %update_output_tile
  // }
  void FuseFill(PatternRewriter &rewriter, LinalgOp tiled_op, FillOp fill,
                BlockArgument loop_output_bb_arg,
                BlockArgument output_tile_bb_arg,
                ExtractSliceOp extract_output_slice,
                InsertSliceOp insert_output_slice) const {
    Location loc = tiled_op.getLoc();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(tiled_op);

    SmallVector<OpFoldResult> offset{rewriter.getIndexAttr(0)};
    Value slice_of_output_tile = rewriter.create<ExtractSliceOp>(
        loc, output_tile_bb_arg, offset, extract_output_slice.getMixedSizes(),
        extract_output_slice.getMixedStrides());

    auto fused_fill = rewriter.create<FillOp>(loc, ValueRange{fill.value()},
                                              ValueRange{slice_of_output_tile});
    rewriter.updateRootInPlace(tiled_op, [&]() {
      tiled_op.getOutputOperand(0)->set(fused_fill.result());
    });

    rewriter.setInsertionPointAfter(tiled_op);
    Value cloned_insert = rewriter.create<mlir::tensor::InsertSliceOp>(
        loc, fused_fill.getResult(0), output_tile_bb_arg, offset,
        extract_output_slice.getMixedSizes(),
        extract_output_slice.getMixedStrides());

    auto yield = tiled_op.getOperation()->getBlock()->getTerminator();
    rewriter.updateRootInPlace(
        yield, [&]() { yield->insertOperands(1, cloned_insert); });
  }

  // Add an operation that combines the partial result with the output.
  //
  // Rewrite
  //
  // %init = linalg.init_tensor
  // %init_tile = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // scf.for iter_args(%fill, %init_tile) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //   %slice_of_output_tile = tensor.extract_slice %init
  //   %fill_of_output_tile = linalg.fill(%cst, %slice_of_output_tile)
  //   %reduce = linalg.generic outs (%fill_of_output_tile)
  //   %update_output_tile = tensor.insert_slice %reduce into %init_tile
  //
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice, %update_output_tile
  // }
  //
  // into
  //
  // %init = linalg.init_tensor
  // %init_tile = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // scf.for iter_args(%fill, %init_tile) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //   %slice_of_output_tile = tensor.extract_slice %init
  //   %fill_of_output_tile = linalg.fill(%cst, %slice_of_output_tile)
  //   %reduce = linalg.generic outs (%fill_of_output_tile)
  //   %update_output_tile = tensor.insert_slice %reduce into %init_tile
  //
  //** %combine = linalg.generic ins (%reduce) outs (%extract_output_slice)
  //** %insert_output_slice = tensor.insert_slice %combine into %fill
  //
  //   linalg.yield %insert_output_slice, %update_output_tile
  // }
  LogicalResult
  CombineReducedTileWithOutput(PatternRewriter &rewriter, GenericOp tiled_op,
                               Value partial_result,
                               ExtractSliceOp extract_output_slice,
                               InsertSliceOp insert_output_slice) const {
    rewriter.setInsertionPointAfter(tiled_op);
    auto num_parallel_loops = tiled_op.getNumParallelLoops();
    SmallVector<mlir::StringRef, 3> parallel_iter_types(
        num_parallel_loops, mlir::getParallelIteratorTypeName());
    auto id_map = rewriter.getMultiDimIdentityMap(num_parallel_loops);

    auto combiner_or = DetectCombiner(tiled_op);
    if (failed(combiner_or))
      return failure();
    Operation *combiner = combiner_or.getValue();

    auto accumulator = rewriter.create<GenericOp>(
        tiled_op.getLoc(), partial_result.getType(),
        makeArrayRef(partial_result),
        makeArrayRef(extract_output_slice.result()),
        makeArrayRef({id_map, id_map}), parallel_iter_types,
        [&](OpBuilder &b, Location nested_loc, ValueRange args) {
          BlockAndValueMapping bvm;
          bvm.map(combiner->getOperands(), args);
          Value result_val = b.clone(*combiner, bvm)->getResult(0);
          b.create<YieldOp>(nested_loc, result_val);
        });

    rewriter.updateRootInPlace(insert_output_slice, [&]() {
      insert_output_slice.sourceMutable().assign(accumulator.getResult(0));
    });
    return success();
  }

  // Unfortunaly, there is no way to modify the results of the loop inplace. So
  // we have to replace it with a clone.
  void CreateLoopWithUpdatedResults(PatternRewriter &rewriter,
                                    scf::LoopNest nest) const {
    scf::ForOp inner = nest.loops.back();
    scf::ForOp outer = nest.loops.front();
    auto loc = outer.getLoc();
    rewriter.setInsertionPoint(outer);
    SmallVector<Value, 2> lbs{outer.getLowerBound(), inner.getLowerBound()};
    SmallVector<Value, 2> ubs{outer.getUpperBound(), inner.getUpperBound()};
    SmallVector<Value, 2> steps{outer.getStep(), inner.getStep()};
    SmallVector<Value, 2> iters{outer.getInitArgs().front(),
                                outer.getInitArgs().back()};
    auto new_nest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, iters,
        [&](mlir::OpBuilder &b, mlir::Location location, ValueRange ivs,
            ValueRange iter_args) -> scf::ValueVector {
          BlockAndValueMapping bvm;
          bvm.map(outer.getInductionVar(), ivs[0]);
          bvm.map(inner.getInductionVar(), ivs[1]);
          bvm.map(inner.getRegionIterArgs(), iter_args);
          for (auto &op : inner.getBody()->without_terminator())
            b.clone(op, bvm);
          auto results = inner.getBody()->getTerminator()->getOperands();
          return scf::ValueVector{bvm.lookup(results.front()),
                                  bvm.lookup(results.back())};
        }

    );
    rewriter.replaceOp(outer, new_nest.getResults()[0]);
  }

  // Fuses FillOp producer of the output argument of the TiledLoopOp and inserts
  // an operation that accumulates the partial result, i.e. reduced tile, and
  // the current value of the output tile.
  LogicalResult RewriteTiledReduction(PatternRewriter &rewriter,
                                      scf::LoopNest nest,
                                      GenericOp tiled_op) const {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(tiled_op);

    // Find scf.for output operand and the corresponding block argument.
    mlir::OpOperand &loop_output_operand =
        nest.loops.front().getIterOpOperands().front();
    BlockArgument &loop_output_bb_arg =
        nest.loops.back().getRegionIterArgs().back();

    // Find `linalg.fill` producer of the output.
    auto fill = loop_output_operand.get().getDefiningOp<FillOp>();
    if (!fill)
      return failure();

    // Find extract_slice/insert_slice pair used to RMW output.
    auto extract_output_slice =
        tiled_op.getOutputOperand(0)->get().getDefiningOp<ExtractSliceOp>();
    if (!extract_output_slice)
      return failure();

    Value tiled_op_result = tiled_op->getResult(0);
    auto insert_output_slice =
        dyn_cast<InsertSliceOp>(*tiled_op_result.getUsers().begin());
    if (!insert_output_slice)
      return failure();

    // Fuse the output.
    BlockArgument output_tile_bb_arg =
        CloneAndAppendInitTensorToTiledLoop(rewriter, fill, nest, tiled_op);
    FuseFill(rewriter, tiled_op, fill, loop_output_bb_arg, output_tile_bb_arg,
             extract_output_slice, insert_output_slice);
    // We have already modified the loop above, so we need to update the
    // results.
    if (failed(CombineReducedTileWithOutput(rewriter, tiled_op, tiled_op_result,
                                            extract_output_slice,
                                            insert_output_slice)))
      return failure();
    CreateLoopWithUpdatedResults(rewriter, nest);
    return success();
  }
};

} // namespace

void populateFuseFillIntoReductionPatterns(RewritePatternSet &patterns) {
  patterns.insert<FuseFillIntoTiledReductionPattern>(patterns.getContext());
}

} // namespace linalg
} // namespace mlir
