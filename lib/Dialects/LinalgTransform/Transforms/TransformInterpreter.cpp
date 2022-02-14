//===-- TransformInterpreter.cpp - Interpreter of Linalg transforms -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files implements and interpreter pass for Linalg Transform dialect. This
// pass reads a module consisting of transformable IR and transformation control
// IR and applies the latter to the former.
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/LinalgTransformOps.h"
#include "Dialects/LinalgTransform/Passes.h"
#include "Dialects/LinalgTransform/ScopedTransform.h"
#include "Dialects/LinalgTransform/SimplePatternRewriter.h"
#include "Dialects/LinalgTransform/TrackingCSE.h"
#include "Dialects/LinalgTransform/TrackingRewriteDriver.h"
#include "Dialects/LinalgTransform/TransformOpMapping.h"
#include "FunctionHelpers.h"
#include "PDL.h"
#include "TrackingListener.h"
#include "Transforms/Functional.h"
#include "Transforms/Listener.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "transform-interpreter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Extracts a vector of int64_t from an array attribute. Asserts if the
/// attribute contains values other than integers.
static SmallVector<int64_t> extractI64Array(ArrayAttr attr) {
  SmallVector<int64_t> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getSExtValue());
  return result;
}

/// Extracts a vector of unsigned from an array attribute. Asserts if the
/// attribute contains values other than intergers. May truncate.
static SmallVector<unsigned> extractUIntArray(ArrayAttr attr) {
  SmallVector<unsigned> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getZExtValue());
  return result;
}

/// Returns the neutral value for a Linalg operation that produces the given
/// operand, construct using the provided builder. Currently assumes the
/// reduction in the Linalg operation is an addition and, therefore, the neutral
/// value is zero.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get().getType());
  return b.create<arith::ConstantOp>(op.getOwner()->getLoc(), t,
                                     b.getZeroAttr(t));
}

//===----------------------------------------------------------------------===//
// Functional Rewrite Helpers
//===----------------------------------------------------------------------===//

using FunctionalLinalgTransform =
    std::function<FailureOr<LinalgOp>(LinalgOp, PatternRewriter &)>;

/// Fallback "pattern" for simply forwarding a result when an interpreter op is
/// a no-op.
static FailureOr<LinalgOp> forwardOp(LinalgOp op, PatternRewriter &rewriter) {
  return op;
}

//===----------------------------------------------------------------------===//
// Linalg Transforms
//===----------------------------------------------------------------------===//

/// Find all ops in `module` that match the PDL pattern specified by the MatchOp
/// and store them in the operation map.
static LogicalResult executeMatchOp(transform::MatchOp op, ModuleOp module,
                                    TransformOpMapping &operations) {
  FailureOr<SmallVector<Operation *>> ops = findMatchingOps(op, module);
  if (failed(ops))
    return failure();
  LLVM_DEBUG(DBGS() << "matched " << ops->size() << " ops\n");
  operations.try_emplace(op.target(), std::move(*ops));
  return success();
}

/// Applies the pad pattern to the given target operation as indicated by the
/// tile op that subsumes padding. Populates `nextTargets` with transformable
/// operations for further transformations (currently, the single padded op).
static FunctionalLinalgTransform
buildPadFromTileOpPattern(linalg::transform::TileOp tileOp) {
  if (!tileOp.pad())
    return forwardOp;

  // Capture `tileOp` by-copy because it lives on the stack of the current
  // function but lambdas outlive it. They are marked as mutable because op
  // accessors are non-const.
  auto packFunc = [tileOp](OpOperand &opOperand) mutable {
    return opOperand.getOperandNumber() < tileOp.pack_paddings().size()
               ? !tileOp.pack_paddings()[opOperand.getOperandNumber()]
                      .cast<IntegerAttr>()
                      .getValue()
                      .isZero()
               : false;
  };
  auto hoistingFunc = [tileOp](OpOperand &opOperand) mutable {
    return opOperand.getOperandNumber() < tileOp.hoist_paddings().size()
               ? tileOp.hoist_paddings()[opOperand.getOperandNumber()]
                     .cast<IntegerAttr>()
                     .getValue()
                     .getSExtValue()
               : 0;
  };
  auto transposeFunc = [tileOp](OpOperand &opOperand) mutable {
    if (opOperand.getOperandNumber() >= tileOp.transpose_paddings().size())
      return SmallVector<int64_t>();

    auto transposePaddings =
        tileOp.transpose_paddings()[opOperand.getOperandNumber()]
            .cast<ArrayAttr>();
    return extractI64Array(transposePaddings);
  };
  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  paddingOptions.setPaddingNoFoldComputationFunction(packFunc);
  paddingOptions.setPaddingHoistComputationFunction(hoistingFunc);
  paddingOptions.setPaddingTransposeComputationFunction(transposeFunc);

  return callLinalgPattern<LinalgPaddingPattern>(tileOp.getContext(),
                                                 paddingOptions);
}

/// Applies the generalization pattern to the given target operation as
/// indicated by the tile op that subsumes padding. Populates `nextTargets` with
/// transformable operations for further transformations (currently, the single
/// generalized op).
static FunctionalLinalgTransform
buildGeneralizeFromTileOpPattern(linalg::transform::TileOp tileOp) {
  if (!tileOp.generalize())
    return forwardOp;
  return callLinalgPattern<LinalgGeneralizationPattern>(tileOp.getContext());
}

/// Applies the transformation specified by the given tile operation to the
/// given target operation. Populates `results` with transformation operations
/// for further transformations if the pattern applied successfully (currently,
/// the single op after tiling).
// TODO: if the tiling pattern failed, we still want to populate results with
// something.
static FailureOr<LinalgOp> executeTileOp(LinalgOp target,
                                         linalg::transform::TileOp tileOp) {
  LinalgTilingOptions tilingOptions;
  SmallVector<int64_t> tileSizes = extractI64Array(tileOp.sizes());
  // "scalarize_dyn_dims" actually sets the same lambda as the tile sizes and
  // asserts that it is not already set.
  if (!tileSizes.empty() || !tileOp.scalarize_dyn_dims())
    tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setInterchange(extractUIntArray(tileOp.interchange()));
  tilingOptions.setPeeledLoops(extractI64Array(tileOp.peel()));
  if (tileOp.scalarize_dyn_dims())
    tilingOptions.scalarizeDynamicDims();

  LinalgTilingPattern pattern(tileOp.getContext(), tilingOptions);
  auto functionalTile = [&](LinalgOp op,
                            PatternRewriter &rewriter) -> FailureOr<LinalgOp> {
    auto result = pattern.returningMatchAndRewrite(op, rewriter);
    if (failed(result))
      return failure();
    return result->op;
  };

  auto tileSeq = functional::SequenceBuilder()
                     .begin(std::move(functionalTile))
                     .then(buildPadFromTileOpPattern(tileOp))
                     .then(buildGeneralizeFromTileOpPattern(tileOp));

  return functional::applyAt(target, tileSeq);
}

/// Applies the transformation specified by the given decompose operation to the
/// given target operation.
static LogicalResult
executeDecomposeOp(ModuleOp module,
                   linalg::transform::DecomposeOp decomposeOp) {
  MLIRContext *ctx = module->getContext();
  RewritePatternSet patterns(ctx);
  // TODO: make this targetable.
  populateDecomposeConvolutionPatterns(patterns, LinalgTransformationFilter());
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return failure();

  // TODO: make this chainable, it isn't in the original codegenstrategy.
  return success();
}

/// Applies the transformation specified by the given vectorize operation to the
/// given target operation AND some related operations.Populates `results` with
/// transformation operations for further transformations if the pattern applied
/// successfully (currently, the main "contraction" op after vectorization).
static FailureOr<LinalgOp>
executeVectorizeOp(LinalgOp target,
                   linalg::transform::VectorizeOp vectorizeOp) {
  // TODO: this is copy-pasta from LinalgStrategyVectorizePass, it shouldn't be.
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                       /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  if (vectorizeOp.vectorize_padding())
    linalg::populatePadOpVectorizationPatterns(patterns);
  LinalgVectorizationPattern pattern(vectorizeOp.getContext());
  auto functionalVectorize = [&](LinalgOp op, PatternRewriter &rewriter) {
    return pattern.matchAndRewrite(op, rewriter);
  };

  /// Apply the transformations in a scope.
  return transform::scoped(
      target,
      [&](transform::ScopeOp scope, Operation *op) -> FailureOr<LinalgOp> {
        if (failed(functional::applyAt(op, functionalVectorize)) ||
            failed(applyPatternsAndFoldGreedily(scope, std::move(patterns))))
          return failure();
        // FIXME: Vectorization doesn't return anything.
        return LinalgOp();
      });

  // TODO: vectorization may fail because the op is not vectorizable, unclear
  // what to do here. We should probably report it somehow, but we may also
  // want to go on and keep the original for continuation. Should we have
  // some notion of transformation optionality vs. mandatory (like lowering)?
  // How to find ops that were not replaced?
}

/// Returns true of the numbered vector lowering stage is included into the list
/// of stages specified on the given lowerVectors operation.
static bool stageIncluded(int stage, transform::LowerVectorsOp lowerVectorsOp) {
  for (auto s : lowerVectorsOp.stages().getAsValueRange<IntegerAttr>()) {
    if (s.getSExtValue() == stage)
      return true;
  }
  return false;
}

/// Applies the transformation specified by the given lower vectors operation
/// to the given function.
static LogicalResult
executeLowerVectorsOp(ModuleOp module,
                      linalg::transform::LowerVectorsOp lowerVectorsOp) {
  MLIRContext *ctx = module->getContext();
  RewritePatternSet patterns(ctx);

  vector::VectorTransposeLowering vectorTransposeLowering =
      llvm::StringSwitch<vector::VectorTransposeLowering>(
          lowerVectorsOp.transpose_lowering())
          .Case("eltwise", vector::VectorTransposeLowering::EltWise)
          .Case("flat_transpose", vector::VectorTransposeLowering::Flat)
          .Case("shuffle", vector::VectorTransposeLowering::Shuffle)
          .Default(vector::VectorTransposeLowering::EltWise);
  vector::VectorMultiReductionLowering vectorMultiReductionLowering =
      llvm::StringSwitch<vector::VectorMultiReductionLowering>(
          lowerVectorsOp.multireduction_lowering())
          .Case("innerreduction",
                vector::VectorMultiReductionLowering::InnerReduction)
          .Default(vector::VectorMultiReductionLowering::InnerParallel);
  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          lowerVectorsOp.contraction_lowering())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  // TODO: fix the annoying name mismatch (vector-transfers vs vector-transfer).
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(lowerVectorsOp.split_transfers())
      .Case("none", vector::VectorTransferSplit::None)
      .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
      .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
      .Default(vector::VectorTransferSplit::None);

  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransformsOptions(vectorContractLowering)
      .setVectorMultiReductionLowering(vectorMultiReductionLowering)
      .setVectorTransposeLowering(vectorTransposeLowering)
      .setVectorTransferSplit(vectorTransferSplit);

  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions()
          .enableFullUnroll(lowerVectorsOp.unroll_vector_transfers())
          .enableLowerPermutationMaps();

  int maxTransferRank = 1;

  auto avx2LoweringOptions =
      x86vector::avx2::LoweringOptions().setTransposeOptions(
          x86vector::avx2::TransposeLoweringOptions()
              .lower4x8xf32(lowerVectorsOp.transpose_avx2_lowering())
              .lower8x8xf32(lowerVectorsOp.transpose_avx2_lowering()));

  // TODO: this is copy-pasta from LinalgStrategyLowerVectorsPass, shouldn't be.
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  if (stageIncluded(1, lowerVectorsOp)) {
    patterns.add<mlir::vector::ContractionOpToOuterProductOpLowering,
                 mlir::vector::ContractionOpToMatmulOpLowering,
                 mlir::vector::ContractionOpLowering>(vectorTransformOptions,
                                                      ctx);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  }
  if (stageIncluded(2, lowerVectorsOp)) {
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vectorTransformOptions.vectorMultiReductionLowering);
  }
  if (stageIncluded(3, lowerVectorsOp)) {
    patterns.add<vector::VectorTransferFullPartialRewriter>(
        ctx, vectorTransformOptions);
  }
  if (stageIncluded(4, lowerVectorsOp)) {
    vector::populateVectorTransferLoweringPatterns(patterns, maxTransferRank);
  }
  if (stageIncluded(5, lowerVectorsOp)) {
    populateVectorToSCFConversionPatterns(
        patterns, vectorTransferToSCFOptions.setTargetRank(maxTransferRank));
  }
  if (stageIncluded(6, lowerVectorsOp)) {
    vector::populateVectorShapeCastLoweringPatterns(patterns);
  }
  if (stageIncluded(7, lowerVectorsOp)) {
    vector::populateVectorTransposeLoweringPatterns(patterns,
                                                    vectorTransformOptions);
    if (lowerVectorsOp.transpose_avx2_lowering())
      x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
          patterns, avx2LoweringOptions, /*benefit=*/10);
  }

  // TODO: these transformations are currently not targeted at concrete ops.
  // LinalgTransformationFilter filter = makeTransformationFilter(target);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return failure();

  // TODO: make composable...
  return success();
}

/// Applies the transformation specified by the given bufferize operation to
/// the module containing the given function.
static LogicalResult
executeBufferizeOp(ModuleOp module,
                   linalg::transform::BufferizeOp bufferizeOp) {
  PassManager pm(module->getContext());

  pm.addPass(createLinalgComprehensiveModuleBufferizePass());
  if (failed(pm.run(module)))
    return failure();

  // Perform buffer-level hoistings.
  module.walk([&](FuncOp funcOp) { hoistRedundantVectorTransfers(funcOp); });
  return success();
}

/// Applies the transformation specified by the given Lower to LLVM operation
/// to the module containing the given function.
static LogicalResult
executeLowerToLLVMOp(ModuleOp module,
                     linalg::transform::LowerToLLVMOp lowerToLLVMOp) {
  // TODO: it is feasible to scope lowering at arbitrary level and introduce
  // unrealized casts, but there needs to be the final module-wise cleanup in
  // the end. Keep module-level for now.
  PassManager pm(module->getContext());

  pm.addNestedPass<FuncOp>(createConvertVectorToSCFPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertLinalgToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass(
      // clang-format off
      LowerVectorToLLVMOptions()
        .enableReassociateFPReductions(false)
        .enableIndexOptimizations(false)
        .enableArmNeon(false)
        .enableArmSVE(false)
        .enableAMX(false)
        .enableX86Vector(false)));
  // clang-format on
  pm.addNestedPass<FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createMemRefToLLVMPass());
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  return pm.run(module);
}

static FailureOr<scf::ForOp>
executeGetParentLoopOp(Operation *source,
                       linalg::transform::GetParentLoopOp getParentLoopOp) {
  int64_t nLoops = getParentLoopOp.num_loops();
  for (int64_t i = 0; i < nLoops; ++i) {
    source = source->getParentOfType<scf::ForOp>();
    if (!source) {
      getParentLoopOp.emitError() << "the transformed op is enclosed by " << i
                                  << " loops, but " << nLoops << " expected";
      return failure();
    }
  }
  return cast<scf::ForOp>(source);
}

static LogicalResult
executeUnrollLoopOp(scf::ForOp loop,
                    linalg::transform::UnrollLoopOp unrollLoopOp) {
  return loopUnrollByFactor(loop, unrollLoopOp.factor());
}

static void
loopScheduling(scf::ForOp forOp,
               std::vector<std::pair<Operation *, unsigned>> &schedule,
               unsigned II, unsigned readLatency) {
  auto getLatency = [&](Operation *op) {
    if (isa<vector::TransferReadOp>(op))
      return readLatency;
    return unsigned(1);
  };

  DenseMap<Operation *, unsigned> opCycles;
  std::map<unsigned, std::vector<Operation *>> wrappedSchedule;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    unsigned earlyCycle = 0;
    for (Value operand : op.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      earlyCycle = std::max(earlyCycle, opCycles[def] + getLatency(def));
    }
    opCycles[&op] = earlyCycle;
    wrappedSchedule[earlyCycle % II].push_back(&op);
  }
  for (auto it : wrappedSchedule) {
    for (Operation *op : it.second) {
      unsigned cycle = opCycles[op];
      schedule.push_back(std::make_pair(op, cycle / II));
    }
  }
}

static FailureOr<scf::ForOp>
executePipelineLoopOp(scf::ForOp loop,
                      linalg::transform::PipelineLoopOp pipelineLoopOp) {
  // TODO: make the pipelining pattern return the transformed loop.
  if (!pipelineLoopOp->getUses().empty()) {
    InFlightDiagnostic diag = pipelineLoopOp.emitError()
                              << "NYI: cannot target the result of pipelining";
    diag.attachNote(pipelineLoopOp->use_begin()->getOwner()->getLoc())
        << "use here";
    return failure();
  }

  scf::PipeliningOption schedule;
  schedule.getScheduleFn =
      [pipelineLoopOp](
          scf::ForOp forOp,
          std::vector<std::pair<Operation *, unsigned>> &schedule) mutable {
        loopScheduling(forOp, schedule, pipelineLoopOp.iteration_interval(),
                       pipelineLoopOp.read_latency());
      };

  RewritePatternSet patterns(loop->getContext());
  scf::populateSCFLoopPipeliningPatterns(patterns, schedule);
  assert(patterns.getNativePatterns().size() == 1 &&
         "expected one pipelining pattern");
  auto functionalPattern = [&patterns](scf::ForOp forOp,
                                       PatternRewriter &rewriter) {
    RewritePattern *pattern = patterns.getNativePatterns().front().get();
    return pattern->matchAndRewrite(forOp, rewriter);
  };
  if (failed(functional::applyAt(loop, std::move(functionalPattern))))
    return failure();

  return scf::ForOp();
}

scf::ExecuteRegionOp outlineInExecuteRegion(RewriterBase &b, Operation *op) {
  op->dump();
  if (op->getNumRegions() != 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      b.create<scf::ExecuteRegionOp>(op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = b.cloneWithoutRegions(*op);
    Region &clonedRegion = clonedOp->getRegions().front();
    assert(clonedRegion.empty() && "expected empty region");
    b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                         clonedRegion.end());
    b.create<scf::YieldOp>(op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

static FailureOr<FuncOp>
executeOutlineLoopOp(scf::ForOp loop,
                     linalg::transform::OutlineLoopOp outlineLoopOp,
                     TransformOpMapping &operations) {
  PatternRewriterListener rewriter(loop->getContext());
  TrackingListener listener(operations);
  rewriter.addListener(&listener);
  Location loc = loop.getLoc();
  scf::ExecuteRegionOp exec = outlineInExecuteRegion(rewriter, loop);
  assert(exec && "failed to produce execute_region");
  FailureOr<FuncOp> outlined = outlineSingleBlockRegion(
      rewriter, loc, exec.getRegion(), outlineLoopOp.func_name());
  return outlined;
}

/// Run enabling transformations (LICM and its variants, single-iteration loop
/// removal, CSE) on the given function.
static LogicalResult performEnablerTransformations(
    FuncOp func, TransformOpMapping &operations,
    linalg::LinalgEnablingOptions options = linalg::LinalgEnablingOptions()) {
  MLIRContext *ctx = func->getContext();
  RewritePatternSet patterns(ctx);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  if (failed(applyPatternsTrackAndFoldGreedily(func, operations, std::move(patterns))))
    return failure();

  // This assumes LICM never removes operations so we don't need tracking.
  if (options.licm) {
    WalkResult result =
        func->walk([](LoopLikeOpInterface loopLike) -> WalkResult {
          return moveLoopInvariantCode(loopLike);
        });
    if (result.wasInterrupted())
      return failure();
  }

  func.walk([](Operation *op) {
    (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<AffineForOp, scf::ForOp>(
            [](auto loop) { return promoteIfSingleIteration(loop); })
        .Default([](Operation *) { return success(); });
  });

  if (options.hoistRedundantVectorTransfers)
    hoistRedundantVectorTransfers(func);
  if (options.hoistRedundantVectorTransfersOnTensor)
    hoistRedundantVectorTransfersOnTensor(func);

  eliminateCommonSubexpressionsWithTrackedOps(func, operations);
  return success();
}

/// Run enabling transformations on the given model while preserving the
/// operation tracking information.
static LogicalResult performEnablerTransformations(
    ModuleOp module, TransformOpMapping &operations,
    linalg::LinalgEnablingOptions options = linalg::LinalgEnablingOptions()) {
  for (auto func : module.getOps<FuncOp>()) {
    if (failed(performEnablerTransformations(func, operations, options)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Linalg Interpreter Driver
//===----------------------------------------------------------------------===//

template <typename ConfigOpTy>
static void removeCurrentTarget(ConfigOpTy configOp,
                                TransformOpMapping &operations) {
  // Since there is only allowed use of the value in the transformation dialect,
  // we can remove it from the mapping after processing its only user. This
  // ensures we don't accidentally keep pointers to operations that may have
  // been deleted by the current transformation.
  if (!configOp.target())
    return;
  Value target = configOp.target();
  assert(target.hasOneUse() && "expected values corresponding to transformed "
                               "operations to have one use");
  operations.erase(target);
}

/// Applies `transform` with options provided by `configOp` to all operations
/// specified as targets by `configOp`.
template <typename ConfigOpTy, typename FnTy>
static LogicalResult executeTransformOnEach(ModuleOp module,
                                            ConfigOpTy configOp, FnTy transform,
                                            TransformOpMapping &operations) {
  auto it = operations.find(configOp.target());
  if (it == operations.end()) {
    LLVM_DEBUG(DBGS() << "failed to find a target for:\n" << configOp << "\n");
    return failure();
  }
  ArrayRef<Operation *> targets = it->second;

  using TransformedOpType =
      typename llvm::function_traits<FnTy>::template arg_t<0>;
  SmallVector<Operation *> results = functional::applyForEach(
      targets, [&](Operation *op, PatternRewriter &) -> FailureOr<Operation *> {
        LLVM_DEBUG(DBGS() << "attempting to transform: " << op << "\n");
        auto specificOp =
            functional::detail::IsaOr<TransformedOpType>::dyn_cast(op);
        if (!specificOp) {
          LLVM_DEBUG(DBGS() << "unexpected operation type\n");
          return failure();
        }
        auto result = transform(specificOp, configOp);
        LLVM_DEBUG(DBGS() << "transformation "
                          << (failed(result) ? "failed" : "succeeded") << "\n");
        if (failed(result))
          return failure();
        return result->getOperation();
      });

  // All transformations must succeed.
  if (results.size() != targets.size())
    return failure();

  bool inserted =
      operations.insert({configOp.transformed(), std::move(results)}).second;
  assert(inserted && "value is already associated with another operation list");
  (void)inserted;

  removeCurrentTarget(configOp, operations);
  return success();
}

template <typename ConfigOpTy, typename FnTy>
static LogicalResult
executeNonReturningTransformOnEach(ModuleOp module, ConfigOpTy configOp,
                                   FnTy transform,
                                   TransformOpMapping &operations) {
  auto it = operations.find(configOp.target());
  if (it == operations.end()) {
    LLVM_DEBUG(DBGS() << "failed to find a target for:\n" << configOp << "\n");
    return failure();
  }
  ArrayRef<Operation *> targets = it->second;

  using TransformedOpType =
      typename llvm::function_traits<FnTy>::template arg_t<0>;
  for (Operation *op : targets) {
    LLVM_DEBUG(DBGS() << "attempting to transform: " << op << "\n");
    auto specificOp =
        functional::detail::IsaOr<TransformedOpType>::dyn_cast(op);
    if (!specificOp) {
      LLVM_DEBUG(DBGS() << "unexpected operation type\n");
      return failure();
    }
    LogicalResult result = transform(specificOp, configOp);
    LLVM_DEBUG(DBGS() << "transformation "
                      << (failed(result) ? "failed" : "succeeded") << "\n");
    if (failed(result))
      return failure();
  }

  removeCurrentTarget(configOp, operations);
  return success();
}

/// Applies the transformation specified by the given Linalg Transform dialect
/// operation to the given target operation. The `operations` table contains the
/// mapping between SSA values that correspond to operation handles produced and
/// used by Linalg Transform dialect operations, and the Operation* objects in
/// the code.
static LogicalResult executeTransform(Operation *operation, ModuleOp module,
                                      TransformOpMapping &operations) {
  if (auto matchOp = dyn_cast<transform::MatchOp>(operation))
    return executeMatchOp(matchOp, module, operations);

  if (auto tileOp = dyn_cast<transform::TileOp>(operation))
    return executeTransformOnEach(module, tileOp, &executeTileOp, operations);

  if (auto decomposeOp = dyn_cast<transform::DecomposeOp>(operation))
    return executeDecomposeOp(module, decomposeOp);

  if (auto vectorizeOp = dyn_cast<transform::VectorizeOp>(operation)) {
    return executeTransformOnEach(module, vectorizeOp, &executeVectorizeOp,
                                  operations);
  }

  if (auto lowerVectorsOp =
          dyn_cast<transform::LowerVectorsOp>(operation))
    return executeLowerVectorsOp(module, lowerVectorsOp);

  if (auto bufferizeOp = dyn_cast<transform::BufferizeOp>(operation))
    return executeBufferizeOp(module, bufferizeOp);

  if (auto lowerToLLVMOp = dyn_cast<transform::LowerToLLVMOp>(operation))
    return executeLowerToLLVMOp(module, lowerToLLVMOp);

  if (auto getParentLoopOp = dyn_cast<transform::GetParentLoopOp>(operation)) {
    return executeTransformOnEach(module, getParentLoopOp,
                                  executeGetParentLoopOp, operations);
  }

  if (auto unrollLoopOp = dyn_cast<transform::UnrollLoopOp>(operation)) {
    return executeNonReturningTransformOnEach(module, unrollLoopOp,
                                              executeUnrollLoopOp, operations);
  }

  if (auto pipelineLoopOp = dyn_cast<transform::PipelineLoopOp>(operation)) {
    return executeTransformOnEach(module, pipelineLoopOp, executePipelineLoopOp,
                                  operations);
  }

  if (auto outlineLoopOp = dyn_cast<transform::OutlineLoopOp>(operation)) {
    return executeTransformOnEach(
        module, outlineLoopOp,
        [&](scf::ForOp forOp, transform::OutlineLoopOp configOp) {
          return executeOutlineLoopOp(forOp, configOp, operations);
        },
        operations);
  }

  return operation->emitError() << "unknown transformation operation";
}

/// Applies the transformations listed in the `sequence` to operations starting
/// from `target`. The following transformations may be applied to operations
/// produced by previous transformations as indicated by SSA value flow in the
/// Linalg Transform dialect.
static LogicalResult executeSequence(linalg::transform::SequenceOp sequence,
                                     ModuleOp module) {
  TransformOpMapping operations;

  MLIRContext *ctx = module->getContext();
  RewritePatternSet patternList(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patternList);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patternList, ctx);
  FrozenRewritePatternSet patterns(std::move(patternList));

  // Run the canonicalizations upfront so we don't match and transform
  // operations only to drop them later.
  eliminateCommonSubexpressionsWithTrackedOps(module, operations);
  if (failed(applyPatternsTrackAndFoldGreedily(module, operations, patterns))) {
    LLVM_DEBUG(DBGS() << "failed to apply canonicalization patterns\n");
    return failure();
  }

  for (Operation &transform : sequence.body().front()) {
    if (failed(executeTransform(&transform, module, operations)))
      return transform.emitError() << "failed to apply";

    LLVM_DEBUG(DBGS() << "successfully applied transform: " << transform
                      << "\n");

    // Run CSE, enabling transformations and canonicalization. This is similar
    // to running the respective pass, but (a) keeps tracking the value/op
    // mapping and (b) avoids constructing the pattern set + pass pipeline on
    // every step.
    // TODO: consider better targeting than module-level transformations here:
    // e.g., the enabler internals can apply to one function only. Furthermore,
    // we don't need all of enabler transformations after/before all passes.
    eliminateCommonSubexpressionsWithTrackedOps(module, operations);

    // TODO: this runs CSE internally, mostly redundant with the above.
    if (failed(performEnablerTransformations(module, operations))) {
      LLVM_DEBUG(DBGS() << "enabler transformations failed\n");
      return failure();
    }

    if (failed(
            applyPatternsTrackAndFoldGreedily(module, operations, patterns))) {
      LLVM_DEBUG(DBGS() << "failed to apply canonicalization patterns\n");
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Linalg Interpreter Pass
//===----------------------------------------------------------------------===//

namespace {
/// Pass that executes transformations specified by a module-level
/// linalg_transform.apply operation on the same module.
struct InterpreterPass : public PassWrapper<InterpreterPass, Pass> {
  StringRef getArgument() const final { return "linalg-interp-transforms"; }
  StringRef getDescription() const final {
    return "Executes transformations specified in Linalg Transform dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect, StandardOpsDialect,
                    tensor::TensorDialect, vector::VectorDialect,
                    LLVM::LLVMDialect, bufferization::BufferizationDialect,
                    pdl_interp::PDLInterpDialect>();

    linalg::comprehensive_bufferize::affine_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::linalg_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::std_ext::
        registerModuleBufferizationExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module)
      return signalPassFailure();

    auto result = module.walk([&](linalg::transform::SequenceOp sequenceOp) {
      if (failed(executeSequence(sequenceOp, module)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
/// Creates a Linalg Transform interpreter pass.
std::unique_ptr<Pass> createLinalgTransformInterpreterPass() {
  return std::make_unique<InterpreterPass>();
}
} // namespace mlir

/// Registration hook for the Linalg Transform interpreter pass.
void mlir::linalg::transform::registerLinalgTransformInterpreterPass() {
  PassRegistration<InterpreterPass>();
}
