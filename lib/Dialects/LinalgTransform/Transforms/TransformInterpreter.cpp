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
#include "Dialects/LinalgTransform/TrackingRewriteDriver.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ArithInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/SCFInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/TensorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/VectorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "transform-interpreter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::linalg;

/// Extracts a vector of int64_t from an array attribute. Asserts if the
/// attribute contains values other than integers.
SmallVector<int64_t> extractI64Array(ArrayAttr attr) {
  SmallVector<int64_t> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getSExtValue());
  return result;
}

/// Extracts a vector of unsigned from an array attribute. Asserts if the
/// attribute contains values other than intergers. May truncate.
SmallVector<unsigned> extractUIntArray(ArrayAttr attr) {
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

/// The value used by the interpreter pass in the Linalg transform marker
/// attribute.
constexpr llvm::StringLiteral kInterpreterTransformMarkerValue =
    "linalg_transform.marker";

/// Populates `operations` with ops nested in `container` that have the
/// transform marker set, and removes the marker.
static void findMarkedOps(Operation *container,
                          SmallVectorImpl<Operation *> &operations) {
  container->walk([&](Operation *op) {
    auto attr =
        op->getAttrOfType<StringAttr>(LinalgTransforms::kLinalgTransformMarker);
    if (attr && attr.getValue() == kInterpreterTransformMarkerValue) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
      operations.push_back(op);
    }
  });
}

/// Returns a transformation filter that only matches the `target` operation and
/// sets the interpreter transform marker on it to ensure further discovery.
// TODO: If the patterns could communicate the result somehow, we wouldn't need
// to go through markers.
static LinalgTransformationFilter makeTransformationFilter(Operation *target) {
  auto initialTargetFilter = [target](Operation *operation) {
    return success(operation == target);
  };
  return LinalgTransformationFilter(
      initialTargetFilter, {},
      StringAttr::get(kInterpreterTransformMarkerValue, target->getContext()));
}

/// Applies the pad pattern to the given target operation as indicated by the
/// tile op that subsumes padding. Populates `nextTargets` with transformable
/// operations for further transformations (currently, the single padded op).
static LogicalResult
executePadFromTileOp(linalg::LinalgOp target, linalg::transform::TileOp tileOp,
                     SmallVectorImpl<Operation *> &nextTargets) {
  if (!target)
    return failure();

  if (!tileOp.pad()) {
    nextTargets.push_back(target);
    return success();
  }

  auto packFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < tileOp.pack_paddings().size()
               ? !tileOp.pack_paddings()[opOperand.getOperandNumber()]
                      .cast<IntegerAttr>()
                      .getValue()
                      .isZero()
               : false;
  };
  auto hoistingFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < tileOp.hoist_paddings().size()
               ? tileOp.hoist_paddings()[opOperand.getOperandNumber()]
                     .cast<IntegerAttr>()
                     .getValue()
                     .getSExtValue()
               : 0;
  };
  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  paddingOptions.setPaddingNoFoldComputationFunction(packFunc);
  paddingOptions.setPaddingHoistComputationFunction(hoistingFunc);

  MLIRContext *ctx = target->getContext();
  FuncOp parentFunc = target->getParentOfType<FuncOp>();
  RewritePatternSet patterns(ctx);
  patterns.insert<LinalgPaddingPattern>(ctx, paddingOptions,
                                        makeTransformationFilter(target));
  if (failed(applyPatternsAndFoldGreedily(target->getParentOfType<FuncOp>(),
                                          std::move(patterns))))
    return failure();

  findMarkedOps(parentFunc, nextTargets);
  return success();
}

/// Applies the generalization pattern to the given target operation as
/// indicated by the tile op that subsumes padding. Populates `nextTargets` with
/// transformable operations for further transformations (currently, the single
/// generalized op).
static LogicalResult
executeGeneralizeFromTileOp(linalg::LinalgOp target,
                            linalg::transform::TileOp tileOp,
                            SmallVectorImpl<Operation *> &nextTargets) {
  if (!target)
    return failure();

  if (!tileOp.generalize()) {
    nextTargets.push_back(target);
    return success();
  }

  MLIRContext *ctx = target->getContext();
  FuncOp parentFunc = target->getParentOfType<FuncOp>();
  RewritePatternSet patterns(ctx);
  patterns.insert<LinalgGeneralizationPattern>(
      ctx, makeTransformationFilter(target));
  if (failed(applyPatternsAndFoldGreedily(target->getParentOfType<FuncOp>(),
                                          std::move(patterns))))
    return failure();

  findMarkedOps(parentFunc, nextTargets);
  return success();
}

/// Applies the interchange pattern to the given target operation as indicated
/// by the tile op that subsumes padding. Populates `nextTargets` with
/// transformable operations for further transformations (currently, the single
/// interchanged op).
static LogicalResult
executeInterchangeFromTileOp(linalg::LinalgOp target,
                             linalg::transform::TileOp tileOp,
                             SmallVectorImpl<Operation *> &nextTargets) {
  if (!target)
    return failure();

  if (tileOp.interchange().empty()) {
    nextTargets.push_back(target);
    return success();
  }

  MLIRContext *ctx = target->getContext();
  FuncOp parentFunc = target->getParentOfType<FuncOp>();
  RewritePatternSet patterns(ctx);
  auto interchangeVector =
      llvm::to_vector<4>(llvm::map_range(tileOp.interchange(), [](Attribute a) {
        return static_cast<unsigned>(
            a.cast<IntegerAttr>().getValue().getZExtValue());
      }));
  patterns.insert<GenericOpInterchangePattern>(
      ctx, interchangeVector, makeTransformationFilter(target));
  if (failed(applyPatternsAndFoldGreedily(target->getParentOfType<FuncOp>(),
                                          std::move(patterns))))
    return failure();

  findMarkedOps(parentFunc, nextTargets);
  return success();
}

/// Applies the transformation specified by the given tile operation to the
/// given target operation. Populates `results` with transformation operations
/// for further transformations if the pattern applied successfully (currently,
/// the single op after tiling).
// TODO: if the tiling pattern failed, we still want to populate results with
// something.
static LogicalResult executeTileOp(linalg::LinalgOp target,
                                   linalg::transform::TileOp tileOp,
                                   SmallVectorImpl<Operation *> &results) {
  LinalgTilingOptions tilingOptions;
  tilingOptions.setTileSizes(extractI64Array(tileOp.sizes()));
  tilingOptions.setInterchange(extractUIntArray(tileOp.interchange()));
  if (tileOp.scalarize_dyn_dims())
    tilingOptions.scalarizeDynamicDims();

  MLIRContext *ctx = target->getContext();
  FuncOp parentFunc = target->getParentOfType<FuncOp>();
  RewritePatternSet patterns(ctx);
  patterns.insert<LinalgGenericTilingPattern>(
      ctx, makeTransformationFilter(target), tilingOptions);
  if (failed(applyPatternsAndFoldGreedily(target->getParentOfType<FuncOp>(),
                                          std::move(patterns))))
    return failure();

  // TODO: this should go into a custom driver, presumably. And this looks
  // redundant with that, maybe we shouldn't have tiling subsume pad, generalize
  // and interchange.
  SmallVector<Operation *> targets;
  findMarkedOps(parentFunc, targets);

  // TODO: we need some sort of array monad for this to work nicely.
  SmallVector<Operation *> nextTargets;
  for (Operation *nextTarget : targets) {
    if (failed(executePadFromTileOp(dyn_cast<LinalgOp>(nextTarget), tileOp,
                                    nextTargets)))
      return failure();
  }

  std::swap(targets, nextTargets);
  nextTargets.clear();

  for (Operation *nextTarget : targets) {
    if (failed(executeGeneralizeFromTileOp(dyn_cast<LinalgOp>(nextTarget),
                                           tileOp, nextTargets)))
      return failure();
  }

  std::swap(targets, nextTargets);
  nextTargets.clear();

  for (Operation *nextTarget : targets) {
    if (failed(executeInterchangeFromTileOp(dyn_cast<LinalgOp>(nextTarget),
                                            tileOp, nextTargets)))
      return failure();
  }

  llvm::append_range(results, nextTargets);
  return success();
}

/// Applies the transformation specified by the given decompose operation to the
/// given target operation.
static LogicalResult
executeDecomposeOp(FuncOp target, linalg::transform::DecomposeOp decomposeOp) {
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  populateDecomposeConvolutionPatterns(patterns,
                                       makeTransformationFilter(target));
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return failure();

  // TODO: make this chainable, it isn't in the original codegenstrategy.
  SmallVector<Operation *> ignored;
  findMarkedOps(target, ignored);

  return success();
}

/// Applies the transformation specified by the given vectorize operation to the
/// given target operation AND some related operations.Populates `results` with
/// transformation operations for further transformations if the pattern applied
/// successfully (currently, the main "contraction" op after vectorization).
static LogicalResult
executeVectorizeOp(linalg::LinalgOp target,
                   linalg::transform::VectorizeOp vectorizeOp,
                   SmallVectorImpl<Operation *> &results) {
  MLIRContext *ctx = target->getContext();
  FuncOp parentFunc = target->getParentOfType<FuncOp>();
  RewritePatternSet patterns(ctx);

  LinalgTransformationFilter filter = makeTransformationFilter(target);
  LinalgVectorizationOptions options;
  patterns.insert<LinalgVectorizationPattern>(ctx, filter, options);

  // TODO: this is copy-pasta from LinalgStrategyVectorizePass, it shouldn't be.
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                       /*benefit=*/2);
  if (vectorizeOp.vectorize_padding())
    linalg::populatePadTensorOpVectorizationPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target->getParentOfType<FuncOp>(),
                                          std::move(patterns))))
    return failure();

  // TODO: complication, only the vectorization pattern is actually targeted,
  // the rest of the patterns cannot be targeted and need a different scoping
  // mechanism (not sure how much IR must be pulled into the scope).
  //
  // TODO: vectorization may fail because the op is not vectorizable, unclear
  // what to do here. We should probably report it somehow, but we may also
  // want to go on and keep the original for continuation. Should we have
  // some notion of transformation optionality vs. mandatory (like lowering)?
  // How to find ops that were not replaced?
  findMarkedOps(parentFunc, results);

  return success();
}

/// Returns true of the numbered vector lowering stage is included into the list
/// of stages specified on the given lowerVectors opreation.
static bool stageIncluded(int stage, transform::LowerVectorsOp lowerVectorsOp) {
  for (auto s : lowerVectorsOp.stages().getAsValueRange<IntegerAttr>()) {
    if (s.getSExtValue() == stage)
      return true;
  }
  return false;
}

/// Appplies the transformation specified by the given lower vectors operation
/// to the given function.
static LogicalResult
executeLowerVectorsOp(FuncOp target,
                      linalg::transform::LowerVectorsOp lowerVectorsOp) {
  MLIRContext *ctx = target->getContext();
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

  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransformsOptions(vectorContractLowering)
      .setVectorMultiReductionLowering(vectorMultiReductionLowering)
      .setVectorTransposeLowering(vectorTransposeLowering);

  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions()
          .enableFullUnroll(true)
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
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return failure();

  // TODO: make composable...
  return success();
}

/// Appplies the transformation specified by the given bufferize operation to
/// the module containing the given function.
static LogicalResult
executeBufferizeOp(FuncOp target, linalg::transform::BufferizeOp bufferizeOp) {
  // TODO: is this even feasible to scope bufferization at something else than
  // module? It feels like it should be a hard barrier for all transformations.
  auto module = target->getParentOfType<ModuleOp>();
  PassManager pm(target->getContext());

  pm.addPass(createLinalgComprehensiveModuleBufferizePass());
  if (failed(pm.run(module)))
    return failure();

  // Perform buffer-level hoistings.
  module.walk([&](FuncOp funcOp) { hoistRedundantVectorTransfers(funcOp); });
  return success();
}

/// Appplies the transformation specified by the given Lower to LLVM operation
/// to the module containing the given function.
static LogicalResult
executeLowerToLLVMOp(FuncOp target,
                     linalg::transform::LowerToLLVMOp lowerToLLVMOp) {
  // TODO: it is feasible to scope lowering at arbitrary level and introduce
  // unrealized casts, but there needs to be the final module-wise cleanup in
  // the end. Keep module-level for now.
  PassManager pm(target->getContext());

  pm.addNestedPass<FuncOp>(createConvertVectorToSCFPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createLowerToCFGPass());
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
  return pm.run(target->getParentOfType<ModuleOp>());
}

/// Appplies the transformation specified by the given Linalg Transform dialect
/// operation to the given target operation. The `operations` table contains the
/// mapping between SSA values that correspond to operation handles produced and
/// used by Linalg Transform dialect operations, and the Operation* objects in
/// the code.
static LogicalResult
executeTransform(Operation *operation, FuncOp surroundingFunc,
                 DenseMap<Value, Operation *> &operations) {

  if (auto tileOp = dyn_cast<linalg::transform::TileOp>(operation)) {
    Operation *target = operations.lookup(tileOp.op());
    // TODO: better error reporting story
    assert(target);
    auto linalgTarget = dyn_cast<LinalgOp>(target);
    if (!linalgTarget) {
      LLVM_DEBUG(DBGS() << "target is not a linalg op");
      return failure();
    }

    SmallVector<Operation *> results;
    if (failed(executeTileOp(linalgTarget, tileOp, results)))
      return failure();

    assert(results.size() == 1);
    operations.try_emplace(tileOp.transformed(), results.front());
    return success();
  }

  if (auto decomposeOp = dyn_cast<linalg::transform::DecomposeOp>(operation))
    return executeDecomposeOp(surroundingFunc, decomposeOp);

  if (auto vectorizeOp = dyn_cast<linalg::transform::VectorizeOp>(operation)) {
    Operation *target = operations.lookup(vectorizeOp.op());
    // TODO: better error reporting story
    assert(target);
    auto linalgTarget = dyn_cast<LinalgOp>(target);
    if (!linalgTarget) {
      LLVM_DEBUG(DBGS() << "target is not a linalg op");
      return failure();
    }

    SmallVector<Operation *> results;
    if (failed(executeVectorizeOp(linalgTarget, vectorizeOp, results)))
      return failure();

    // TODO: unclear what to do if vectorization failed. Same for tiling btw.
    // assert(results.size() == 1);
    // operations.try_emplace(vectorizeOp.transformed(), results.front());
    return success();
  }

  if (auto lowerVectorsOp =
          dyn_cast<linalg::transform::LowerVectorsOp>(operation))
    return executeLowerVectorsOp(surroundingFunc, lowerVectorsOp);

  if (auto bufferizeOp = dyn_cast<linalg::transform::BufferizeOp>(operation))
    return executeBufferizeOp(surroundingFunc, bufferizeOp);

  if (auto lowerToLLVMOp = dyn_cast<transform::LowerToLLVMOp>(operation))
    return executeLowerToLLVMOp(surroundingFunc, lowerToLLVMOp);

  return operation->emitError() << "unknown transformation operation";
}

/// Applies the transformations listed in the `sequence` to operations starting
/// from `target`. The following transformations may be applied to operations
/// produced by previous transformations as indicated by SSA value flow in the
/// Linalg Transform dialect.
static LogicalResult executeSequence(linalg::transform::SequenceOp sequence,
                                     Operation *target) {
  Block *applyBlock =
      &sequence->getParentOfType<transform::ApplyOp>().transforms().front();
  if (applyBlock->getNumArguments() != 1)
    return sequence.emitError()
           << "only single-argument sequence blocks are supported";

  DenseMap<Value, Operation *> operations;
  operations.try_emplace(applyBlock->getArgument(0), target);

  // Stash the function for the cases where we need to apply function-wise
  // transforms.
  FuncOp func = target->getParentOfType<FuncOp>();
  ModuleOp module = func->getParentOfType<ModuleOp>();

  MLIRContext *ctx = target->getContext();
  RewritePatternSet patternList(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patternList);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patternList, ctx);
  FrozenRewritePatternSet patterns(std::move(patternList));

  for (Operation &transform : sequence.body().front()) {
    if (failed(executeTransform(&transform, func, operations))) {
      // TODO: should this be a user-visible error?
      LLVM_DEBUG(DBGS() << "failed to apply transform: " << transform << "\n");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "successfully applied transform: " << transform
                      << "\n");

    // Run canonicalization, this is similar to running the canonicalizer pass,
    // but (a) keeps tracking the value/op mapping and (b) avoids constructing
    // the pattern set + pass pipeline on every step.
    Operation *canonicalizationRoot = func;
    if (isa<transform::LowerToLLVMOp>(transform)) {
      // We cannot run the canonicalizer at a function level past LLVM
      // conversion, so check that it is the last one.
      // TODO: this may go away if we have a better scoping mechanism for large
      // transformations such as bufferization and lowerings and track functions
      // across them similarly to how we track linalg operations.
      assert(&transform == &sequence.body().front().back() &&
             "expected lowering to llvm to be the last transformation");
      canonicalizationRoot = module;
    }

    if (failed(applyPatternsTrackAndFoldGreedily(canonicalizationRoot,
                                                 operations, patterns))) {
      LLVM_DEBUG(DBGS() << "failed to apply canonicalization patterns\n");
      return failure();
    }
  }

  return success();
}

/// Hook for PDL driver to check if an operation (`value`) is directly nested in
/// a function with the name provided as constant parameter.
static LogicalResult nestedInFunc(PDLValue value, ArrayAttr constantParams,
                                  PatternRewriter &rewriter) {
  auto *operation = value.cast<Operation *>();
  auto func = operation->getParentOfType<FuncOp>();
  assert(constantParams.size() == 1 &&
         "expected a constant param with function name");
  auto functionSymbol = constantParams[0].dyn_cast<SymbolRefAttr>();
  assert(functionSymbol && "expected a function name");

  if (!func)
    return rewriter.notifyMatchFailure(operation, "not nested in a function");
  return success(SymbolTable::lookupNearestSymbolFrom<FuncOp>(
                     operation, functionSymbol) == func);
}

constexpr llvm::StringLiteral kInterpreterMatched = "linalg_transform.matched";

/// Hook for PDL driver to check if the operation is not marked with the matched
/// marker.
static LogicalResult notTagged(PDLValue value, ArrayAttr constantParams,
                               PatternRewriter &rewriter) {
  auto *operation = value.cast<Operation *>();
  return success(!operation->getAttr(kInterpreterMatched));
}

/// Factory for PDL driver rewrite hooks that do nothing but add matched
/// operations to the `targets` list.
static PDLRewriteFunction
makePDLRewriter(SmallVectorImpl<Operation *> &targets) {
  auto rewriter = [&targets](ArrayRef<PDLValue> args, ArrayAttr constantParams,
                             PatternRewriter &rewriter, PDLResultList &) {
    assert(args.size() == 1 && "expected one argument");

    // TODO: this is a hack in absence of a custom driver that can consume PDL.
    // Just add an attribute to the op that is selected as a transformation
    // target, do nothing else. Separate code will find them and perform actual
    // transformations, otherwise we risk having one (nested) rewriter modify
    // the state known to another (outer PDL-level) rewriter.
    targets.push_back(args.front().cast<Operation *>());
    args.front().cast<Operation *>()->setAttr(kInterpreterMatched,
                                              rewriter.getUnitAttr());
  };
  return rewriter;
}

/// Executes the Linalg Transform apply operation. This first runs the `when`
/// condition to find the list of ops that match the transformation criteria
/// using PDL interpreter and then applies the transformations indicated in the
/// sequence region.
static LogicalResult executeApply(linalg::transform::ApplyOp applyOp,
                                  ModuleOp module) {
  Block *condition = &applyOp.condition().front();
  if (condition->empty())
    return applyOp.emitError() << "expected non-empty 'when' condition";

  auto pdlMatchOp = dyn_cast<linalg::transform::PDLMatchOp>(condition->front());
  if (!llvm::hasSingleElement(*condition) || !pdlMatchOp) {
    return applyOp.emitError()
           << "only a single '"
           << linalg::transform::PDLMatchOp::getOperationName()
           << "' condition is currently supported";
  }

  auto patternOp = SymbolTable::lookupNearestSymbolFrom<pdl::PatternOp>(
      pdlMatchOp, pdlMatchOp.pattern());

  OwningModuleRef pdlModuleOp = ModuleOp::create(pdlMatchOp.getLoc());
  pdlModuleOp->getBody(0)->getOperations().splice(
      pdlModuleOp->getBody(0)->begin(), patternOp->getBlock()->getOperations(),
      patternOp.getOperation());
  PDLPatternModule pdlModule(std::move(pdlModuleOp));
  pdlModule.registerConstraintFunction("nestedInFunc", nestedInFunc);
  pdlModule.registerConstraintFunction("notTagged", notTagged);

  Block *schedule = &applyOp.transforms().front();
  if (schedule->empty())
    return applyOp.emitError() << "expected non-empty transforms block";
  auto sequenceOp = dyn_cast<linalg::transform::SequenceOp>(schedule->front());
  if (!llvm::hasSingleElement(*schedule) || !sequenceOp) {
    return applyOp.emitError()
           << "only a single '"
           << linalg::transform::SequenceOp::getOperationName()
           << "' is currently supported";
  }

  SmallVector<Operation *> rewriteTargets;
  pdlModule.registerRewriteFunction("linalg_transform.apply",
                                    makePDLRewriter(rewriteTargets));

  RewritePatternSet patterns(std::move(pdlModule));
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    emitError(module.getLoc()) << "failed to apply PDL patterns";
    return failure();
  }

  for (Operation *op : rewriteTargets)
    if (failed(executeSequence(sequenceOp, op)))
      return failure();
  return success();
}

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
    linalg::comprehensive_bufferize::arith_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::linalg_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::scf_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::std_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::tensor_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::vector_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module)
      return signalPassFailure();

    auto result = module.walk([&](linalg::transform::ApplyOp applyOp) {
      if (failed(executeApply(applyOp, module)))
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
