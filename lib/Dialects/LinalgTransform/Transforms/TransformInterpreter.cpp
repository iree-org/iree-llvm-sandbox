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
#include "FunctionHelpers.h"
#include "Transforms/Functional.h"
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
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
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
  // populateDecomposeConvolutionPatterns(patterns,
  //                                      makeTransformationFilter(target));
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

  bufferization::AnalysisBufferizationOptions options;
  options.memCpyFn = [](OpBuilder &builder, Location loc, Value from,
                        Value to) {
    return success(linalg::makeMemRefCopyOp(builder, loc, from, to));
  };
  pm.addPass(createLinalgComprehensiveModuleBufferizePass(options));
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
//
/// Apply at most one pattern from the given list to each operation nested in
/// `parent`.
static SmallVector<LinalgOp>
applyPatternsOnce(Operation *parent, const FrozenRewritePatternSet &patterns) {
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // TODO: The C++ functional API needs better interoperability with PDL.
  return functional::applyForEachIn(
      parent,
      [&](Operation *op, PatternRewriter &rewriter) -> FailureOr<LinalgOp> {
        if (succeeded(applicator.matchAndRewrite(op, rewriter)))
          if (auto linalgOp = dyn_cast<LinalgOp>(op))
            return linalgOp;
        return failure();
      });
}

/// Hook for PDL driver to check if an operation (`value`) is directly nested in
/// a function with the name provided as constant parameter.
/// TODO: PDL needs user-defined "questions".
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
  return success(functionSymbol.getLeafReference() == func.getName());
}

/// PDL rewrite hook that does nothing.
static void noOpRewriter(ArrayRef<PDLValue> args, ArrayAttr constantParams,
                         PatternRewriter &rewriter, PDLResultList &results) {
  assert(args.size() == 1 && "expected one argument");
#ifndef NDEBUG
  args.front().cast<Operation *>()->setAttr("linalg_transform.matched",
                                            rewriter.getUnitAttr());
#endif
}

/// Returns the operations that `transformOp` should be run on. The
/// `transformOp` is expected to have the TargetableTransformOpTrait. Its target
/// may be either an SSA value that corresponds to a list of ops produced by
/// other transforms and stored in `operations` or a list of ops that match the
/// PDL pattern specified by its name.
template <typename OpTy>
static FailureOr<SmallVector<LinalgOp>>
findTransformTarget(OpTy transformOp, ModuleOp module,
                    const DenseMap<Value, SmallVector<LinalgOp>> &operations) {
  if (Value target = transformOp.target())
    return operations.lookup(target);

  assert(transformOp.targetMatcher().hasValue() &&
         "expected either an operand or a matcher attribute");
  auto patternOp = SymbolTable::lookupNearestSymbolFrom<pdl::PatternOp>(
      transformOp, *transformOp.targetMatcher());
  if (!patternOp)
    return {transformOp->emitError() << "could not find the pattern by name"};

  // Clone the pattern operation into the temporary module used by the driver
  // as it might be referenced multiple times.
  OwningOpRef<ModuleOp> pdlModuleOp = ModuleOp::create(transformOp.getLoc());
  pdlModuleOp->getBody(0)->getOperations().push_front(patternOp->clone());
  PDLPatternModule pdlModule(std::move(pdlModuleOp));
  pdlModule.registerConstraintFunction("nestedInFunc", nestedInFunc);

  pdlModule.registerRewriteFunction("linalg_transform.apply", noOpRewriter);

  RewritePatternSet patterns(std::move(pdlModule));
  return applyPatternsOnce(module, std::move(patterns));
}

/// Applies `transform` with options provided by `configOp` to all operations
/// specified as targets by `configOp`.
template <typename OpTy, typename FnTy>
static LogicalResult
executeTransformOnEach(ModuleOp module, OpTy configOp, FnTy transform,
                       DenseMap<Value, SmallVector<LinalgOp>> &operations) {
  FailureOr<SmallVector<LinalgOp>> targets =
      findTransformTarget(configOp, module, operations);
  if (failed(targets)) {
    LLVM_DEBUG(DBGS() << "failed to find a linalg op target");
    return failure();
  }

  SmallVector<LinalgOp> results =
      functional::applyForEach(*targets, [&](LinalgOp op, PatternRewriter &) {
        LLVM_DEBUG(DBGS() << "attempting to transform: " << op << "\n");
        auto result = transform(op, configOp);
        LLVM_DEBUG(DBGS() << "transformation "
                          << (failed(result) ? "failed" : "succeeded") << "\n");
        return result;
      });

  // All transformations must succeed.
  if (results.size() != targets->size())
    return failure();

  bool inserted =
      operations.insert({configOp.transformed(), std::move(results)}).second;
  assert(inserted && "value is already associated with another operation list");
  (void)inserted;

  // Since there is only allowed use of the value in the transformation dialect,
  // we can remove it from the mapping after processing its only user. This
  // ensures we don't accidentally keep pointers to operations that may have
  // been deleted by the current transformation.
  if (configOp.target()) {
    Value target = configOp.target();
    assert(target.hasOneUse());
    operations.erase(target);
  }

  return success();
}

/// Applies the transformation specified by the given Linalg Transform dialect
/// operation to the given target operation. The `operations` table contains the
/// mapping between SSA values that correspond to operation handles produced and
/// used by Linalg Transform dialect operations, and the Operation* objects in
/// the code.
static LogicalResult executeTransform(Operation *operation, ModuleOp module,
                                      TransformOpMapping &operations) {
  if (auto tileOp = dyn_cast<linalg::transform::TileOp>(operation)) {
    return executeTransformOnEach(module, tileOp, &executeTileOp, operations);
  }

  if (auto decomposeOp = dyn_cast<linalg::transform::DecomposeOp>(operation))
    return executeDecomposeOp(module, decomposeOp);

  if (auto vectorizeOp = dyn_cast<linalg::transform::VectorizeOp>(operation)) {
    return executeTransformOnEach(module, vectorizeOp, &executeVectorizeOp,
                                  operations);
  }

  if (auto lowerVectorsOp =
          dyn_cast<linalg::transform::LowerVectorsOp>(operation))
    return executeLowerVectorsOp(module, lowerVectorsOp);

  if (auto bufferizeOp = dyn_cast<linalg::transform::BufferizeOp>(operation))
    return executeBufferizeOp(module, bufferizeOp);

  if (auto lowerToLLVMOp = dyn_cast<transform::LowerToLLVMOp>(operation))
    return executeLowerToLLVMOp(module, lowerToLLVMOp);

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
