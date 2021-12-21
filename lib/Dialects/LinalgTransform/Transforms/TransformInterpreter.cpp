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
#include "Dialects/LinalgTransform/TrackingCSE.h"
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
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
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
  SmallVector<Operation *> ignored;
  findMarkedOps(module, ignored);

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

/// Appplies the transformation specified by the given bufferize operation to
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

/// Appplies the transformation specified by the given Lower to LLVM operation
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
  return pm.run(module);
}

namespace {
/// The only purpose of this class is to enable creation of PatternRewriter
/// instances as the base class doesn't have a public constructor.
class SimplePatternRewriter : public PatternRewriter {
public:
  explicit SimplePatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};
} // namespace

/// Apply at most one pattern from the given list to each operation nested in
/// `parent`.
static void applyPatternsOnce(Operation *parent,
                              const FrozenRewritePatternSet &patterns) {
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // This assumes that patterns are only used for matching so there is no need
  // for worklists or any sort of insertion/deletion tracking. This may change
  // later, at which point the IR transformations will have to notify this
  // rewriter somehow. Alternatively, we could have only the matching part, but
  // we would need directl access to PDLBytecode for that.
  SimplePatternRewriter rewriter(parent->getContext());
  parent->walk([&](Operation *op) {
    rewriter.setInsertionPoint(op);
    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      LLVM_DEBUG(DBGS() << "failed to match any pattern to " << *op << "\n");
    }
  });
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

#ifndef NDEBUG
constexpr llvm::StringLiteral kInterpreterMatched = "linalg_transform.matched";
#endif

/// Factory for PDL driver rewrite hooks that do nothing but add matched
/// operations to the `targets` list.
static PDLRewriteFunction
makePDLRewriter(SmallVectorImpl<Operation *> &targets) {
  auto rewriter = [&targets](ArrayRef<PDLValue> args, ArrayAttr constantParams,
                             PatternRewriter &rewriter, PDLResultList &) {
    assert(args.size() == 1 && "expected one argument");

    // Just add an attribute to the op that is selected as a transformation
    // target, do nothing else. Separate code will find them and perform actual
    // transformations, otherwise we risk having one (nested) rewriter modify
    // the state known to another (outer PDL-level) rewriter.
    targets.push_back(args.front().cast<Operation *>());
#ifndef NDEBUG
    args.front().cast<Operation *>()->setAttr(kInterpreterMatched,
                                              rewriter.getUnitAttr());
#endif
  };
  return rewriter;
}

/// Appends to `targets` the operations that `transformOp` should be run op.
/// The `transformOp` is expected to have the TargetableTransformOpTrait. Its
/// target may be either an SSA value that corresponds to a list of ops
/// produced by other transforms and stored in `operations` or a list of ops
/// that match the PDL pattern specified by its name.
template <typename OpTy>
static LogicalResult findTransformTarget(
    OpTy transformOp, ModuleOp module,
    const DenseMap<Value, SmallVector<Operation *, 4>> &operations,
    SmallVectorImpl<Operation *> &targets) {
  if (transformOp.op()) {
    size_t initialSize = targets.size();
    llvm::append_range(targets, operations.lookup(transformOp.op()));
    if (llvm::any_of(llvm::drop_begin(targets, initialSize),
                     [](Operation *op) { return !isa<LinalgOp>(op); })) {
      return failure();
    }
    return success();
  }

  assert(transformOp.matcher().hasValue() &&
         "expected either an operand or a matcher attribute");
  auto patternOp = SymbolTable::lookupNearestSymbolFrom<pdl::PatternOp>(
      transformOp, *transformOp.matcher());
  if (!patternOp)
    return transformOp->emitError() << "could not find the pattern by name";

  // Clone the pattern operation into the temporary module used by the driver
  // as it might be referenced multiple times.
  OwningModuleRef pdlModuleOp = ModuleOp::create(transformOp.getLoc());
  pdlModuleOp->getBody(0)->getOperations().push_front(patternOp->clone());
  PDLPatternModule pdlModule(std::move(pdlModuleOp));
  pdlModule.registerConstraintFunction("nestedInFunc", nestedInFunc);

  pdlModule.registerRewriteFunction("linalg_transform.apply",
                                    makePDLRewriter(targets));

  RewritePatternSet patterns(std::move(pdlModule));
  applyPatternsOnce(module, std::move(patterns));
  if (targets.empty()) {
    // TODO: better error reporting story.
    LLVM_DEBUG(DBGS() << "could not match any operation with " << transformOp
                      << "\n");
    return failure();
  }

  return success();
}

/// Applies `transform` with options provided by `configOp` to all operations
/// specified as targets by `configOp`.
template <typename OpTy, typename FnTy>
static LogicalResult executeTransformOnEach(
    ModuleOp module, OpTy configOp, FnTy transform,
    DenseMap<Value, SmallVector<Operation *, 4>> &operations) {
  SmallVector<Operation *> targets;
  if (failed(findTransformTarget(configOp, module, operations, targets))) {
    LLVM_DEBUG(DBGS() << "could not find a linalg op target");
    return failure();
  }

  for (Operation *target : targets) {
    assert(target && "null target");
    auto linalgTarget = dyn_cast<LinalgOp>(target);
    if (!linalgTarget) {
      LLVM_DEBUG(DBGS() << "non-linalg operation found " << *target << "\n");
      return failure();
    }
    SmallVector<Operation *> results;
    if (failed(transform(linalgTarget, configOp, results)))
      return failure();

    assert(results.size() <= 1 && "multi-result transformation");
    if (results.empty()) {
      LLVM_DEBUG(DBGS() << "failed to transform " << *target << "\n");
      continue;
    }
    operations[configOp.transformed()].push_back(results.front());

    // TODO: what to do if the following target was somehow modified by
    // operation on the current target?
  }
  return success();
}

/// Appplies the transformation specified by the given Linalg Transform dialect
/// operation to the given target operation. The `operations` table contains the
/// mapping between SSA values that correspond to operation handles produced and
/// used by Linalg Transform dialect operations, and the Operation* objects in
/// the code.
static LogicalResult
executeTransform(Operation *operation, ModuleOp module,
                 DenseMap<Value, SmallVector<Operation *, 4>> &operations) {
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
  DenseMap<Value, SmallVector<Operation *, 4>> operations;

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
    if (failed(executeTransform(&transform, module, operations))) {
      // TODO: should this be a user-visible error?
      LLVM_DEBUG(DBGS() << "failed to apply transform: " << transform << "\n");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "successfully applied transform: " << transform
                      << "\n");

    // TODO: remove entries from `operations` if the value is not used by any of
    // the remaining transformations. This will allow the operation to be
    // replaced/erased by cleanups below and is slightly more effective.

    // Run canonicalization and CSE. This is similar to running the
    // canonicalizer pass, but (a) keeps tracking the value/op mapping and (b)
    // avoids constructing the pattern set + pass pipeline on every step.
    // TODO: consider better targeting that module-level transformations here.
    eliminateCommonSubexpressionsWithTrackedOps(module, operations);
    if (failed(
            applyPatternsTrackAndFoldGreedily(module, operations, patterns))) {
      LLVM_DEBUG(DBGS() << "failed to apply canonicalization patterns\n");
      return failure();
    }
  }

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
