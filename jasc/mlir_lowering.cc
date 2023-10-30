#include "mlir_lowering.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/Support/SourceMgr.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"
#include "mhlo/transforms/passes.h"

#include "dialect/dialect.h"
#include "dialect/ops.h"
#include "gpu_lowering_passes.h"
#include "transform_ops/dialect_extension.h"

namespace jasc {
namespace {

// Runs the pass manager on the model and handles errors.
absl::Status RunPassManager(mlir::PassManager& pm, mlir::ModuleOp module,
                            bool dump_ir) {
  std::string error_message;
  llvm::raw_string_ostream os(error_message);
  mlir::MLIRContext* context = module.getContext();
  llvm::SourceMgr srcMgr;
  mlir::SourceMgrDiagnosticHandler handler(srcMgr, context, os);

  bool multithreaded = context->isMultithreadingEnabled();
  if (dump_ir) {
    context->disableMultithreading();
    pm.enableIRPrinting([](auto*, auto*) { return false; });
  }

  mlir::LogicalResult result = pm.run(module);
  if (multithreaded && dump_ir) {
    context->enableMultithreading();
  }

  if (mlir::succeeded(result)) return absl::OkStatus();
  return absl::InternalError("Failed to apply transformations:\n\n" +
                             error_message);
}

// Base class for passes applicable to any operations. This is needed to
// reduce template nesting below.
template <typename Derived>
class OpPassWrapper : public mlir::PassWrapper<Derived, mlir::OperationPass<>> {
};
class ApplyTransformScriptPass
    : public mlir::transform::TransformInterpreterPassBase<
          ApplyTransformScriptPass, OpPassWrapper> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyTransformScriptPass)

  ApplyTransformScriptPass() = default;
  ApplyTransformScriptPass(const ApplyTransformScriptPass& pass)
      : mlir::transform::TransformInterpreterPassBase<ApplyTransformScriptPass,
                                                      OpPassWrapper>(pass) {}
  ApplyTransformScriptPass(const mlir::transform::TransformOptions& _options)
      : mlir::transform::TransformInterpreterPassBase<ApplyTransformScriptPass,
                                                      OpPassWrapper>(_options) {
  }

  void runOnOperation() override {
    options.enableEnforceSingleToplevelTransformOp(
        enforceSingleToplevelTransformOp);
    TransformInterpreterPassBase::runOnOperation();
  }

  // Returns the name of the pass suitable for the pass manager.
  llvm::StringRef getArgument() const override {
    return "jasc-apply-transform-script";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<jasc::JascDialect>();
    jasc::registerTransformDialectExtension(registry);
  }

  // MLIR pass options. This MUST use exactly the specified names.
  // clang-format off
  Option<std::string> transformFileName{  // NOLINT
      *this, "transform-file-name", llvm::cl::init(""),
      llvm::cl::desc(
          "Optional filename containing a transform dialect specification to "
          "apply. If left empty, the IR is assumed to contain one top-level "
          "transform dialect operation somewhere in the module.")};
  Option<std::string> debugPayloadRootTag{  // NOLINT
      *this, "debug-payload-root-tag", llvm::cl::init(""),
      llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as payload IR root. If empty select the pass "
          "anchor "
          "operation as the payload IR root.")};
  Option<std::string> debugTransformRootTag{  // NOLINT
      *this, "debug-transform-root-tag", llvm::cl::init(""),
      llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as container IR for top-level transform ops. This "
          "allows user control on what transformation to apply. If empty, "
          "select the container of the top-level transform op.")};
  ListOption<std::string> transformLibraryPaths{  // NOLINT
      *this, "transform-library-paths", llvm::cl::ZeroOrMore,
      llvm::cl::desc(
          "Optional name of the file containing transform dialect symbol "
          "definitions to be injected into the transform module.")};
  Option<bool> enforceSingleToplevelTransformOp{
      *this, "enforce-single-top-level-transform-op", llvm::cl::init(true),
      llvm::cl::desc("Ensure that only a single top-level transform op is "
                     "present in the IR.")};
  // clang-format on
};

std::unique_ptr<mlir::Pass> CreateApplyTransformScriptPass(
    llvm::StringRef name) {
  auto pass = std::make_unique<ApplyTransformScriptPass>();
  std::string path = "";
  path.append(name);
  pass->transformFileName = path;
  return std::move(pass);
}

class LowerTagRegionsPass
    : public mlir::PassWrapper<LowerTagRegionsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTagRegionsPass)

  // Returns the name of the pass suitable for the pass manager.
  llvm::StringRef getArgument() const override {
    return "jasc-lower-tag-regions";
  }

  void runOnOperation() override {
    getOperation().walk([](jasc::TagRegionOp tag_region_op) {
      mlir::StringAttr name = tag_region_op.getNameAttr();
      tag_region_op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
        if (llvm::isa<jasc::ReturnOp, mlir::tensor::EmptyOp>(op)) return;
        llvm::SmallVector<mlir::Attribute> new_array;
        auto old_array = op->getAttrOfType<mlir::ArrayAttr>("jasc_tags");
        if (old_array != nullptr) {
          new_array.append(old_array.begin(), old_array.end());
        }
        new_array.push_back(name);
        mlir::MLIRContext* ctx = op->getContext();
        op->setAttr("jasc_tags", mlir::ArrayAttr::get(ctx, new_array));
      });
    });

    getOperation().walk([](jasc::TagRegionOp tag_region_op) {
      mlir::Block& body = tag_region_op.getBody().front();
      mlir::Block& parent_block = *tag_region_op->getBlock();
      tag_region_op->replaceAllUsesWith(body.getTerminator()->getOperands());
      parent_block.getOperations().splice(
          mlir::Block::iterator(tag_region_op), body.getOperations(),
          body.begin(), mlir::Block::iterator(body.getTerminator()));
      tag_region_op.erase();
    });
  }
};

mlir::LogicalResult AllocRemoval(mlir::memref::CopyOp copy,
                                 mlir::PatternRewriter& rewriter) {
  mlir::Value from = copy.getSource();
  mlir::Value to = copy.getTarget();
  if (from.getDefiningOp() == nullptr) return mlir::failure();
  if (!llvm::isa<mlir::memref::AllocOp, mlir::gpu::AllocOp>(
          from.getDefiningOp())) {
    return mlir::failure();
  }

  // Only go up one level to grab the parent function; the match we're looking
  // for is at the very end of a function.
  auto func = llvm::dyn_cast_or_null<mlir::func::FuncOp>(copy->getParentOp());
  if (!func) {
    return mlir::failure();
  }

  // If the copy target is a function argument, use it directly.
  if (llvm::is_contained(func.getArguments(), to)) {
    rewriter.replaceAllUsesWith(from, to);
    rewriter.eraseOp(from.getDefiningOp());
    rewriter.eraseOp(copy);
    return mlir::success();
  }
  return mlir::failure();
}

class RemoveCopyToOutParamsPass
    : public mlir::PassWrapper<RemoveCopyToOutParamsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveCopyToOutParamsPass);

  // Returns the name of the pass suitable for the pass manager.
  llvm::StringRef getArgument() const override {
    return "jasc-remove-copy-to-out-params";
  }

  // Runs the pass.
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add(AllocRemoval);
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void AddStableHLOToLinalgPasses(mlir::PassManager& pm) {
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeHloToLinalgPass(true));
  pm.addPass(std::make_unique<LowerTagRegionsPass>());
}

void AddBufferizationPasses(
    mlir::PassManager& pm,
    mlir::bufferization::OneShotBufferizationOptions options,
    bool run_sparsification) {
  // TODO(ulysse): Avoid unnecessary copies introduced by bufferization.
  pm.addPass(mlir::createCSEPass());

  options.bufferizeFunctionBoundaries = true;
  mlir::bufferization::BufferResultsToOutParamsOptions out_params_options;
  if (run_sparsification) {
    // Setup both sparsification and bufferization.
    //
    // TODO(peiming, ajcbik, springerm): Make sparse compiler compatible with
    // one-shot bufferization. At the moment, they have to be intermixed, which
    // prevents us from running two passes independently and from sparsifying
    // kernel using transform IR.
    mlir::SparsificationOptions sparsification_options;
    sparsification_options.enableRuntimeLibrary = false;
    sparsification_options.enableIndexReduction = true;
    // Sparsification set up.
    // TODO(peiming, ajcbik): Maybe lift vectorization to transform IR instead?
    pm.addPass(mlir::createSparsificationAndBufferizationPass(
        options, sparsification_options,
        /*createSparseDeallocs=*/false,
        /*enableRuntimeLibrary=*/false,
        /*enableBufferInitialization=*/false,
        /*vectorLength=*/0,
        /*enableVLAVectorization=*/false,
        /*enableSIMDIndex32*/ false));
    pm.addPass(mlir::createStorageSpecifierToLLVMPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::bufferization::createFinalizingBufferizePass());
    // TODO(peiming, ajcbik): Find a way to avoid generating reallocations.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::memref::createExpandReallocPass(false));
  } else {
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
  }
  // Sparse compiler might insert extra function calls for complex operations.
  out_params_options.filterFn = [](mlir::func::FuncOp* func) {
    // Only transform the entry point.
    return func->getSymName() == "main";
  };
  pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass(
      out_params_options));
  pm.addPass(std::make_unique<RemoveCopyToOutParamsPass>());
  // TODO(mluecke): Add deallocation passes here when upstream problems are
  //                fixed
}

// No-op pass to register dialects needed for LLVM lowering.
class RegisterLLVMTranslationDialectsPass
    : public mlir::PassWrapper<RegisterLLVMTranslationDialectsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RegisterLLVMTranslationDialectsPass)

  // Returns the name of the pass suitable for the pass manager.
  llvm::StringRef getArgument() const override {
    return "jasc-register-llvm-translation-dialects";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    mlir::registerAllToLLVMIRTranslations(registry);
  }

  void runOnOperation() override {}
};

void AddLowerBufferizedLinalgToCF(mlir::PassManager& pm) {
  pm.addPass(std::make_unique<RegisterLLVMTranslationDialectsPass>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createLowerAffinePass());
}

void AddLowerCFToLLVMPasses(mlir::PassManager& pm) {
  mlir::ConvertVectorToLLVMPassOptions vector_to_llvm_opts;
  vector_to_llvm_opts.reassociateFPReductions = true;
  vector_to_llvm_opts.useOpaquePointers = true;
  pm.addPass(mlir::createConvertVectorToLLVMPass(vector_to_llvm_opts));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToLLVMPass());

  // Expand complicated MemRef operations before lowering them.
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  // The expansion may create affine expressions. Get rid of them.
  pm.addPass(mlir::createLowerAffinePass());

  mlir::FinalizeMemRefToLLVMConversionPassOptions memref_to_llvm_opts;
  // memref_to_llvm_opts.useOpaquePointers = true;
  pm.addPass(
      mlir::createFinalizeMemRefToLLVMConversionPass(memref_to_llvm_opts));

  mlir::ConvertFuncToLLVMPassOptions func_to_llvm_opts;
  // func_to_llvm_opts.useOpaquePointers = true;
  func_to_llvm_opts.useBarePtrCallConv = false;
  pm.addPass(mlir::createConvertFuncToLLVMPass(func_to_llvm_opts));
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

mlir::BaseMemRefType ConvertGpuArgType(
    mlir::TensorType tensor_type, mlir::Attribute, mlir::func::FuncOp,
    const mlir::bufferization::BufferizationOptions&) {
  // Override the memory space to global.
  auto memory_space = mlir::gpu::AddressSpaceAttr::get(
      tensor_type.getContext(), mlir::gpu::AddressSpace::Global);
  return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(
      tensor_type, memory_space);
}

class EraseTransformScriptPass
    : public mlir::PassWrapper<EraseTransformScriptPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseTransformScriptPass)

  // Returns the name of the pass suitable for the pass manager.
  llvm::StringRef getArgument() const override {
    return "jasc-erase-transform-script";
  }

  void runOnOperation() override {
    // Erase the first top-level transform so we can lower normally.
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [](mlir::transform::TransformOpInterface top_level_transform) {
          top_level_transform->erase();
          return mlir::WalkResult::interrupt();
        });
  }
};

void AddTransformInterpreterPasses(mlir::PassManager& pm) {
  pm.addPass(mlir::createLocationSnapshotPass());
  mlir::transform::TransformOptions transformOptions;
  transformOptions.enableEnforceSingleToplevelTransformOp(false);
  pm.addPass(std::make_unique<ApplyTransformScriptPass>(transformOptions));
  pm.addPass(std::make_unique<EraseTransformScriptPass>());
}

}  // namespace

absl::Status ApplyTransformScript(mlir::ModuleOp module, bool dump_ir) {
  mlir::PassManager pm(module.getContext());
  AddTransformInterpreterPasses(pm);
  return RunPassManager(pm, module, dump_ir);
}

absl::Status LowerStableHloToCpuLLVM(mlir::ModuleOp module, bool dump_ir) {
  mlir::PassManager pm(module.getContext());
  AddStableHLOToLinalgPasses(pm);
  AddTransformInterpreterPasses(pm);
  // Convert create_empty_tensor to allocs to ensure that they are not touched
  // by CSE. Maybe we can create them directly during transformations instead.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createEmptyTensorToAllocTensorPass());
  mlir::bufferization::OneShotBufferizationOptions bufferization_options;
  bufferization_options.setFunctionBoundaryTypeConversion(
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  AddBufferizationPasses(pm, bufferization_options,
                         /*run_sparsification=*/true);
  AddLowerBufferizedLinalgToCF(pm);
  AddLowerCFToLLVMPasses(pm);
  return RunPassManager(pm, module, dump_ir);
}

absl::Status LowerStableHloToGpuLLVM(mlir::ModuleOp module, bool dump_ir) {
#ifdef MLIR_GPU_TO_CUBIN_PASS_ENABLE
  mlir::PassManager pm(module.getContext());
  AddStableHLOToLinalgPasses(pm);
  AddTransformInterpreterPasses(pm);
  mlir::bufferization::OneShotBufferizationOptions bufferization_options;
  bufferization_options.allocationFn = &CreateGpuAlloc;
  bufferization_options.memCpyFn = &CreateGpuMemCpy;
  bufferization_options.functionArgTypeConverterFn = &ConvertGpuArgType;
  bufferization_options.inferFunctionResultLayout = false;

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(CreateSetDefaultGpuMemorySpacePass());

  AddBufferizationPasses(pm, bufferization_options,
                         /*run_sparsification=*/false);
  pm.addNestedPass<mlir::func::FuncOp>(CreateMemcpyToGpuDialectPass());
  pm.addPass(CreateApplyTransformScriptPass("gpu_post_bufferize.mlir"));
  AddLowerBufferizedLinalgToCF(pm);
  pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertGpuOpsToNVVMOps());

  // TODO(ulysse): see how much of the remaining can we share with GPUs.
  // Note: a lot of the GPU lowering code is hidden in GPUToLLVM.
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createLowerAffinePass());
  mlir::ConvertFuncToLLVMPassOptions func_to_llvm_opts;
  func_to_llvm_opts.useOpaquePointers = true;
  func_to_llvm_opts.useBarePtrCallConv = true;
  pm.addPass(mlir::createConvertFuncToLLVMPass(func_to_llvm_opts));
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createGpuSerializeToCubinPass(
      "nvptx64-nvidia-cuda", "sm_35", "+ptx60"));
  pm.addPass(CreateGpuToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  return RunPassManager(pm, module, dump_ir);
#else
  return absl::InternalError("MLIR_GPU_TO_CUBIN_PASS_ENABLE not defined");
#endif  // MLIR_GPU_TO_CUBIN_PASS_ENABLE
}

absl::Status LowerStableHloToLinalg(mlir::ModuleOp module, bool dump_ir) {
  mlir::PassManager pm(module.getContext());
  AddStableHLOToLinalgPasses(pm);
  AddTransformInterpreterPasses(pm);
  return RunPassManager(pm, module, dump_ir);
}

void registerMLIRLoweringPasses() {
  mlir::PassRegistration<ApplyTransformScriptPass>();
  mlir::PassRegistration<EraseTransformScriptPass>();
  mlir::PassRegistration<LowerTagRegionsPass>();
  mlir::PassRegistration<RegisterLLVMTranslationDialectsPass>();
  mlir::PassRegistration<RemoveCopyToOutParamsPass>();
}

}  // namespace jasc
