//===-- gpu_lowering_passes.cc - Passes for GPU lowerings -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <utility>

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace jasc {

mlir::FailureOr<mlir::Value> CreateGpuAlloc(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            mlir::MemRefType memref_type,
                                            mlir::ValueRange dyn_sizes,
                                            unsigned int) {
  // TODO(ulysse): See if we can simplify this code. Synchronization tokens
  // are only needed because the GpuToLLVM pass expects them.
  auto token_type = mlir::gpu::AsyncTokenType::get(builder.getContext());
  auto wait_op =
      builder.create<mlir::gpu::WaitOp>(loc, token_type, mlir::ValueRange());
  auto alloc_op = builder.create<mlir::gpu::AllocOp>(
      loc, mlir::TypeRange({memref_type, token_type}), wait_op.getResults(),
      dyn_sizes, mlir::ValueRange());
  return alloc_op.getMemref();
}

mlir::LogicalResult CreateGpuDealloc(mlir::OpBuilder& builder,
                                     mlir::Location loc, mlir::Value memref) {
  // TODO(ulysse): See if we can simplify this code. Synchronization tokens
  // are only needed because the GpuToLLVM pass expects them.
  auto token_type = mlir::gpu::AsyncTokenType::get(builder.getContext());
  auto sync_op =
      builder.create<mlir::gpu::WaitOp>(loc, token_type, mlir::ValueRange());
  builder.create<mlir::gpu::DeallocOp>(loc, mlir::TypeRange({token_type}),
                                       sync_op.getResults(), memref);
  return mlir::success();
}

mlir::LogicalResult CreateGpuMemCpy(mlir::OpBuilder& builder,
                                    mlir::Location loc, mlir::Value from,
                                    mlir::Value to) {
  // TODO(ulysse): See if we can simplify this code. Synchronization tokens
  // are only needed because the GpuToLLVM pass expects them.
  auto token_type = mlir::gpu::AsyncTokenType::get(builder.getContext());
  auto sync_op =
      builder.create<mlir::gpu::WaitOp>(loc, token_type, mlir::ValueRange());
  builder.create<mlir::gpu::MemcpyOp>(loc, token_type, sync_op.getResults(), to,
                                      from);
  return mlir::success();
}

// Convert "memref.alloc" with no deallocation into "memref.alloca".
mlir::LogicalResult AllocToAlloca(mlir::memref::AllocOp alloc,
                                  mlir::PatternRewriter& rewriter) {
  for (mlir::Operation* user : alloc->getUsers()) {
    if (llvm::isa<mlir::memref::DeallocOp>(user)) return mlir::failure();
  }
  rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(alloc, alloc.getType(),
                                                      alloc.getDynamicSizes(),
                                                      alloc.getAlignmentAttr());
  return mlir::success();
}

namespace {

// Annotates the tensor alloc operations to use the global memory space. Add
// tensor alloc operations after constants to copy them to the global memory
// space.
class SetDefaultGpuMemorySpace
    : public mlir::PassWrapper<SetDefaultGpuMemorySpace,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetDefaultGpuMemorySpace)

  // Returns the name of the pass suitable for the pass manager.
  llvm::StringRef getArgument() const override {
    return "jasc-set-default-gpu-memory-space";
  }

  void runOnOperation() override {
    getOperation().walk([](mlir::arith::ConstantOp constant) {
      auto tensor_type = constant.getType().dyn_cast<mlir::RankedTensorType>();
      if (tensor_type == nullptr) return;

      mlir::OpBuilder builder(constant.getContext());
      builder.setInsertionPointAfter(constant);
      auto alloc_op = builder.create<mlir::bufferization::AllocTensorOp>(
          constant.getLoc(), tensor_type, mlir::ValueRange(),
          constant.getResult());
      constant.getResult().replaceAllUsesExcept(alloc_op.getResult(), alloc_op);
    });
    getOperation().walk([](mlir::bufferization::AllocTensorOp alloc) {
      alloc.setMemorySpaceAttr(mlir::gpu::AddressSpaceAttr::get(
          alloc->getContext(), mlir::gpu::AddressSpace::Global));
    });
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::bufferization::BufferizationDialect,
                    mlir::gpu::GPUDialect, mlir::arith::ArithDialect>();
  }
};

// Custom version of GpuToLLVMConversionPass to support memory space
// annotations.
class GpuToLLVMConversionPass
    : public mlir::PassWrapper<GpuToLLVMConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuToLLVMConversionPass)

  // Returns the name of the pass suitable for the pass manager.
  llvm::StringRef getArgument() const override {
    return "jasc-gpu-to-llvm-conversion";
  }

  void runOnOperation() override {
    mlir::LowerToLLVMOptions options(&getContext());
    options.useOpaquePointers = true;
    options.useBarePtrCallConv = true;

    mlir::LLVMTypeConverter converter(&getContext(), options);
    converter.addTypeAttributeConversion(
        [](mlir::BaseMemRefType type,
           mlir::gpu::AddressSpaceAttr memory_space) {
          // Erase memory space information.
          auto int_type = mlir::IntegerType::get(type.getContext(), 64);
          return mlir::IntegerAttr::get(int_type, 0);
        });

    mlir::LLVMConversionTarget target(getContext());
    target.addIllegalDialect<mlir::gpu::GPUDialect>();
    mlir::RewritePatternSet patterns(&getContext());
    mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

    // TODO(ulysse): change some of the patterns to avoid creating new streams.
    // Use a different converter for GPU dialect calls as we don't want to use
    // bare pointers here.
    mlir::LowerToLLVMOptions gpu_options(&getContext());
    gpu_options.useOpaquePointers = true;
    gpu_options.useBarePtrCallConv = false;
    mlir::LLVMTypeConverter gpu_converter(&getContext(), gpu_options);
    gpu_converter.addTypeAttributeConversion(
        [](mlir::BaseMemRefType type,
           mlir::gpu::AddressSpaceAttr memory_space) {
          // Erase memory space information.
          auto int_type = mlir::IntegerType::get(type.getContext(), 64);
          return mlir::IntegerAttr::get(int_type, 0);
        });

    mlir::populateGpuToLLVMConversionPatterns(
        gpu_converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

mlir::LogicalResult MemcpyToGpuPattern(mlir::memref::CopyOp copy,
                                       mlir::PatternRewriter& rewriter) {
  if (copy->getParentOfType<mlir::gpu::GPUModuleOp>() ||
      copy->getParentOfType<mlir::gpu::LaunchOp>())
    return mlir::failure();
  mlir::LogicalResult result = CreateGpuMemCpy(
      rewriter, copy.getLoc(), copy.getSource(), copy.getTarget());
  if (mlir::failed(result)) return mlir::failure();
  rewriter.eraseOp(copy);
  return mlir::success();
}

class MemcpyToGpuDialect
    : public mlir::PassWrapper<MemcpyToGpuDialect,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemcpyToGpuDialect)

  // Returns the name of the pass suitable for the pass manager.
  llvm::StringRef getArgument() const override {
    return "jasc-memcpy-to-gpu-dialect";
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add(MemcpyToGpuPattern);
    patterns.add(AllocToAlloca);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateSetDefaultGpuMemorySpacePass() {
  return std::make_unique<SetDefaultGpuMemorySpace>();
}

std::unique_ptr<mlir::Pass> CreateGpuToLLVMConversionPass() {
  return std::make_unique<GpuToLLVMConversionPass>();
}

std::unique_ptr<mlir::Pass> CreateMemcpyToGpuDialectPass() {
  return std::make_unique<MemcpyToGpuDialect>();
}

void registerGPULoweringPasses() {
  mlir::PassRegistration<GpuToLLVMConversionPass>();
  mlir::PassRegistration<MemcpyToGpuDialect>();
  mlir::PassRegistration<SetDefaultGpuMemorySpace>();
}

}  // namespace jasc
