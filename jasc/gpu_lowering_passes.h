//===-- gpu_lowering_passes.h - Passes for GPU lowering ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef JASC_GPU_LOWERING_PASSES_H_
#define JASC_GPU_LOWERING_PASSES_H_

#include <memory>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"

namespace jasc {

// Creates a pass that annotates the tensor alloc operations to use the global
// memory space. The pass also adds tensor alloc operations after constants to
// copy them to the global memory space.
std::unique_ptr<mlir::Pass> CreateSetDefaultGpuMemorySpacePass();

// Creates a custom version of GpuToLLVMConversionPass to support memory space
// annotations.
std::unique_ptr<mlir::Pass> CreateGpuToLLVMConversionPass();

// Creates a pass that converts memref.copy to the GPU dialect.
std::unique_ptr<mlir::Pass> CreateMemcpyToGpuDialectPass();

mlir::FailureOr<mlir::Value> CreateGpuAlloc(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            mlir::MemRefType memref_type,
                                            mlir::ValueRange dyn_sizes,
                                            unsigned int);

mlir::LogicalResult CreateGpuDealloc(mlir::OpBuilder& builder,
                                     mlir::Location loc, mlir::Value memref);

mlir::LogicalResult CreateGpuMemCpy(mlir::OpBuilder& builder,
                                    mlir::Location loc, mlir::Value from,
                                    mlir::Value to);

void registerGPULoweringPasses();

}  // namespace jasc

#endif  // JASC_GPU_LOWERING_PASSES_H_