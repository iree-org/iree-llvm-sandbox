//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_MODELBUILDER_VULKANWRAPPERPASS_H_
#define IREE_LLVM_SANDBOX_MODELBUILDER_VULKANWRAPPERPASS_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace spirv {
class FuncOp;
}

class ModuleOp;
template <typename T>
class OperationPass;
/// Create a c interface function wrapping a vulkan dispatch for the existing
/// GPU module.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createAddVulkanLaunchWrapperPass(
    llvm::ArrayRef<int64_t> workloadSize, llvm::ArrayRef<Type> args);

/// Set SPIRV ABI for kernel arguments. This hardcode the binding information
/// to be able to wok with vulkan runner.
std::unique_ptr<OperationPass<spirv::FuncOp>> createSetSpirvABIPass();
}  // namespace mlir

#endif  // IREE_LLVM_SANDBOX_MODELBUILDER_VULKANWRAPPERPASS_H_
