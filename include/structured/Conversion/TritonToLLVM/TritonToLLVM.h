//===-- TritonToLLVM.h - Utils to convert from Triton to LLVM ---*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_CONVERSION_TRITONTOLLVM_TRITONTOLLVM_H
#define STRUCTURED_CONVERSION_TRITONTOLLVM_TRITONTOLLVM_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;
class LLVMTypeConverter;

/// Populate the given list with patterns that convert from Triton to LLVM.
void populateTritonToLLVMConversionPatterns(RewritePatternSet &patterns,
                                            LLVMTypeConverter &typeConverter);

/// Create a pass to convert Triton operations to the LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToLLVMPass();

} // namespace mlir

#endif // STRUCTURED_CONVERSION_TRITONTOLLVM_TRITONTOLLVM_H
