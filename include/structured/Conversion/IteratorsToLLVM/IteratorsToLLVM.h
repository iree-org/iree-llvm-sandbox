//===-- IteratorsToLLVM.h - Utils to convert from Iterators -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_CONVERSION_ITERATORSTOLLVM_ITERATORSTOLLVM_H
#define STRUCTURED_CONVERSION_ITERATORSTOLLVM_ITERATORSTOLLVM_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;
class TypeConverter;

namespace iterators {

/// Populate the given list with patterns that convert from Iterators to LLVM.
void populateIteratorsToLLVMConversionPatterns(RewritePatternSet &patterns,
                                               TypeConverter &typeConverter);

} // namespace iterators

/// Create a pass to convert Iterators operations to the LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertIteratorsToLLVMPass();

} // namespace mlir

#endif // STRUCTURED_CONVERSION_ITERATORSTOLLVM_ITERATORSTOLLVM_H
