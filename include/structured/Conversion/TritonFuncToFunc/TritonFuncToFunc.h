//===-- TritonFuncToFunc.h - Convert Triton func ops to func ----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_CONVERSION_TRITONFUNCTOFUNC_TRITONFUNCTOFUNC_H
#define STRUCTURED_CONVERSION_TRITONFUNCTOFUNC_TRITONFUNCTOFUNC_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;
class TypeConverter;

/// Populate the given list with patterns that convert Triton func ops func.
void populateTritonFuncToFuncConversionPatterns(RewritePatternSet &patterns);

/// Create a pass to convert Triton func ops to the func dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonFuncToFuncPass();

} // namespace mlir

#endif // STRUCTURED_CONVERSION_TRITONFUNCTOFUNC_TRITONFUNCTOFUNC_H
