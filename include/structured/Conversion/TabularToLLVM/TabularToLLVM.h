//===-- TabularToLLVM.h - Utils to convert from Tabular ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_CONVERSION_TABULARTOLLVM_TABULARTOLLVM_H
#define ITERATORS_CONVERSION_TABULARTOLLVM_TABULARTOLLVM_H

#include <memory>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

namespace tabular {

/// Maps types from the Tabular dialect to corresponding types in LLVM.
class TabularTypeConverter : public TypeConverter {
public:
  TabularTypeConverter(LLVMTypeConverter &llvmTypeConverter);

  /// Maps a TabularViewType to an LLVMStruct of pointers, i.e., to a "struct of
  /// arrays".
  static Optional<Type> convertTabularViewType(Type type);

private:
  LLVMTypeConverter llvmTypeConverter;
};

/// Populate the given list with patterns that convert from Tabular to LLVM.
void populateTabularToLLVMConversionPatterns(RewritePatternSet &patterns,
                                             TypeConverter &typeConverter);

} // namespace tabular

/// Create a pass to convert Tabular operations to the LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTabularToLLVMPass();

} // namespace mlir

#endif // STRUCTURED_CONVERSION_TABULARTOLLVM_TABULARTOLLVM_H
