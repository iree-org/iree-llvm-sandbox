//===-- DataFlowToIterators.h - Utils to convert from DataFlow --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_CONVERSION_DATAFLOWTOITERATORS_DATAFLOWTOITERATORS_H
#define ITERATORS_CONVERSION_DATAFLOWTOITERATORS_DATAFLOWTOITERATORS_H

#include "iterators/Dialect/DataFlow/IR/DataFlow.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;

namespace dataflow {

/// Populate the given list with patterns that convert from DataFlow to
/// Iterators.
void populateDataFlowToIteratorsConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter);

} // namespace dataflow

/// Create a pass to convert Iterators operations to the LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertDataFlowToIteratorsPass();

} // namespace mlir

#endif // ITERATORS_CONVERSION_DATAFLOWTOITERATORS_DATAFLOWTOITERATORS_H
