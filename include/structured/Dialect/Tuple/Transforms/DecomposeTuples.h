//===- DecomposeTuples.h - Pass Utilities -----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_DIALECT_TUPLE_TRANSFORMS_DECOMPOSETUPLES_H
#define STRUCTURED_DIALECT_TUPLE_TRANSFORMS_DECOMPOSETUPLES_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace mlir {
namespace tuple {

void populateDecomposeTuplesPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns);

} // namespace tuple
} // namespace mlir

#endif // STRUCTURED_DIALECT_TUPLE_TRANSFORMS_DECOMPOSETUPLES_H
