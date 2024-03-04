//===-- Export.h - Export Substrait dialect to protobuf ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_TARGET_SUBSTRAITPB_EXPORT_H
#define STRUCTURED_TARGET_SUBSTRAITPB_EXPORT_H

#include "llvm/Support/raw_ostream.h"

namespace mlir {
class Operation;
class LogicalResult;

namespace substrait {

LogicalResult translateSubstraitToProtobuf(Operation *op,
                                           llvm::raw_ostream &output);

} // namespace substrait
} // namespace mlir

#endif // STRUCTURED_TARGET_SUBSTRAITPB_EXPORT_H
