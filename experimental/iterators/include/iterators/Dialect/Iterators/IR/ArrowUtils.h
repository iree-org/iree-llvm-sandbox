//===-- ArrowUtils.h - IR utils related to Apache Arrow  --------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ARROWUTILS_H
#define ITERATORS_DIALECT_ITERATORS_IR_ARROWUTILS_H

namespace mlir {
class MLIRContext;
namespace LLVM {
class LLVMStructType;
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace iterators {

/// Returns the LLVM struct type for Arrow arrays of the Arrow C data interface.
/// For the official definition of the type, see
/// https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions.
LLVM::LLVMStructType getArrowArrayType(MLIRContext *context);

/// Returns the LLVM struct type for Arrow schemas of the Arrow C data
/// interface. For the official definition of the type, see
/// https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions.
LLVM::LLVMStructType getArrowSchemaType(MLIRContext *context);

/// Returns the LLVM struct type for Arrow streams of the Arrow C stream
/// interface. For the official definition of the type, see
/// https://arrow.apache.org/docs/format/CStreamInterface.html#structure-definition.
LLVM::LLVMStructType getArrowArrayStreamType(MLIRContext *context);

} // namespace iterators
} // namespace mlir

#endif // ITERATORS_DIALECT_ITERATORS_IR_ARROWUTILS_H
