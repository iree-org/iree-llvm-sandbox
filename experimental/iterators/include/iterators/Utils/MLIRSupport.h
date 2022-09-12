//===-- MLIRSupport.h - Utils for dealing with MLIR -------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_UTILS_MLIR_SUPPORT_H
#define ITERATORS_UTILS_MLIR_SUPPORT_H

#include "mlir/IR/Block.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
class NamedAttribute;
class OpBuilder;
class ImplicitLocOpBuilder;
} // namespace mlir

namespace mlir {
namespace scf {
class WhileOp;

WhileOp createWhileOp(
    OpBuilder &builder, Location loc, TypeRange resultTypes,
    ValueRange operands,
    llvm::function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        beforeBuilder,
    llvm::function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        afterBuilder,
    llvm::ArrayRef<NamedAttribute> attributes = {});

WhileOp createWhileOp(
    ImplicitLocOpBuilder &builder, TypeRange resultTypes, ValueRange operands,
    llvm::function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        beforeBuilder,
    llvm::function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        afterBuilder,
    llvm::ArrayRef<NamedAttribute> attributes = {});

} // namespace scf

namespace LLVM {
class InsertValueOp;
class ExtractValueOp;

InsertValueOp createInsertValueOp(OpBuilder &builder, Location loc,
                                  Value container, Value value,
                                  ArrayRef<int64_t> position);

InsertValueOp createInsertValueOp(ImplicitLocOpBuilder &builder,
                                  Value container, Value value,
                                  ArrayRef<int64_t> position);

ExtractValueOp createExtractValueOp(OpBuilder &builder, Location loc, Type res,
                                    Value container,
                                    ArrayRef<int64_t> position);

ExtractValueOp createExtractValueOp(ImplicitLocOpBuilder &builder, Type res,
                                    Value container,
                                    ArrayRef<int64_t> position);

} // namespace LLVM
} // namespace mlir

#endif // ITERATORS_UTILS_MLIR_SUPPORT_H
