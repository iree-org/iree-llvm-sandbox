//===-- MLIRSupport.cpp - Utils for dealing with MLIR -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Utils/MLIRSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

LLVM::LLVMFuncOp mlir::LLVM::lookupOrCreateFn(ModuleOp moduleOp, StringRef name,
                                              ArrayRef<Type> paramTypes,
                                              Type resultType, bool isVarArg) {
  auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (func)
    return func;
  OpBuilder b(moduleOp.getBodyRegion());
  return b.create<LLVM::LLVMFuncOp>(
      moduleOp->getLoc(), name,
      LLVM::LLVMFunctionType::get(resultType, paramTypes, isVarArg));
}

// TODO: Move builder to core MLIR.
scf::WhileOp mlir::scf::createWhileOp(
    OpBuilder &builder, Location loc, TypeRange resultTypes,
    ValueRange operands,
    function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        beforeBuilder,
    function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        afterBuilder,
    ArrayRef<NamedAttribute> attributes) {
  auto op =
      builder.create<scf::WhileOp>(loc, resultTypes, operands, attributes);
  OpBuilder::InsertionGuard guard(builder);

  // Before region.
  llvm::SmallVector<Location, 4> beforeLocs(operands.size(), loc);
  Block *before =
      builder.createBlock(&op.getBefore(), {}, operands.getTypes(), beforeLocs);
  beforeBuilder(builder, loc, before->getArguments());

  // After region.
  llvm::SmallVector<Location, 4> afterLocs(resultTypes.size(), loc);
  Block *after =
      builder.createBlock(&op.getAfter(), {}, resultTypes, afterLocs);
  afterBuilder(builder, loc, after->getArguments());

  return op;
}

scf::WhileOp mlir::scf::createWhileOp(
    ImplicitLocOpBuilder &builder, TypeRange resultTypes, ValueRange operands,
    function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        beforeBuilder,
    function_ref<void(OpBuilder &, Location, Block::BlockArgListType)>
        afterBuilder,
    ArrayRef<NamedAttribute> attributes) {
  return createWhileOp(builder, builder.getLoc(), resultTypes, operands,
                       beforeBuilder, afterBuilder, attributes);
}

LLVM::InsertValueOp
mlir::LLVM::createInsertValueOp(OpBuilder &builder, Location loc,
                                Value container, Value value,
                                ArrayRef<int64_t> position) {
  // Create index attribute.
  DenseI64ArrayAttr positionAttr = builder.getDenseI64ArrayAttr(position);

  // Insert into struct.
  return builder.create<LLVM::InsertValueOp>(loc, container, value,
                                             positionAttr);
}

LLVM::InsertValueOp
mlir::LLVM::createInsertValueOp(ImplicitLocOpBuilder &builder, Value container,
                                Value value, ArrayRef<int64_t> position) {
  return createInsertValueOp(builder, builder.getLoc(), container, value,
                             position);
}

LLVM::ExtractValueOp
mlir::LLVM::createExtractValueOp(OpBuilder &builder, Location loc, Type res,
                                 Value container, ArrayRef<int64_t> position) {
  // Create index attribute.
  DenseI64ArrayAttr positionAttr = builder.getDenseI64ArrayAttr(position);

  // Extract from struct.
  return builder.create<LLVM::ExtractValueOp>(loc, res, container,
                                              positionAttr);
}

LLVM::ExtractValueOp
mlir::LLVM::createExtractValueOp(ImplicitLocOpBuilder &builder, Type res,
                                 Value container, ArrayRef<int64_t> position) {
  return createExtractValueOp(builder, builder.getLoc(), res, container,
                              position);
}
