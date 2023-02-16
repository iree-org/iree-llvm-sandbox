//===-- ArrowUtils.cpp - IR utils related to Apache Arrow  ------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/ArrowUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace mlir {
namespace iterators {

LLVMStructType getArrowArrayType(MLIRContext *context) {
  LLVMStructType arrowArray =
      LLVMStructType::getIdentified(context, "ArrowArray");
  if (arrowArray.isInitialized())
    return arrowArray;

  Type voidPtr = LLVMPointerType::get(context);
  Type i64 = IntegerType::get(context, 64);
  Type arrowArrayPtr = LLVMPointerType::get(arrowArray);
  auto voidType = LLVMVoidType::get(context);
  Type releaseFunc = LLVMFunctionType::get(voidType, arrowArrayPtr);

  ArrayRef<Type> body = {
      i64 /*length*/,
      i64 /*null_count*/,
      i64 /*offset*/,
      i64 /*n_buffers*/,
      i64 /*n_children*/,
      LLVMPointerType::get(voidPtr) /*buffers*/,
      LLVMPointerType::get(arrowArrayPtr) /*children*/,
      arrowArrayPtr /*dictionary*/,
      LLVMPointerType::get(releaseFunc) /*release*/,
      voidPtr /*private_data*/
  };

  LogicalResult status = arrowArray.setBody(body, /*isPacked=*/false);
  assert(succeeded(status) && "could not create ArrowArray struct");
  return arrowArray;
}

LLVMStructType getArrowSchemaType(MLIRContext *context) {
  auto arrowSchema = LLVMStructType::getIdentified(context, "ArrowSchema");
  if (arrowSchema.isInitialized())
    return arrowSchema;

  Type charPtr = LLVMPointerType::get(IntegerType::get(context, 8));
  Type voidPtr = LLVMPointerType::get(context);
  Type i64 = IntegerType::get(context, 64);
  Type arrowSchemaPtr = LLVMPointerType::get(arrowSchema);
  auto voidType = LLVMVoidType::get(context);
  Type releaseFunc = LLVMFunctionType::get(voidType, arrowSchemaPtr);

  ArrayRef<Type> body{
      charPtr /*format*/,
      charPtr /*name*/,
      charPtr /*metadata*/,
      i64 /*flags*/,
      i64 /*n_children*/,
      LLVMPointerType::get(arrowSchemaPtr) /*children*/,
      arrowSchemaPtr /*dictionary*/,
      LLVMPointerType::get(releaseFunc) /*release*/,
      voidPtr /*private_data*/
  };

  LogicalResult status = arrowSchema.setBody(body, /*isPacked=*/false);
  assert(succeeded(status) && "could not create ArrowSchema struct");
  return arrowSchema;
}

LLVMStructType getArrowArrayStreamType(MLIRContext *context) {
  auto arrowArrayStream =
      LLVMStructType::getIdentified(context, "ArrowArrayStream");
  if (arrowArrayStream.isInitialized())
    return arrowArrayStream;

  Type voidPtr = LLVMPointerType::get(context);
  Type charPtr = LLVMPointerType::get(IntegerType::get(context, 8));
  Type i32 = IntegerType::get(context, 32);
  auto voidType = LLVMVoidType::get(context);
  Type arrowArray = getArrowArrayType(context);
  auto arrowSchema = getArrowSchemaType(context);
  Type arrowArrayPtr = LLVMPointerType::get(arrowArray);
  Type arrowSchemaPtr = LLVMPointerType::get(arrowSchema);
  Type arrowArrayStreamPtr = LLVMPointerType::get(arrowArrayStream);

  Type getSchemaFunc =
      LLVMFunctionType::get(i32, {arrowArrayStreamPtr, arrowSchemaPtr});
  Type getNextFunc =
      LLVMFunctionType::get(i32, {arrowArrayStreamPtr, arrowArrayPtr});
  Type getLastErrorFunc = LLVMFunctionType::get(charPtr, arrowArrayStreamPtr);
  Type releaseFunc = LLVMFunctionType::get(voidType, arrowArrayStreamPtr);

  ArrayRef<Type> body{
      LLVMPointerType::get(getSchemaFunc) /*get_schema*/,
      LLVMPointerType::get(getNextFunc) /*get_next*/,
      LLVMPointerType::get(getLastErrorFunc) /*get_last_error*/,
      LLVMPointerType::get(releaseFunc) /*release*/, voidPtr /*private_data*/
  };

  LogicalResult status = arrowArrayStream.setBody(body, /*isPacked=*/false);
  assert(succeeded(status) && "could not create ArrowArrayStream struct");
  return arrowArrayStream;
}

} // namespace iterators
} // namespace mlir
