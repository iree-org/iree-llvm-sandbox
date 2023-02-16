//===-- ArrowUtils.cpp - Utils for converting Arrow to LLVM -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArrowUtils.h"

#include "iterators/Dialect/Iterators/IR/ArrowUtils.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::iterators;
using namespace mlir::LLVM;

namespace mlir {
namespace iterators {

LLVMFuncOp lookupOrInsertArrowArrayGetSize(ModuleOp module) {
  MLIRContext *context = module.getContext();
  Type i64 = IntegerType::get(context, 64);
  Type array = getArrowArrayType(context);
  Type arrayPtr = LLVMPointerType::get(array);
  return lookupOrCreateFn(module, "mlirIteratorsArrowArrayGetSize", {arrayPtr},
                          i64);
}

LLVMFuncOp lookupOrInsertArrowArrayGetColumn(ModuleOp module,
                                             Type elementType) {
  assert(elementType.isIntOrFloat() &&
         "only int or float types supported currently");
  MLIRContext *context = module.getContext();

  // Assemble types for signature.
  Type elementPtr = LLVMPointerType::get(elementType);
  Type i64 = IntegerType::get(context, 64);
  Type array = getArrowArrayType(context);
  Type arrayPtr = LLVMPointerType::get(array);
  Type schema = getArrowSchemaType(context);
  Type schemaPtr = LLVMPointerType::get(schema);

  // Assemble function name.
  StringRef typeNameBase;
  if (elementType.isSignedInteger() || elementType.isSignlessInteger())
    typeNameBase = "Int";
  else if (elementType.isUnsignedInteger())
    typeNameBase = "UInt";
  else {
    assert(elementType.isF16() || elementType.isF32() || elementType.isF64());
    typeNameBase = "Float";
  }
  std::string typeWidth = std::to_string(elementType.getIntOrFloatBitWidth());
  std::string funcName =
      ("mlirIteratorsArrowArrayGet" + typeNameBase + typeWidth + "Column")
          .str();

  // Lookup or insert function.
  return lookupOrCreateFn(module, funcName, {arrayPtr, schemaPtr, i64},
                          elementPtr);
}

LLVMFuncOp lookupOrInsertArrowArrayRelease(ModuleOp module) {
  MLIRContext *context = module.getContext();
  Type array = getArrowArrayType(context);
  Type arrayPtr = LLVMPointerType::get(array);
  Type voidType = LLVMVoidType::get(context);
  return lookupOrCreateFn(module, "mlirIteratorsArrowArrayRelease", {arrayPtr},
                          voidType);
}

LLVMFuncOp lookupOrInsertArrowSchemaRelease(ModuleOp module) {
  MLIRContext *context = module.getContext();
  Type schema = getArrowSchemaType(context);
  Type schemaPtr = LLVMPointerType::get(schema);
  Type voidType = LLVMVoidType::get(context);
  return lookupOrCreateFn(module, "mlirIteratorsArrowSchemaRelease",
                          {schemaPtr}, voidType);
}

LLVMFuncOp lookupOrInsertArrowArrayStreamGetSchema(ModuleOp module) {
  MLIRContext *context = module.getContext();
  Type arrayStream = getArrowArrayStreamType(context);
  Type arrayStreamPtr = LLVMPointerType::get(arrayStream);
  Type schema = getArrowSchemaType(context);
  Type schemaPtr = LLVMPointerType::get(schema);
  Type i32 = IntegerType::get(context, 32);
  return lookupOrCreateFn(module, "mlirIteratorsArrowArrayStreamGetSchema",
                          {arrayStreamPtr, schemaPtr}, i32);
}

LLVMFuncOp lookupOrInsertArrowArrayStreamGetNext(ModuleOp module) {
  MLIRContext *context = module.getContext();
  Type i1 = IntegerType::get(context, 1);
  Type stream = getArrowArrayStreamType(context);
  Type streamPtr = LLVMPointerType::get(stream);
  Type array = getArrowArrayType(context);
  Type arrayPtr = LLVMPointerType::get(array);
  return lookupOrCreateFn(module, "mlirIteratorsArrowArrayStreamGetNext",
                          {streamPtr, arrayPtr}, i1);
}

LLVMFuncOp lookupOrInsertArrowArrayStreamRelease(ModuleOp module) {
  MLIRContext *context = module.getContext();
  Type arrayStream = getArrowArrayStreamType(context);
  Type arrayStreamPtr = LLVMPointerType::get(arrayStream);
  Type voidType = LLVMVoidType::get(context);
  return lookupOrCreateFn(module, "mlirIteratorsArrowArrayStreamRelease",
                          {arrayStreamPtr}, voidType);
}

} // namespace iterators
} // namespace mlir
