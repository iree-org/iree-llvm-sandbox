//===-- Iterators.cpp - Iterators dialect -----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/Iterators.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iterators;

//===----------------------------------------------------------------------===//
// Iterators dialect
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/IteratorsOpsDialect.cpp.inc"

void IteratorsDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "iterators/Dialect/Iterators/IR/IteratorsOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Iterators interfaces
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/IteratorsOpInterfaces.cpp.inc"
#include "iterators/Dialect/Iterators/IR/IteratorsTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Iterators operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOps.cpp.inc"

LogicalResult ViewAsColumnarBatchOp::verify() {
  auto batchType = batch().getType().cast<ColumnarBatchType>();
  TupleType tupleType = batchType.getElementType();

  // Verify matching element types.
  if (!llvm::all_of_zip(memrefs().getTypes(), tupleType.getTypes(),
                        [](Type t1, Type t2) {
                          return t1.cast<MemRefType>().getElementType() == t2;
                        })) {

    std::string batchElementTypes;
    {
      llvm::raw_string_ostream stream(batchElementTypes);
      llvm::interleaveComma(tupleType.getTypes(), stream,
                            [&](Type type) { type.print(stream); });
    }
    std::string memrefElementTypes;
    {
      llvm::raw_string_ostream stream(memrefElementTypes);
      llvm::interleaveComma(memrefs().getTypes(), stream, [&](Type type) {
        type.cast<MemRefType>().getElementType().print(stream);
      });
    }
    return emitOpError()
           << "type mismatch: should return a batch of tuples consisting of "
              "the element types of the input memrefs (namely "
           << "tuple<" << memrefElementTypes << ">) "
           << "but returns a batch of "
           << "tuple<" << batchElementTypes << ">.";
  }

  {
    for (const auto &indexedMemref : llvm::enumerate(memrefs().getTypes())) {
      // Must have static rank = 1.
      auto memrefType = indexedMemref.value().cast<MemRefType>();
      if (memrefType.getRank() != 1)
        return emitOpError()
               << "unsupported type: input memref #" << indexedMemref.index()
               << " has rank != 1, which is not supported.";

      // Only support contiguous memory layout.
      if (!isStaticShapeAndContiguousRowMajor(memrefType))
        return emitOpError()
               << "unsupported type: input memref #" << indexedMemref.index()
               << " does not have a static shape and a contiguous memory "
                  "layout, both of which are required.";
    }

    // Verify all memrefs are of equal static length.
    if (!llvm::is_splat(llvm::map_range(memrefs().getTypes(), [](Type t) {
          return t.cast<MemRefType>().getDimSize(0);
        }))) {
      std::string lengths;
      {
        llvm::raw_string_ostream stream(lengths);
        llvm::interleaveComma(memrefs().getTypes(), stream, [&](Type type) {
          stream << type.cast<MemRefType>().getDimSize(0);
        });
      }
      return emitOpError()
             << "type mismatch: input memrefs cannot have different static "
                "shapes (sizes found for dimension 0: "
             << lengths << ").";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Iterators types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.cpp.inc"
