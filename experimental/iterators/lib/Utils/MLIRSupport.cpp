//===-- MLIRSupport.cpp - Utils for dealing with MLIR -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Utils/MLIRSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class NamedAttribute;
} // namespace mlir

using namespace mlir;
using namespace iterators::utils;

namespace iterators {
namespace utils {

scf::WhileOp
createWhileOp(OpBuilder &builder, Location loc, TypeRange resultTypes,
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
      builder.createBlock(&op.getBefore(), {}, operands, beforeLocs);
  beforeBuilder(builder, loc, before->getArguments());

  // After region.
  llvm::SmallVector<Location, 4> afterLocs(resultTypes.size(), loc);
  Block *after =
      builder.createBlock(&op.getAfter(), {}, resultTypes, afterLocs);
  afterBuilder(builder, loc, after->getArguments());

  return op;
}

} // namespace utils
} // namespace iterators
