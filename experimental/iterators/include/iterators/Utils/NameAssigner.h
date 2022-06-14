//===-- IteratorAnalysis.h - Lowering information of iterators --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_UTILS_NAMEASSIGNER_H
#define ITERATORS_UTILS_NAMEASSIGNER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

#include <stdint.h>

namespace mlir {
namespace iterators {

/// Pre-assigns unique symbol names in the given module. Uniqueness is
/// guaranteed among all symbols already existing in the module and those pre-
/// assigned through the same instance of this class assuming no other symbols
/// are assigned (at least not with colliding prefixes) even if the symbols are
/// not immediately registered in the module. An instance of this class
/// maintains the set of names it previously assigned and makes subsequent names
/// unique by appending a number to each given prefix, which it increments until
/// a non-colliding name is obtained.
class NameAssigner {
public:
  NameAssigner(ModuleOp module);

  /// Pre-assigns a name in the current module, making it unique if necessary.
  StringAttr assignName(StringRef prefix);

private:
  ModuleOp module;
  llvm::DenseSet<StringRef> names;
  uint64_t uniqueNumber = 0;
};

} // namespace iterators
} // namespace mlir

#endif // ITERATORS_UTILS_NAMEASSIGNER_H
