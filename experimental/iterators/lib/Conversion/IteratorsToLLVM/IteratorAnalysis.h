//===-- IteratorAnalysis.h - Lowering information of iterators --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ITERATORS_LIB_CONVERSION_ITERATORSTOLLVM_ITERATORANALYSIS_H
#define EXPERIMENTAL_ITERATORS_LIB_CONVERSION_ITERATORSTOLLVM_ITERATORANALYSIS_H

#include "iterators/Utils/NameAssigner.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
namespace mlir {
class ModuleOp;
class Operation;

namespace iterators {

class IteratorOpInterface;

/// Information about each iterator op constructed by IteratorAnalysis.
struct IteratorInfo {
  /// Takes the `LLVM::LLVMStructType` as a parameter, to ensure proper build
  /// order (all uses are visited before any def).
  IteratorInfo(IteratorOpInterface op, NameAssigner &nameAssigner,
               LLVM::LLVMStructType t);

  // Rule of five: default constructors/assignment operators
  IteratorInfo() = default;
  IteratorInfo(const IteratorInfo &other) = default;
  IteratorInfo(IteratorInfo &&other) = default;
  IteratorInfo &operator=(const IteratorInfo &other) = default;
  IteratorInfo &operator=(IteratorInfo &&other) = default;

  // Pre-assigned symbols that should be used for the Open/Next/Close
  // functions of this iterator.
  SymbolRefAttr openFunc;
  SymbolRefAttr nextFunc;
  SymbolRefAttr closeFunc;

  /// State of the iterator including the state of its potential upstreams.
  LLVM::LLVMStructType stateType;
};

/// Constructs information about the state type and the Open/Next/Close
/// functions of all iterator ops nested inside the given parent op.
/// The state type of each iterator usually consists of a private part, which
/// the iterator accesses in its Open/Next/Close logic, as well as the state of
/// all of its transitive upstream iterators, i.e., the iterators that produce
/// the operand streams. Ignores non-iterator ops, i.e., those that do not
/// implement IteratorOpInterface.
class IteratorAnalysis {
  using OperationMap = llvm::DenseMap<Operation *, IteratorInfo>;

public:
  explicit IteratorAnalysis(Operation *rootOp);

  /// Returns the operation this analysis was constructed from.
  Operation *getRootOperation() const { return rootOp; }

  /// Return the result of the analysis on `op`.
  /// Expects an entry for `op` to already exist.
  IteratorInfo getExpectedIteratorInfo(IteratorOpInterface op) const;

  /// Set the info for `op`.
  /// Expects an entry for `op` to **not** already exist.
  void setIteratorInfo(IteratorOpInterface op, const IteratorInfo &info);

private:
  /// Operation this analysis was constructed from.
  Operation *rootOp;

  /// Helper class that pre-assigns unique symbols for Open/Next/Close.
  /// The scope of unicity is the immediately enclosing ModuleOp.
  NameAssigner nameAssigner;

  /// Results of the analysis.
  OperationMap opMap;
};

} // namespace iterators
} // namespace mlir

#endif // EXPERIMENTAL_ITERATORS_LIB_CONVERSION_ITERATORSTOLLVM_ITERATORANALYSIS_H
