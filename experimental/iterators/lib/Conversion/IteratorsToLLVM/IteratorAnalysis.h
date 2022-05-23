//===-- IteratorAnalysis.h - Lowering information of iterators --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ITERATORS_LIB_CONVERSION_ITERATORSTOLLVM_ITERATORANALYSIS_H
#define EXPERIMENTAL_ITERATORS_LIB_CONVERSION_ITERATORSTOLLVM_ITERATORANALYSIS_H

#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Utils/NameAssigner.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class ModuleOp;
class Operation;

namespace iterators {

/// Constructs information about the state type and the Open/Next/Close
/// functions of all iterator ops nested inside the given parent op. The state
/// type of each iterator usually consists of a private part, which the iterator
/// accesses in its Open/Next/Close logic, as well as the state of all of its
/// transitive upstream iterators, i.e., the iterators that produce the operand
/// streams.
class IteratorAnalysis {
public:
  /// Information about each op constructed by this analysis.
  struct IteratorInfo {
    /// State of the iterator including the state of its potential upstreams.
    LLVM::LLVMStructType stateType;
    // Pre-assigned symbols that should be used for the Open/Next/Close
    // functions of this iterator.
    SymbolRefAttr openFunc;
    SymbolRefAttr nextFunc;
    SymbolRefAttr closeFunc;
  };

private:
  using OperationMap = llvm::DenseMap<Operation *, IteratorInfo>;

public:
  explicit IteratorAnalysis(Operation *parentOp, ModuleOp module);

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return parentOp; }

  /// Returns the information for the given op.
  llvm::Optional<IteratorInfo> getIteratorInfo(Operation *op) const;

private:
  /// Assembles all required information of a given iterator op.
  void buildIteratorInfo(Operation *op);

  /// Pre-assigns names for the Open/Next/Close functions of the given iterator
  /// op. The conversion is expected to create these names in the lowering of
  /// the corresponding op and can look them up in the lowering of downstream
  /// iterators.
  llvm::SmallVector<SymbolRefAttr, 3> assignFunctionNames(Operation *op);

  // Compute the state type of an op of a given type.
  LLVM::LLVMStructType computeStateType(SampleInputOp op);
  LLVM::LLVMStructType computeStateType(ReduceOp op);

  Operation *parentOp;
  OperationMap opMap;
  NameAssigner nameAssigner;
};

} // namespace iterators
} // namespace mlir

#endif // EXPERIMENTAL_ITERATORS_LIB_CONVERSION_ITERATORSTOLLVM_ITERATORANALYSIS_H
