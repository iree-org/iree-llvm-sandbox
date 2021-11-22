//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ModelBuilder.h
// -----------------------------------------------------------------------------
//
// MLIR Model Builders demonstrate C++ metaprogramming features that are
// available in MLIR core. At a high-level, metaprogramming can be interpreted
// as "program with a level of indirection": one writes C++ that emits MLIR.
// The MLIR is then JIT compiled into a binary that can be invoked.
//
// The ModelBuilder exposes relevant core MLIR classes and APIs that are
// sufficient to build whole models. This set of classes and APIs encompass:
//  1. mlir::FuncOp creation.
//  2. key types creation such as mlir::FloatType, mlir::IntegerType,
//     mlir::VectorType, and mlir::MemRefType.
//  3. layer creation functions such as FCBiasTanh.
//
// Usage:
// ======
//
// ```
//
//    ModelBuilder builder;
//    auto func = builder.makeFunction(...);
//    OpBuilder b(&func.getBody());
//    ScopedContext scope(b, func.getLoc());
//
//    // ... build the body of func ...
//
//    builder.getOperation().print(llvm::outs()); // print MLIR
// ```

#ifndef IREE_LLVM_SANDBOX_MODELBUILDER_MODELBUILDER_H_
#define IREE_LLVM_SANDBOX_MODELBUILDER_MODELBUILDER_H_

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

namespace edsc {
/// Helper class to transparently handle builder insertion points by RAII.
/// As its name indicates, a ScopedContext is means to be used locally in a
/// scoped fashion. This abstracts away all the boilerplate related to
/// checking proper usage of captures, NestedBuilders as well as handling the
/// setting and restoring of insertion points.
class ScopedContext {
 public:
  ScopedContext(OpBuilder &b);
  ScopedContext(OpBuilder &b, Location location);

  /// Sets the insertion point of the builder to 'newInsertPt' for the duration
  /// of the scope. The existing insertion point of the builder is restored on
  /// destruction.
  ScopedContext(OpBuilder &b, OpBuilder::InsertPoint newInsertPt,
                Location location);
  ~ScopedContext();

  static MLIRContext *getContext();
  static OpBuilder &getBuilderRef();
  static Location getLocation();

 private:
  /// Only NestedBuilder (which is used to create an operation with a body)
  /// may access private members in order to implement scoping.
  friend class NestedBuilder;

  ScopedContext() = delete;
  ScopedContext(const ScopedContext &) = delete;
  ScopedContext &operator=(const ScopedContext &) = delete;

  static ScopedContext *&getCurrentScopedContext();

  /// Top level OpBuilder.
  OpBuilder &builder;
  /// Guard to the previous insertion point.
  OpBuilder::InsertionGuard guard;
  /// Current location.
  Location location;
  /// Parent context we return into.
  ScopedContext *enclosingScopedContext;
};

template <typename Op>
struct ValueBuilder {
  template <typename... Args>
  ValueBuilder(Args... args) {
    value = ScopedContext::getBuilderRef()
                .create<Op>(ScopedContext::getLocation(), args...)
                .getResult();
  }
  operator Value() { return value; }
  Value value;
};

template <typename Op>
struct OperationBuilder {
  template <typename... Args>
  OperationBuilder(Args... args) {
    op = ScopedContext::getBuilderRef().create<Op>(ScopedContext::getLocation(),
                                                   args...);
  }
  operator Op() { return op; }
  operator Operation *() { return op.getOperation(); }
  Op op;
};

/// Creates a block in the region that contains the insertion block of the
/// OpBuilder currently at the top of ScopedContext stack (appends the block to
/// the region). Be aware that this will NOT update the insertion point of the
/// builder to insert into the newly constructed block.
Block *createBlock(TypeRange argTypes = llvm::None);

/// Creates a block in the specified region using OpBuilder at the top of
/// ScopedContext stack (appends the block to the region). Be aware that this
/// will NOT update the insertion point of the builder to insert into the newly
/// constructed block.
Block *createBlockInRegion(Region &region, TypeRange argTypes = llvm::None);

/// Calls "builderFn" with ScopedContext reconfigured to insert into "block" and
/// passes in the block arguments. If the block has a terminator, the operations
/// are inserted before the terminator, otherwise appended to the block.
void appendToBlock(Block *block, function_ref<void(ValueRange)> builderFn);

/// Creates a block in the region that contains the insertion block of the
/// OpBuilder currently at the top of ScopedContext stack, and calls "builderFn"
/// to populate the body of the block while passing it the block arguments.
Block *buildInNewBlock(TypeRange argTypes,
                       function_ref<void(ValueRange)> builderFn);

/// Creates a block in the specified region using OpBuilder at the top of
/// ScopedContext stack, and calls "builderFn" to populate the body of the block
/// while passing it the block arguments.
Block *buildInNewBlock(Region &region, TypeRange argTypes,
                       function_ref<void(ValueRange)> builderFn);

/// A StructuredIndexed represents an indexable quantity that is either:
/// 1. a captured value, which is suitable for buffer and tensor operands, or;
/// 2. a captured type, which is suitable for tensor return values.
///
/// A StructuredIndexed itself is indexed and passed to `makeGenericLinalgOp`.
/// It enable an idiomatic syntax for index expressions such as:
///
/// ```
///      StructuredIndexed A(buffer_or_tensor_value), B(buffer_or_tensor_value),
///        C(buffer_value_or_tensor_type);
///      makeGenericLinalgOp({A({m, n}), B({k, n})}, {C({m, n})}, ... );
/// ```
struct StructuredIndexed {
  StructuredIndexed(Value v) : value(v) {}
  StructuredIndexed(Type t) : type(t) {}
  StructuredIndexed operator()(ArrayRef<AffineExpr> indexings) {
    return value ? StructuredIndexed(value, indexings)
                 : StructuredIndexed(type, indexings);
  }

  StructuredIndexed(Value v, ArrayRef<AffineExpr> indexings)
      : value(v), exprs(indexings.begin(), indexings.end()) {
    assert((v.getType().isa<MemRefType, RankedTensorType, VectorType>()) &&
           "MemRef, RankedTensor or Vector expected");
  }
  StructuredIndexed(Type t, ArrayRef<AffineExpr> indexings)
      : type(t), exprs(indexings.begin(), indexings.end()) {
    assert((t.isa<MemRefType, RankedTensorType, VectorType>()) &&
           "MemRef, RankedTensor or Vector expected");
  }

  bool hasValue() const { return (bool)value; }
  Value getValue() const {
    assert(value && "StructuredIndexed Value not set.");
    return value;
  }
  Type getType() const {
    assert((value || type) && "StructuredIndexed Value and Type not set.");
    return value ? value.getType() : type;
  }
  ArrayRef<AffineExpr> getExprs() const { return exprs; }
  operator Value() const { return getValue(); }
  operator Type() const { return getType(); }

 private:
  // Only one of Value or type may be set.
  Type type;
  Value value;
  SmallVector<AffineExpr, 4> exprs;
};

namespace op {

Value operator+(Value lhs, Value rhs);
Value operator-(Value lhs, Value rhs);
Value operator*(Value lhs, Value rhs);
Value operator/(Value lhs, Value rhs);
Value operator%(Value lhs, Value rhs);
Value floorDiv(Value lhs, Value rhs);
Value ceilDiv(Value lhs, Value rhs);

/// Logical operator overloadings.
Value negate(Value value);
Value operator&&(Value lhs, Value rhs);
Value operator||(Value lhs, Value rhs);
Value operator^(Value lhs, Value rhs);

/// Comparison operator overloadings.
Value eq(Value lhs, Value rhs);
Value ne(Value lhs, Value rhs);
Value slt(Value lhs, Value rhs);
Value sle(Value lhs, Value rhs);
Value sgt(Value lhs, Value rhs);
Value sge(Value lhs, Value rhs);
Value ult(Value lhs, Value rhs);
Value ule(Value lhs, Value rhs);
Value ugt(Value lhs, Value rhs);
Value uge(Value lhs, Value rhs);

} // namespace op

/// A TemplatedIndexedValue brings an index notation over the template Load and
/// Store parameters. Assigning to an IndexedValue emits an actual `Store`
/// operation, while converting an IndexedValue to a Value emits an actual
/// `Load` operation.
template <typename Load, typename Store>
class TemplatedIndexedValue {
 public:
  explicit TemplatedIndexedValue(Value v) : value(v) {}

  TemplatedIndexedValue(const TemplatedIndexedValue &rhs) = default;

  TemplatedIndexedValue operator()() { return *this; }
  /// Returns a new `TemplatedIndexedValue`.
  TemplatedIndexedValue operator()(Value index) {
    TemplatedIndexedValue res(value);
    res.indices.push_back(index);
    return res;
  }
  template <typename... Args>
  TemplatedIndexedValue operator()(Value index, Args... indices) {
    return TemplatedIndexedValue(value, index).append(indices...);
  }
  TemplatedIndexedValue operator()(ValueRange indices) {
    return TemplatedIndexedValue(value, indices);
  }

  /// Emits a `store`.
  Store operator=(const TemplatedIndexedValue &rhs) {
    return Store(rhs, value, indices);
  }
  Store operator=(Value rhs) { return Store(rhs, value, indices); }

  /// Emits a `load` when converting to a Value.
  operator Value() const { return Load(value, indices); }

  /// Returns the base memref.
  Value getBase() const { return value; }

  /// Returns the underlying memref.
  MemRefType getMemRefType() const {
    return value.getType().template cast<MemRefType>();
  }

  /// Returns the underlying MemRef elemental type cast as `T`.
  template <typename T>
  T getElementalTypeAs() const {
    return value.getType()
        .template cast<MemRefType>()
        .getElementType()
        .template cast<T>();
  }

  /// Arithmetic operator overloadings.
  Value operator+(Value e);
  Value operator-(Value e);
  Value operator*(Value e);
  Value operator/(Value e);
  Value operator%(Value e);
  Value operator^(Value e);
  Value operator+(TemplatedIndexedValue e) {
    return *this + static_cast<Value>(e);
  }
  Value operator-(TemplatedIndexedValue e) {
    return *this - static_cast<Value>(e);
  }
  Value operator*(TemplatedIndexedValue e) {
    return *this * static_cast<Value>(e);
  }
  Value operator/(TemplatedIndexedValue e) {
    return *this / static_cast<Value>(e);
  }
  Value operator%(TemplatedIndexedValue e) {
    return *this % static_cast<Value>(e);
  }
  Value operator^(TemplatedIndexedValue e) {
    return *this ^ static_cast<Value>(e);
  }

  /// Assignment-arithmetic operator overloadings.
  Store operator+=(Value e);
  Store operator-=(Value e);
  Store operator*=(Value e);
  Store operator/=(Value e);
  Store operator%=(Value e);
  Store operator^=(Value e);
  Store operator+=(TemplatedIndexedValue e) {
    return this->operator+=(static_cast<Value>(e));
  }
  Store operator-=(TemplatedIndexedValue e) {
    return this->operator-=(static_cast<Value>(e));
  }
  Store operator*=(TemplatedIndexedValue e) {
    return this->operator*=(static_cast<Value>(e));
  }
  Store operator/=(TemplatedIndexedValue e) {
    return this->operator/=(static_cast<Value>(e));
  }
  Store operator%=(TemplatedIndexedValue e) {
    return this->operator%=(static_cast<Value>(e));
  }
  Store operator^=(TemplatedIndexedValue e) {
    return this->operator^=(static_cast<Value>(e));
  }

  /// Logical operator overloadings.
  Value operator&&(Value e);
  Value operator||(Value e);
  Value operator&&(TemplatedIndexedValue e) {
    return *this && static_cast<Value>(e);
  }
  Value operator||(TemplatedIndexedValue e) {
    return *this || static_cast<Value>(e);
  }

  /// Comparison operator overloadings.
  Value eq(Value e);
  Value ne(Value e);
  Value slt(Value e);
  Value sle(Value e);
  Value sgt(Value e);
  Value sge(Value e);
  Value ult(Value e);
  Value ule(Value e);
  Value ugt(Value e);
  Value uge(Value e);
  Value slt(TemplatedIndexedValue e) {
    return slt(*this, static_cast<Value>(e));
  }
  Value sle(TemplatedIndexedValue e) {
    return sle(*this, static_cast<Value>(e));
  }
  Value sgt(TemplatedIndexedValue e) {
    return sgt(*this, static_cast<Value>(e));
  }
  Value sge(TemplatedIndexedValue e) {
    return sge(*this, static_cast<Value>(e));
  }
  Value ult(TemplatedIndexedValue e) {
    return ult(*this, static_cast<Value>(e));
  }
  Value ule(TemplatedIndexedValue e) {
    return ule(*this, static_cast<Value>(e));
  }
  Value ugt(TemplatedIndexedValue e) {
    return ugt(*this, static_cast<Value>(e));
  }
  Value uge(TemplatedIndexedValue e) {
    return uge(*this, static_cast<Value>(e));
  }

 private:
  TemplatedIndexedValue(Value value, ValueRange indices)
      : value(value), indices(indices.begin(), indices.end()) {}

  TemplatedIndexedValue &append() { return *this; }

  template <typename T, typename... Args>
  TemplatedIndexedValue &append(T index, Args... indices) {
    this->indices.push_back(static_cast<Value>(index));
    append(indices...);
    return *this;
  }
  Value value;
  SmallVector<Value, 8> indices;
};

/// Arithmetic operator overloadings.
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator+(Value e) {
  using op::operator+;
  return static_cast<Value>(*this) + e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator-(Value e) {
  using op::operator-;
  return static_cast<Value>(*this) - e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator*(Value e) {
  using op::operator*;
  return static_cast<Value>(*this) * e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator/(Value e) {
  using op::operator/;
  return static_cast<Value>(*this) / e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator%(Value e) {
  using op::operator%;
  return static_cast<Value>(*this) % e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator^(Value e) {
  using op::operator^;
  return static_cast<Value>(*this) ^ e;
}

/// Assignment-arithmetic operator overloadings.
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator+=(Value e) {
  using op::operator+;
  return Store(*this + e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator-=(Value e) {
  using op::operator-;
  return Store(*this - e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator*=(Value e) {
  using op::operator*;
  return Store(*this * e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator/=(Value e) {
  using op::operator/;
  return Store(*this / e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator%=(Value e) {
  using op::operator%;
  return Store(*this % e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator^=(Value e) {
  using op::operator^;
  return Store(*this ^ e, getBase(), indices);
}

/// Logical operator overloadings.
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator&&(Value e) {
  using op::operator&&;
  return static_cast<Value>(*this) && e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator||(Value e) {
  using op::operator||;
  return static_cast<Value>(*this) || e;
}

/// Comparison operator overloadings.
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::eq(Value e) {
  return eq(value, e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::ne(Value e) {
  return ne(value, e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::slt(Value e) {
  using op::slt;
  return slt(static_cast<Value>(*this), e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::sle(Value e) {
  using op::sle;
  return sle(static_cast<Value>(*this), e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::sgt(Value e) {
  using op::sgt;
  return sgt(static_cast<Value>(*this), e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::sge(Value e) {
  using op::sge;
  return sge(static_cast<Value>(*this), e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::ult(Value e) {
  using op::ult;
  return ult(static_cast<Value>(*this), e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::ule(Value e) {
  using op::ule;
  return ule(static_cast<Value>(*this), e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::ugt(Value e) {
  using op::ugt;
  return ugt(static_cast<Value>(*this), e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::uge(Value e) {
  using op::uge;
  return uge(static_cast<Value>(*this), e);
}

/// Creates a perfect nest of affine "for" loops, given the list of lower
/// bounds, upper bounds and steps. The three lists are expected to contain the
/// same number of elements. Uses the OpBuilder and Location stored in
/// ScopedContext and assumes they are non-null. The optional "bodyBuilderFn"
/// callback is called to construct the body of the innermost loop and is passed
/// the list of loop induction variables, in order from outermost to innermost.
/// The function is expected to use the builder and location stored in
/// ScopedContext at the moment of the call. The function should not create
/// the affine terminator op, which will be added regardless of the
/// "bodyBuilderFn" being present.
void affineLoopNestBuilder(
    ValueRange lbs, ValueRange ubs, ArrayRef<int64_t> steps,
    function_ref<void(ValueRange)> bodyBuilderFn = nullptr);

/// Creates a single affine "for" loop, iterating from max(lbs) to min(ubs) with
/// the given step. Uses the OpBuilder and Location stored in ScopedContext and
/// assumes they are non-null. The optional "bodyBuilderFn" callback is called
/// to construct the body of the loop and is passed the induction variable. The
/// function is expected to use the builder and location stored in ScopedContext
/// at the moment of the call. The function should not create the affine
/// terminator op, which will be added regardless of the "bodyBuilderFn" being
/// present.
void affineLoopBuilder(ValueRange lbs, ValueRange ubs, int64_t step,
                       function_ref<void(Value)> bodyBuilderFn = nullptr);

/// Creates a single affine "for" loop, iterating from max(lbs) to min(ubs) with
/// the given step. Uses the OpBuilder and Location stored in ScopedContext and
/// assumes they are non-null. "iterArgs" is used to specify the initial values
/// of the result affine "for" might yield. The optional "bodyBuilderFn"
/// callback is called to construct the body of the loop and is passed the
/// induction variable and the iteration arguments. The function is expected to
/// use the builder and location stored in ScopedContext at the moment of the
/// call. The function will create the affine terminator op in case "iterArgs"
/// is empty and "bodyBuilderFn" is not present.
void affineLoopBuilder(
    ValueRange lbs, ValueRange ubs, int64_t step, ValueRange iterArgs,
    function_ref<void(Value, ValueRange)> bodyBuilderFn = nullptr);

inline mlir::scf::LoopNest loopNestBuilder(ValueRange lbs, ValueRange ubs,
                                           ValueRange steps,
                                           function_ref<void(ValueRange)> fun) {
  // Delegates actual construction to scf::buildLoopNest by wrapping `fun` into
  // the expected function interface.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return mlir::scf::buildLoopNest(
      ScopedContext::getBuilderRef(), ScopedContext::getLocation(), lbs, ubs,
      steps, [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        ScopedContext context(builder, loc);
        if (fun) fun(ivs);
      });
}

inline mlir::scf::LoopNest loopNestBuilder(Value lb, Value ub, Value step,
                                           function_ref<void(Value)> fun) {
  // Delegates to the ValueRange-based version by wrapping the lambda.
  auto wrapper = [&](ValueRange ivs) {
    assert(ivs.size() == 1);
    if (fun) fun(ivs[0]);
  };
  return loopNestBuilder(ValueRange(lb), ValueRange(ub), ValueRange(step),
                         wrapper);
}

inline mlir::scf::LoopNest loopNestBuilder(
    Value lb, Value ub, Value step, ValueRange iterArgInitValues,
    function_ref<scf::ValueVector(Value, ValueRange)> fun) {
  // Delegates actual construction to scf::buildLoopNest by wrapping `fun` into
  // the expected function interface.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return mlir::scf::buildLoopNest(
      ScopedContext::getBuilderRef(), ScopedContext::getLocation(), lb, ub,
      step, iterArgInitValues,
      [&](OpBuilder &builder, Location loc, ValueRange ivs, ValueRange args) {
        assert(ivs.size() == 1 && "expected one induction variable");
        ScopedContext context(builder, loc);
        if (fun) return fun(ivs[0], args);
        return scf::ValueVector(iterArgInitValues.begin(),
                                iterArgInitValues.end());
      });
}

inline mlir::scf::LoopNest loopNestBuilder(
    ValueRange lbs, ValueRange ubs, ValueRange steps,
    ValueRange iterArgInitValues,
    function_ref<scf::ValueVector(ValueRange, ValueRange)> fun) {
  // Delegates actual construction to scf::buildLoopNest by wrapping `fun` into
  // the expected function interface.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return mlir::scf::buildLoopNest(
      ScopedContext::getBuilderRef(), ScopedContext::getLocation(), lbs, ubs,
      steps, iterArgInitValues,
      [&](OpBuilder &builder, Location loc, ValueRange ivs, ValueRange args) {
        ScopedContext context(builder, loc);
        if (fun) return fun(ivs, args);
        return scf::ValueVector(iterArgInitValues.begin(),
                                iterArgInitValues.end());
      });
}

inline std::function<void(OpBuilder &, Location)> wrapIfBody(
    function_ref<scf::ValueVector()> body, TypeRange expectedTypes) {
  (void)expectedTypes;
  return [=](OpBuilder &builder, Location loc) {
    ScopedContext context(builder, loc);
    scf::ValueVector returned = body();
    assert(ValueRange(returned).getTypes() == expectedTypes &&
           "'if' body builder returned values of unexpected type");
    builder.create<scf::YieldOp>(loc, returned);
  };
}

inline ValueRange conditionBuilder(TypeRange results, Value condition,
                                   function_ref<scf::ValueVector()> thenBody,
                                   function_ref<scf::ValueVector()> elseBody,
                                   scf::IfOp *ifOp = nullptr) {
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(thenBody && "thenBody is mandatory");

  auto newOp = ScopedContext::getBuilderRef().create<scf::IfOp>(
      ScopedContext::getLocation(), results, condition,
      wrapIfBody(thenBody, results), wrapIfBody(elseBody, results));
  if (ifOp) *ifOp = newOp;
  return newOp.getResults();
}

inline std::function<void(OpBuilder &, Location)> wrapZeroResultIfBody(
    function_ref<void()> body) {
  return [=](OpBuilder &builder, Location loc) {
    ScopedContext context(builder, loc);
    body();
    builder.create<scf::YieldOp>(loc);
  };
}

inline ValueRange conditionBuilder(Value condition,
                                   function_ref<void()> thenBody,
                                   function_ref<void()> elseBody,
                                   scf::IfOp *ifOp = nullptr) {
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(thenBody && "thenBody is mandatory");

  auto newOp = ScopedContext::getBuilderRef().create<scf::IfOp>(
      ScopedContext::getLocation(), condition, wrapZeroResultIfBody(thenBody),
      elseBody ? llvm::function_ref<void(OpBuilder &, Location)>(
                     wrapZeroResultIfBody(elseBody))
               : llvm::function_ref<void(OpBuilder &, Location)>(nullptr));
  if (ifOp) *ifOp = newOp;
  return {};
}

}  // namespace edsc

// List of MLIR EDSC instrinsics exposed to external clients of ModelBuilder.
// All other intrinsics are abstracted away via ModelBuilder methods.
// -----------------------------------------------------------------------------
// From the Linalg Dialect.
using linalg_copy = edsc::OperationBuilder<linalg::CopyOp>;
using linalg_dot = edsc::OperationBuilder<linalg::DotOp>;
using linalg_fill = edsc::OperationBuilder<linalg::FillOp>;
using linalg_init_tensor = edsc::ValueBuilder<linalg::InitTensorOp>;
using linalg_matmul = edsc::OperationBuilder<linalg::MatmulOp>;
using linalg_matvec = edsc::OperationBuilder<linalg::MatvecOp>;
using linalg_vecmat = edsc::OperationBuilder<linalg::VecmatOp>;
using linalg_range = edsc::ValueBuilder<linalg::RangeOp>;
using linalg_expand_shape = edsc::ValueBuilder<memref::ExpandShapeOp>;
using linalg_collapse_shape = edsc::ValueBuilder<memref::CollapseShapeOp>;
using linalg_yield = edsc::OperationBuilder<linalg::YieldOp>;
// From the Vector Dialect.
using vector_broadcast = edsc::ValueBuilder<vector::BroadcastOp>;
using vector_contract = edsc::ValueBuilder<vector::ContractionOp>;
using vector_extract = edsc::ValueBuilder<vector::ExtractOp>;
using vector_extract_element = edsc::ValueBuilder<vector::ExtractElementOp>;
using vector_extract_strided_slice =
    edsc::ValueBuilder<vector::ExtractStridedSliceOp>;
using vector_fma = edsc::ValueBuilder<vector::FMAOp>;
using vector_insert = edsc::ValueBuilder<vector::InsertOp>;
using vector_insert_element = edsc::ValueBuilder<vector::InsertElementOp>;
using vector_insert_strided_slice =
    edsc::ValueBuilder<vector::InsertStridedSliceOp>;
using vector_matmul = edsc::ValueBuilder<vector::MatmulOp>;
using vector_outerproduct = edsc::ValueBuilder<vector::OuterProductOp>;
using vector_print = edsc::OperationBuilder<vector::PrintOp>;
using vector_transfer_read = edsc::ValueBuilder<vector::TransferReadOp>;
using vector_transfer_write = edsc::OperationBuilder<vector::TransferWriteOp>;
using vector_transpose = edsc::ValueBuilder<vector::TransposeOp>;
using vector_type_cast = edsc::ValueBuilder<vector::TypeCastOp>;
// From the Memref Dialect.
using memref_alloc = edsc::ValueBuilder<memref::AllocOp>;
using memref_alloca = edsc::ValueBuilder<memref::AllocaOp>;
using memref_cast = edsc::ValueBuilder<memref::CastOp>;
using memref_dealloc = edsc::OperationBuilder<memref::DeallocOp>;
using memref_dim = edsc::ValueBuilder<memref::DimOp>;
using memref_load = edsc::ValueBuilder<memref::LoadOp>;
using memref_store = edsc::OperationBuilder<memref::StoreOp>;
using memref_sub_view = edsc::ValueBuilder<memref::SubViewOp>;
using memref_tensor_load = edsc::ValueBuilder<memref::TensorLoadOp>;
using memref_tensor_store = edsc::OperationBuilder<memref::TensorStoreOp>;
using memref_view = edsc::ValueBuilder<memref::ViewOp>;
// From the Std Dialect.
using std_addi = edsc::ValueBuilder<arith::AddIOp>;
using std_addf = edsc::ValueBuilder<arith::AddFOp>;
using std_call = edsc::OperationBuilder<CallOp>;
using std_constant = edsc::ValueBuilder<arith::ConstantOp>;
using std_constant_float = edsc::ValueBuilder<arith::ConstantFloatOp>;
using std_constant_index = edsc::ValueBuilder<arith::ConstantIndexOp>;
using std_constant_int = edsc::ValueBuilder<arith::ConstantIntOp>;
using std_divis = edsc::ValueBuilder<arith::DivSIOp>;
using std_diviu = edsc::ValueBuilder<arith::DivUIOp>;
using std_fpext = edsc::ValueBuilder<arith::ExtFOp>;
using std_fptrunc = edsc::ValueBuilder<arith::TruncFOp>;
using std_index_cast = edsc::ValueBuilder<arith::IndexCastOp>;
using std_muli = edsc::ValueBuilder<arith::MulIOp>;
using std_mulf = edsc::ValueBuilder<arith::MulIOp>;
using std_ret = edsc::OperationBuilder<ReturnOp>;
using std_select = edsc::ValueBuilder<SelectOp>;
using std_sign_extendi = edsc::ValueBuilder<arith::ExtSIOp>;
using std_splat = edsc::ValueBuilder<SplatOp>;
using std_subf = edsc::ValueBuilder<arith::SubFOp>;
using std_subi = edsc::ValueBuilder<arith::SubIOp>;
using std_zero_extendi = edsc::ValueBuilder<arith::ExtUIOp>;
using tensor_extract = edsc::ValueBuilder<tensor::ExtractOp>;

template <int N>
struct SExtiValueBuilder : public edsc::ValueBuilder<arith::ExtSIOp> {
  using edsc::ValueBuilder<arith::ExtSIOp>::ValueBuilder;
  template <typename... Args>
  SExtiValueBuilder(Args... args)
      : edsc::ValueBuilder<arith::ExtSIOp>(
            edsc::ScopedContext::getBuilderRef().getI32Type(), args...) {}
};

using std_sexti32 = SExtiValueBuilder<32>;

template <arith::CmpFPredicate Predicate>
struct CmpFValueBuilder : public edsc::ValueBuilder<arith::CmpFOp> {
  using edsc::ValueBuilder<arith::CmpFOp>::ValueBuilder;
  template <typename... Args>
  CmpFValueBuilder(Args... args)
      : edsc::ValueBuilder<arith::CmpFOp>(Predicate, args...) {}
};

using std_cmpf_ogt = CmpFValueBuilder<arith::CmpFPredicate::OGT>;
using std_cmpf_olt = CmpFValueBuilder<arith::CmpFPredicate::OLT>;

template <arith::CmpIPredicate Predicate>
struct CmpIValueBuilder : public edsc::ValueBuilder<arith::CmpIOp> {
  using ValueBuilder<arith::CmpIOp>::ValueBuilder;
  template <typename... Args>
  CmpIValueBuilder(Args... args)
      : edsc::ValueBuilder<arith::CmpIOp>(Predicate, args...) {}
};

using std_cmpi_sgt = CmpIValueBuilder<arith::CmpIPredicate::sgt>;

/// Branches into `block` with `operands`.
BranchOp std_br(Block *block, ValueRange operands);

/// Branches into `trueBranch` with `trueOperands` if `cond` evaluates to `true`
/// or to `falseBranch` and `falseOperand` if `cond` evaluates to `false`.
CondBranchOp std_cond_br(Value cond, Block *trueBranch, ValueRange trueOperands,
                         Block *falseBranch, ValueRange falseOperands);
// From the Math Dialect.
using math_tanh = edsc::ValueBuilder<math::TanhOp>;
// From the Affine Dialect.
using affine_apply = edsc::ValueBuilder<AffineApplyOp>;
using affine_if = edsc::OperationBuilder<AffineIfOp>;
using affine_load = edsc::ValueBuilder<AffineLoadOp>;
using affine_min = edsc::ValueBuilder<AffineMinOp>;
using affine_max = edsc::ValueBuilder<AffineMaxOp>;
using affine_store = edsc::OperationBuilder<AffineStoreOp>;
// From the SCF Dialect.
using edsc::loopNestBuilder;
// From the LLVM dialect.
using llvm_add = edsc::ValueBuilder<LLVM::AddOp>;
using llvm_bitcast = edsc::ValueBuilder<LLVM::BitcastOp>;
using llvm_constant = edsc::ValueBuilder<LLVM::ConstantOp>;
using llvm_extractvalue = edsc::ValueBuilder<LLVM::ExtractValueOp>;
using llvm_gep = edsc::ValueBuilder<LLVM::GEPOp>;
using llvm_insertvalue = edsc::ValueBuilder<LLVM::InsertValueOp>;
using llvm_call = edsc::OperationBuilder<LLVM::CallOp>;
using llvm_icmp = edsc::ValueBuilder<LLVM::ICmpOp>;
using llvm_load = edsc::ValueBuilder<LLVM::LoadOp>;
using llvm_store = edsc::OperationBuilder<LLVM::StoreOp>;
using llvm_select = edsc::ValueBuilder<LLVM::SelectOp>;
using llvm_mul = edsc::ValueBuilder<LLVM::MulOp>;
using llvm_ptrtoint = edsc::ValueBuilder<LLVM::PtrToIntOp>;
using llvm_sub = edsc::ValueBuilder<LLVM::SubOp>;
using llvm_undef = edsc::ValueBuilder<LLVM::UndefOp>;
using llvm_urem = edsc::ValueBuilder<LLVM::URemOp>;
using llvm_alloca = edsc::ValueBuilder<LLVM::AllocaOp>;
using llvm_return = edsc::OperationBuilder<LLVM::ReturnOp>;

// -----------------------------------------------------------------------------

// Helper class to simplify MLIR function construction by adding proper
// attributes, some of which pass through to LLVM.
struct MLIRFuncOpConfig {
  // Applies the MLIRFuncOpConfig to `f`.
  void apply(FuncOp &f);

  // Attributes that pass through to LLVM and modify the behavior of the LLVM
  // compiler.
  bool noInline = false;
  MLIRFuncOpConfig &setNoInline(bool v);

  bool preferAvx512 = false;
  MLIRFuncOpConfig &setPreferAvx512(bool v);

  std::string targetCpu = "";
  MLIRFuncOpConfig &setTargetCpu(StringRef s);

  // When true, the function remains body-less. This is good for declaring
  // external functions.
  bool declOnly = false;
  MLIRFuncOpConfig &setDeclOnly(bool v);

  // When true, an mlir_c_iface_xxx shim function is emitted with C compatible
  // strided memref ABI.
  bool emitCInterface = false;
  MLIRFuncOpConfig &setEmitCInterface(bool v);
};

// Entry point class to build a whole model declaratively with C++ EDSCs.
class ModelBuilder : public OpBuilder {
 public:
  using OpBuilder::create;

  // Create a ModelBuilder and sets up an owned MLIRContext, ModuleOp and
  // SymbolTable as well as uniqued MLIR types.
  ModelBuilder();

  // Register all the dialects used by ModelBuilder.
  static void registerAllDialects();

  // Return a reference to the underlying module.
  OwningOpRef<ModuleOp> &getModuleRef() { return module; }

  // Build an MLIR FuncOp that will be callable after JIT compilation occured.
  // `config` is a convenience class provided to simplify the configuration of
  // the function with common attributes that are non-obvious to the newcomer.
  FuncOp makeFunction(StringRef name, ArrayRef<Type> results,
                      ArrayRef<Type> args,
                      MLIRFuncOpConfig config = MLIRFuncOpConfig());
  FuncOp makeFunction(std::function<std::string(FunctionType)> nameBuilder,
                      ArrayRef<Type> results, ArrayRef<Type> args,
                      MLIRFuncOpConfig config = MLIRFuncOpConfig());

  // Add GPU attribute to the module.
  void addGPUAttr();

  // Build a MLIR GPU module. GPUFuncOp can later be added to the module.
  gpu::GPUModuleOp makeGPUModule(StringRef name);

  // Build a MLIR GPU kernel within a GPU module.
  gpu::GPUFuncOp makeGPUKernel(StringRef name, gpu::GPUModuleOp GPUModule,
                               ArrayRef<int32_t> workgroupSize,
                               ArrayRef<Type> args = {},
                               ArrayRef<Type> results = {});

  // Build an MLIR VectorType with a base `elementalType` and a `shape`.
  VectorType getVectorType(ArrayRef<int64_t> shape, Type elementalType);

  // Build an MLIR MemRefType with a base `elementType` and a `shape` that can
  // be any mix of static and dynamic values. For now this only supports a dense
  // and contiguous layout.
  // In the future, this can be extended support more advanced layouts, on a
  // per-need basis.
  MemRefType getMemRefType(ArrayRef<int64_t> shape, Type elementType,
                           unsigned addressSpace = 0);

  // Build an MLIR RankedTensorType with a base `elementType` and a `shape` that
  // can be any mix of static and dynamic values. For now this only supports a
  // dense and contiguous layout.
  // In the future, this can be extended support more advanced layouts, on a
  // per-need basis.
  RankedTensorType getRankedTensorType(ArrayRef<int64_t> shape,
                                       Type elementType);

  // Build the MLIR representation for constants of common types.
  static Value constant_f32(float v);
  static Value constant_f64(double v);
  static Value constant_index(int64_t v);

  // Build the MLIR representation for:
  //   1. fc(I, W, O)
  //   2. pointwise(O, bias) in-place with explicit bias broadcast to compute:
  //      `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  // Returns O.
  // Version with a MemRef output argument.
  static Value FCBiasTanh(std::array<Value, 3> fcArgs, Value biasValueArg);
  // Version with a RankedTensor result.
  static Value FCBiasTanhTensors(RankedTensorType outputTensorType,
                                 std::array<Value, 2> fcArgs,
                                 Value fcInitTensor, Value biasValueArg);

  // Build the MLIR representation for:
  //   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  // This assumes `x` and `bias` capture scalar MLIR values of type f32.
  // This is used as a region builder when constructing e.g. a pointwise op.
  static Value fusedBiasTanh(Value x, Value bias);

  // ---------------------------------------------------------------------------
  // Support for emitting special function calls.
  // ---------------------------------------------------------------------------
  static Value call_tanhf(Value v);
  static void call_print_memref_f32(Value v);  // needs libmlir_runner_utils.so

 protected:
  // Helper function to support calling into known functions (e.g. libmath).
  static Operation *emitCallToRegisteredSymbol(StringRef functionName,
                                               ArrayRef<Type> returnTypes,
                                               ValueRange values);

  // ---------------------------------------------------------------------------
  // Members.
  // ---------------------------------------------------------------------------
 protected:
  // Thread-safe context owned by ModelBuilder. All IR is built in this context.
  static thread_local MLIRContext ctx;
  mlir::OwningModuleRef module;
  // The symbol table for the module.
  mlir::SymbolTable symbolTable;

 public:
  // The mlir::Location of the single owned Module.
  Location loc;
  // The unique mlir::IntegerType of 8 bits.
  IntegerType i8;
  // The unique mlir::FloatType of 32 bits.
  FloatType f32;
  // The unique mlir::FloatType of 64 bits.
  FloatType f64;
};

// -----------------------------------------------------------------------------
// EDSC extensions.
// -----------------------------------------------------------------------------
namespace edsc {
namespace extensions {

template <typename T>
SmallVector<Value, 4> std_constant_indices(ArrayRef<T> a) {
  auto makeIndex = [](int64_t v) {
    return mlir::std_constant_index(v).value;
  };
  return llvm::to_vector<4>(llvm::map_range(a, makeIndex));
}
// Build the MLIR representation for op(a, b) for each pair of elements in
// zip(`a`, `b`).
SmallVector<Value, 4> operator+(ValueRange a, ValueRange b);
SmallVector<Value, 4> operator-(ValueRange a, ValueRange b);
// Build the MLIR representation for select(a cmp b, a, b) for each pair of
// elements in zip(`a`, `b`).
SmallVector<Value, 4> std_max(ValueRange a, ValueRange b);
SmallVector<Value, 4> std_min(ValueRange a, ValueRange b);
// Build the MLIR representation for affine_cmp(a, b) for each pair of elements
// in zip(`a`, `b`).
SmallVector<Value, 4> affine_max(ValueRange a, ValueRange b);
SmallVector<Value, 4> affine_min(ValueRange a, ValueRange b);

}  // namespace extensions

/// Provide an index notation around affine_load and affine_store.
using AffineIndexedValue =
    TemplatedIndexedValue<affine_load, affine_store>;
using MemRefIndexedValue = TemplatedIndexedValue<memref_load, memref_store>;

}  // namespace edsc

using edsc::AffineIndexedValue;
using edsc::MemRefIndexedValue;

inline Value vector_contraction(edsc::StructuredIndexed A,
                                edsc::StructuredIndexed B,
                                edsc::StructuredIndexed C,
                                ArrayRef<IteratorType> iteratorTypes) {
  using IndexingExprs = ArrayRef<ArrayRef<AffineExpr>>;
  return vector_contract(
      A.getValue(), B.getValue(), C.getValue(),
      IndexingExprs{A.getExprs(), B.getExprs(), C.getExprs()},
      ArrayRef<StringRef>{
          llvm::to_vector<8>(llvm::map_range(iteratorTypes, toString))});
}

inline Value vector_contraction_matmul(Value A, Value B, Value C) {
  AffineExpr m, n, k;
  bindDims(edsc::ScopedContext::getContext(), m, n, k);
  return vector_contraction(edsc::StructuredIndexed(A, {m, k}),
                            edsc::StructuredIndexed(B, {k, n}),
                            edsc::StructuredIndexed(C, {m, n}),
                            {IteratorType::Parallel, IteratorType::Parallel,
                             IteratorType::Reduction});
}

inline Operation *makeGenericLinalgOp(
    ArrayRef<IteratorType> iteratorTypes,
    ArrayRef<edsc::StructuredIndexed> inputs,
    ArrayRef<edsc::StructuredIndexed> outputs, TypeRange resultTensorTypes,
    function_ref<void(ValueRange)> regionBuilder,
    ArrayRef<Value> otherValues = {},
    ArrayRef<Attribute> otherAttributes = {}) {
  // Build maps
  SmallVector<SmallVector<AffineExpr, 4>, 4> exprsList;
  exprsList.reserve(inputs.size() + outputs.size());

  for (auto container : {inputs, outputs})
    for (const edsc::StructuredIndexed &s : container)
      exprsList.emplace_back(s.getExprs().begin(), s.getExprs().end());
  auto maps = AffineMap::inferFromExprList(exprsList);

  SmallVector<Value, 4> inputValues, outputValues;
  inputValues.reserve(inputs.size());
  outputValues.reserve(outputs.size());
  std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputValues));
  std::copy(outputs.begin(), outputs.end(), std::back_inserter(outputValues));

  auto iteratorStrTypes =
      llvm::to_vector<8>(llvm::map_range(iteratorTypes, toString));
  // clang-format off
  auto *op =
      edsc::ScopedContext::getBuilderRef()
          .create<linalg::GenericOp>(
              edsc::ScopedContext::getLocation(),
              resultTensorTypes,
              inputValues,
              outputValues,
              maps,
              iteratorStrTypes,
              ""/*doc*/,
              ""/*library_call*/)
          .getOperation();
  // clang-format on

  using namespace edsc;
  SmallVector<Type, 4> blockTypes;
  blockTypes.reserve(inputs.size() + outputs.size());
  for (auto container : {inputs, outputs})
    for (const edsc::StructuredIndexed &s : container)
      blockTypes.push_back(getElementTypeOrSelf(s.getType()));

  assert(op->getNumRegions() == 1);
  assert(op->getRegion(0).empty());
  OpBuilder opBuilder(op);
  edsc::ScopedContext scope(opBuilder, op->getLoc());
  buildInNewBlock(op->getRegion(0), blockTypes, regionBuilder);
  assert(llvm::hasSingleElement(op->getRegion(0)));
  return op;
}

inline void mulRegionBuilder(ValueRange args) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  assert(args.size() == 2 && "expected 2 block arguments");
  Value a(args[0]), b(args[1]);
  linalg_yield(a * b);
}

inline void macRegionBuilder(ValueRange args) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  assert(args.size() == 3 && "expected 3 block arguments");
  Value a(args[0]), b(args[1]), c(args[2]);
  linalg_yield(c + a * b);
}

using UnaryPointwiseOpBuilder = function_ref<Value(Value)>;
using BinaryPointwiseOpBuilder = function_ref<Value(Value, Value)>;
inline Operation *linalg_generic_pointwise(UnaryPointwiseOpBuilder unaryOp,
                                           edsc::StructuredIndexed I,
                                           edsc::StructuredIndexed O) {
  SmallVector<IteratorType, 4> iterTypes(O.getExprs().size(),
                                         IteratorType::Parallel);
  auto fun = [&unaryOp](ValueRange args) {
    assert(!args.empty() && "expected >= 1 block arguments");
    Value a(args[0]);
    linalg_yield(unaryOp(a));
  };
  if (O.getType().isa<RankedTensorType>())
    return makeGenericLinalgOp(iterTypes, /*inputs=*/{I}, /*outputs=*/{O},
                               /*resultTensorTypes=*/{O}, fun);
  return makeGenericLinalgOp(iterTypes, /*inputs=*/{I}, /*outputs=*/{O},
                             /*resultTensorTypes=*/{}, fun);
}

inline Operation *linalg_generic_pointwise_tanh(edsc::StructuredIndexed I,
                                                edsc::StructuredIndexed O) {
  UnaryPointwiseOpBuilder unOp([](Value a) -> Value { return math_tanh(a); });
  return linalg_generic_pointwise(unOp, I, O);
}

/// Binary pointwise operation (with broadcast) entry point.
inline Operation *linalg_generic_pointwise(BinaryPointwiseOpBuilder binaryOp,
                                           edsc::StructuredIndexed I1,
                                           edsc::StructuredIndexed I2,
                                           edsc::StructuredIndexed O) {
  SmallVector<IteratorType, 4> iterTypes(O.getExprs().size(),
                                         IteratorType::Parallel);
  auto fun = [&binaryOp](ValueRange args) {
    assert(args.size() >= 2 && "expected >= 2 block arguments");
    Value a(args[0]), b(args[1]);
    linalg_yield(binaryOp(a, b));
  };
  if (O.getType().isa<RankedTensorType>())
    return makeGenericLinalgOp(iterTypes, /*inputs=*/{I1, I2}, /*outputs=*/{O},
                               /*resultTensorTypes=*/{O}, fun);
  return makeGenericLinalgOp(iterTypes, /*inputs=*/{I1, I2},
                             /*outputs=*/{O}, /*resultTensorTypes=*/{}, fun);
}

inline Operation *linalg_generic_pointwise_add(edsc::StructuredIndexed I1,
                                               edsc::StructuredIndexed I2,
                                               edsc::StructuredIndexed O) {
  using edsc::op::operator+;
  BinaryPointwiseOpBuilder binOp(
      [](Value a, Value b) -> Value { return a + b; });
  return linalg_generic_pointwise(binOp, I1, I2, O);
}

inline Operation *linalg_generic_pointwise_max(edsc::StructuredIndexed I1,
                                               edsc::StructuredIndexed I2,
                                               edsc::StructuredIndexed O) {
  BinaryPointwiseOpBuilder binOp([](Value a, Value b) -> Value {
    using edsc::op::sgt;
    return std_select(sgt(a, b), a, b);
  });
  return linalg_generic_pointwise(binOp, I1, I2, O);
}

using MatmulRegionBuilder = function_ref<void(ValueRange args)>;
inline Operation *linalg_generic_matmul(
    Value vA, Value vB, Value vC,
    MatmulRegionBuilder regionBuilder = macRegionBuilder) {
  // clang-format off
  AffineExpr m, n, k;
  bindDims(edsc::ScopedContext::getContext(), m, n, k);
  edsc::StructuredIndexed A(vA), B(vB), C(vC);
  return makeGenericLinalgOp(
    {IteratorType::Parallel, IteratorType::Parallel, IteratorType::Reduction},
    /*inputs=*/{A({m, k}), B({k, n})},
    /*outputs=*/{C({m, n})},
    /*resultTensorTypes=*/{},
    regionBuilder);
  // clang-format on
}

inline Operation *linalg_generic_matmul(
    Value vA, Value vB, Value vC, RankedTensorType tD,
    MatmulRegionBuilder regionBuilder = macRegionBuilder) {
  // clang-format off
  AffineExpr m, n, k;
  bindDims(edsc::ScopedContext::getContext(), m, n, k);
  edsc::StructuredIndexed A(vA), B(vB), C(vC), D(tD);
  return makeGenericLinalgOp(
    {IteratorType::Parallel, IteratorType::Parallel, IteratorType::Reduction},
    /*inputs=*/{A({m, k}), B({k, n})},
    /*outputs=*/{C({m, n})},
    /*resultTensorTypes=*/{D({m, n})},
    regionBuilder);
  // clang-format on
}

inline Operation *linalg_generic_conv_nhwc(Value vI, Value vW, Value vO,
                                           ArrayRef<int> strides,
                                           ArrayRef<int> dilations) {
  MLIRContext *ctx = edsc::ScopedContext::getContext();
  // TODO: some template magic to make everything rank-polymorphic.
  assert((dilations.empty() || dilations.size() == 2) && "only 2-D conv atm");
  assert((strides.empty() || strides.size() == 2) && "only 2-D conv atm");

  // Some short names.
  auto par = IteratorType::Parallel;
  auto red = IteratorType::Reduction;
  auto s = strides;
  auto d = dilations;

  AffineExpr b, f, h, w, kh, kw, c;
  bindDims(ctx, b, f, h, w, kh, kw, c);
  unsigned numDims = c.cast<AffineDimExpr>().getPosition() + 1;
  edsc::StructuredIndexed I(vI), W(vW), O(vO);
  // clang-format off
  return makeGenericLinalgOp(
    {par, par, par, par, red, red, red},
    /*inputs=*/{
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, f}) },
    /*outputs=*/{ O({b, h, w, f}) },
    /*resultTensorTypes=*/{},
    macRegionBuilder);
  // clang-format on
}

inline Operation *linalg_generic_dilated_conv_nhwc(Value vI, Value vW, Value vO,
                                                   int depth_multiplier,
                                                   ArrayRef<int> strides,
                                                   ArrayRef<int> dilations) {
  MLIRContext *ctx = edsc::ScopedContext::getContext();
  // TODO: some template magic to make everything rank-polymorphic.
  assert((dilations.empty() || dilations.size() == 2) && "only 2-D conv atm");
  assert((strides.empty() || strides.size() == 2) && "only 2-D conv atm");

  // Some short names.
  auto par = IteratorType::Parallel;
  auto red = IteratorType::Reduction;
  auto s = strides;
  auto d = dilations;

  // clang-format off
  AffineExpr b, dm, c, h, w, kh, kw;
  bindDims(ctx, b, dm, c, h, w, kh, kw);
  unsigned numDims = kw.cast<AffineDimExpr>().getPosition() + 1;
  edsc::StructuredIndexed I(vI), W(vW), O(vO);
  return makeGenericLinalgOp(
    {par, par, par, par, par, red, red},
    /*inputs=*/{
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, dm})},
    /*outputs=*/{
      O({b, h, w, simplifyAffineExpr(c * depth_multiplier + dm, numDims, 0)})},
    /*resultTensorTypes=*/{},
    macRegionBuilder);
  // clang-format on
}
}  // namespace mlir

#endif  // IREE_LLVM_SANDBOX_MODELBUILDER_MODELBUILDER_H_
