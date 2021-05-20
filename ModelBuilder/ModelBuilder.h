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

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
using edsc::OperationBuilder;
using edsc::ScopedContext;
using edsc::StructuredIndexed;
using edsc::ValueBuilder;

namespace edsc {

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
using edsc::intrinsics::linalg_fill;
using edsc::intrinsics::linalg_matmul;
using edsc::intrinsics::linalg_yield;
using edsc::ops::linalg_generic_matmul;
// From the Vector Dialect.
using vector_broadcast = ValueBuilder<vector::BroadcastOp>;
using vector_contract = ValueBuilder<vector::ContractionOp>;
using vector_extract = ValueBuilder<vector::ExtractOp>;
using vector_extract_element = ValueBuilder<vector::ExtractElementOp>;
using vector_extract_slices = ValueBuilder<vector::ExtractSlicesOp>;
using vector_extract_strided_slice =
    ValueBuilder<vector::ExtractStridedSliceOp>;
using vector_fma = ValueBuilder<vector::FMAOp>;
using vector_insert = ValueBuilder<vector::InsertOp>;
using vector_insert_element = ValueBuilder<vector::InsertElementOp>;
using vector_insert_slices = ValueBuilder<vector::InsertSlicesOp>;
using vector_insert_strided_slice = ValueBuilder<vector::InsertStridedSliceOp>;
using vector_matmul = ValueBuilder<vector::MatmulOp>;
using vector_outerproduct = ValueBuilder<vector::OuterProductOp>;
using vector_print = OperationBuilder<vector::PrintOp>;
using vector_transfer_read = ValueBuilder<vector::TransferReadOp>;
using vector_transfer_write = OperationBuilder<vector::TransferWriteOp>;
using vector_transpose = ValueBuilder<vector::TransposeOp>;
using vector_type_cast = ValueBuilder<vector::TypeCastOp>;
// From the Std Dialect.
using edsc::MemRefBoundsCapture;
using edsc::VectorBoundsCapture;
using edsc::intrinsics::memref_alloc;
using edsc::intrinsics::memref_dealloc;
using edsc::intrinsics::memref_dim;
using edsc::intrinsics::std_addf;
using edsc::intrinsics::std_call;
using edsc::intrinsics::std_constant_float;
using edsc::intrinsics::std_constant_index;
using edsc::intrinsics::std_mulf;
using edsc::intrinsics::std_ret;
// From the Affine Dialect.
using edsc::intrinsics::affine_max;
using edsc::intrinsics::affine_min;
// From the Loop Dialect.
using edsc::loopNestBuilder;
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
  auto makeIndex = [](int64_t v) { return mlir::std_constant_index(v).value; };
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
    TemplatedIndexedValue<intrinsics::affine_load, intrinsics::affine_store>;
using MemRefIndexedValue =
    TemplatedIndexedValue<intrinsics::memref_load, intrinsics::memref_store>;

}  // namespace edsc

using edsc::AffineIndexedValue;
using edsc::MemRefIndexedValue;

inline Value vector_contraction(StructuredIndexed A, StructuredIndexed B,
                                StructuredIndexed C,
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
  bindDims(ScopedContext::getContext(), m, n, k);
  return vector_contraction(StructuredIndexed(A, {m, k}),
                            StructuredIndexed(B, {k, n}),
                            StructuredIndexed(C, {m, n}),
                            {IteratorType::Parallel, IteratorType::Parallel,
                             IteratorType::Reduction});
}
}  // namespace mlir

#endif  // IREE_LLVM_SANDBOX_MODELBUILDER_MODELBUILDER_H_
