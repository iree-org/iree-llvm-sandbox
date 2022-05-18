//===-- IteratorsToLLVM.h - Conversion from Iterators to LLVM ---*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Conversion/IteratorsToLLVM/IteratorsToLLVM.h"

#include "../PassDetail.h"
#include "IteratorAnalysis.h"
#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Utils/MLIRSupport.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <assert.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::iterators;
using namespace mlir::LLVM;
using namespace mlir::func;
using namespace ::iterators;
using namespace std::string_literals;
using IteratorInfo = IteratorAnalysis::IteratorInfo;

namespace {
struct ConvertIteratorsToLLVMPass
    : public ConvertIteratorsToLLVMBase<ConvertIteratorsToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

/// Maps types from the Iterators dialect to corresponding types in LLVM.
class IteratorsTypeConverter : public TypeConverter {
public:
  IteratorsTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertTupleType);
  }

private:
  /// Maps a TupleType to a corresponding LLVMStructType
  static Optional<Type> convertTupleType(Type type) {
    if (TupleType tupleType = type.dyn_cast<TupleType>()) {
      return LLVMStructType::getNewIdentified(type.getContext(), "tuple",
                                              tupleType.getTypes());
    }
    return llvm::None;
  }
};

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr lookupOrInsertPrintf(OpBuilder &builder,
                                              ModuleOp module) {
  if (module.lookupSymbol<LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(builder.getContext(), "printf");

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  LLVMPointerType charPointerType = LLVMPointerType::get(builder.getI8Type());
  LLVMFunctionType printfFunctionType =
      LLVMFunctionType::get(builder.getI32Type(), charPointerType,
                            /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());
  builder.create<LLVMFuncOp>(module.getLoc(), "printf", printfFunctionType);
  return SymbolRefAttr::get(builder.getContext(), "printf");
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(OpBuilder &builder, Twine name,
                                     Twine value, ModuleOp module) {
  Location loc = module->getLoc();

  // Create the global at the entry of the module.
  GlobalOp global;
  StringAttr nameAttr = builder.getStringAttr(name);
  if (!(global = module.lookupSymbol<GlobalOp>(nameAttr))) {
    StringAttr valueAttr = builder.getStringAttr(value);
    LLVMArrayType globalType =
        LLVMArrayType::get(builder.getI8Type(), valueAttr.size());
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    global = builder.create<GlobalOp>(loc, globalType, /*isConstant=*/true,
                                      Linkage::Internal, nameAttr, valueAttr,
                                      /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<AddressOfOp>(loc, global);
  Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getI64IntegerAttr(0));
  return builder.create<GEPOp>(loc, LLVMPointerType::get(builder.getI8Type()),
                               globalPtr, ArrayRef<Value>({zero, zero}));
}

struct ConstantTupleLowering : public ConversionPattern {
  ConstantTupleLowering(TypeConverter &typeConverter, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, "iterators.constant", benefit,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert tuple type.
    assert(op->getNumResults() == 1);
    Type structType = typeConverter->convertType(op->getResult(0).getType());

    // Undef.
    Value structValue = rewriter.create<UndefOp>(op->getLoc(), structType);

    // Insert values.
    ArrayAttr values = op->getAttr("values").dyn_cast<ArrayAttr>();
    assert(values);
    for (int i = 0; i < static_cast<int>(values.size()); i++) {
      // Create index attribute.
      ArrayAttr indicesAttr = rewriter.getIndexArrayAttr({i});

      // Create constant value op.
      Value valueOp = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), values[i].getType(), values[i]);

      // Insert into struct.
      structValue = rewriter.create<InsertValueOp>(op->getLoc(), structValue,
                                                   valueOp, indicesAttr);
    }

    rewriter.replaceOp(op, structValue);

    return success();
  }
};

struct PrintOpLowering : public ConversionPattern {
  PrintOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, "iterators.print", benefit, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    LLVMStructType structType =
        operands[0].getType().dyn_cast<LLVMStructType>();
    assert(structType);

    // Assemble format string in the form `(%i, %i, ...)`.
    std::string format("(");
    llvm::raw_string_ostream formatStream(format);
    llvm::interleaveComma(structType.getBody(), formatStream, [&](Type type) {
      assert(type == rewriter.getI32Type() && "Only I32 is supported for now");
      formatStream << "%i";
    });
    format += ")\n\0"s;

    // Insert format string as global.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value formatSpec = getOrCreateGlobalString(
        rewriter, Twine("frmt_spec.") + structType.getName(), format, module);

    // Assemble arguments to printf.
    SmallVector<Value, 8> values = {formatSpec};
    ArrayRef<Type> fieldTypes = structType.getBody();
    for (int i = 0; i < static_cast<int>(fieldTypes.size()); i++) {
      // Create index attribute.
      ArrayAttr indicesAttr = rewriter.getIndexArrayAttr({i});

      // Extract from struct.
      Value value = rewriter.create<ExtractValueOp>(op->getLoc(), fieldTypes[i],
                                                    operands[0], indicesAttr);

      values.push_back(value);
    }

    // Generate call to printf.
    FlatSymbolRefAttr printfRef = lookupOrInsertPrintf(rewriter, module);
    rewriter.create<LLVM::CallOp>(op->getLoc(), rewriter.getI32Type(),
                                  printfRef, values);
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SampleInputOp.
//===----------------------------------------------------------------------===//

/// Builds IR that resets the current index to 0. Possible result:
///
/// %0 = llvm.mlir.constant(0 : i32) : i32
/// %1 = llvm.insertvalue %0, %arg0[0 : index] :
///          !llvm.struct<"iterators.sampleInputState", (i32)>
static llvm::SmallVector<Value, 4>
buildOpenBody(SampleInputOp op, RewriterBase &rewriter, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos) {
  // Insert constant zero into state.
  Value zero = rewriter.create<LLVM::ConstantOp>(
      op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
  Value updatedState = rewriter.create<InsertValueOp>(
      op->getLoc(), initialState, zero, rewriter.getIndexArrayAttr({0}));
  return {updatedState};
}

/// Builds IR that produces a tuple with the current index and increments that
/// index. Pseudo-code:
///
/// if currentIndex > max: return {}
/// return currentIndex++
///
/// Possible result:
//
/// %0 = llvm.extractvalue %arg0[0 : index] :
///          !llvm.struct<"iterators.sampleInputState", (i32)>
/// %c4_i32 = arith.constant 4 : i32
/// %1 = arith.cmpi slt, %0, %c4_i32 : i32
/// %2 = scf.if %1 -> (!llvm.struct<"iterators.sampleInputState", (i32)>) {
///   %c1_i32 = arith.constant 1 : i32
///   %5 = arith.addi %0, %c1_i32 : i32
///   %6 = llvm.insertvalue %5, %arg0[0 : index] :
///            !llvm.struct<"iterators.sampleInputState", (i32)>
///   scf.yield %6 : !llvm.struct<"iterators.sampleInputState", (i32)>
/// } else {
///   scf.yield %arg0 : !llvm.struct<"iterators.sampleInputState", (i32)>
/// }
/// %3 = llvm.mlir.undef : !llvm.struct<(i32)>
/// %4 = llvm.insertvalue %0, %3[0 : index] : !llvm.struct<(i32)>
/// return %2, %1, %4 : !llvm.struct<"iterators.sampleInputState", (i32)>,
///                     i1, !llvm.struct<(i32)>
static llvm::SmallVector<Value, 4>
buildNextBody(SampleInputOp op, RewriterBase &rewriter, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  // Extract current index.
  Value currentIndex = rewriter.create<ExtractValueOp>(
      op->getLoc(), rewriter.getI32Type(), initialState,
      rewriter.getIndexArrayAttr({0}));

  // Test if we have reached the end of the range.
  Value four = rewriter.create<arith::ConstantIntOp>(op->getLoc(), /*value=*/4,
                                                     /*width=*/32);
  ArithBuilder ab(rewriter, op.getLoc());
  Value hasNext = ab.slt(currentIndex, four);
  auto ifOp = rewriter.create<scf::IfOp>(
      op->getLoc(), initialState.getType(), hasNext,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        Value one = builder.create<arith::ConstantIntOp>(loc, /*value=*/1,
                                                         /*width=*/32);
        ArithBuilder ab(builder, loc);
        Value updatedCurrentIndex = ab.add(currentIndex, one);
        Value updatedState = builder.create<InsertValueOp>(
            loc, initialState, updatedCurrentIndex,
            builder.getIndexArrayAttr({0}));
        builder.create<scf::YieldOp>(loc, updatedState);
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        builder.create<scf::YieldOp>(loc, initialState);
      });

  // Assemble element that will be returned.
  Value emptyNextElement = rewriter.create<UndefOp>(op->getLoc(), elementType);
  Value nextElement = rewriter.create<InsertValueOp>(
      op->getLoc(), emptyNextElement, currentIndex,
      rewriter.getIndexArrayAttr({0}));

  return {ifOp->getResult(0), hasNext, nextElement};
}

/// Forwards the initial state. The SampleInputOp doesn't do anything on Close.
static llvm::SmallVector<Value, 4>
buildCloseBody(SampleInputOp op, RewriterBase &rewriter, Value initialState,
               ArrayRef<IteratorInfo> upstreamInfos) {
  return {initialState};
}

/// Builds IR that creates an initial iterator state consisting of an
/// (uninitialized) current index. Possible result:
///
/// %0 = llvm.mlir.constant(0 : i32) : i32
/// %1 = llvm.insertvalue %0, %arg0[0 : index] :
///          !llvm.struct<"iterators.sampleInputState", (i32)>
/// return %1 : !llvm.struct<"iterators.sampleInputState", (i32)>
static Value buildStateCreation(SampleInputOp op, RewriterBase &rewriter,
                                LLVM::LLVMStructType stateType,
                                ValueRange upstreamStates) {
  Value initialState = rewriter.create<UndefOp>(op->getLoc(), stateType);
  return initialState;
}

//===----------------------------------------------------------------------===//
// ReduceOp.
//===----------------------------------------------------------------------===//

/// Builds IR that opens the nested upstream iterator. Possible output:
///
/// %0 = llvm.extractvalue %arg0[0 : index] :
///          !llvm.struct<"iterators.reduceState", (...)>
/// %1 = call @iterators.sampleInput.Open.0(%0) : (...) -> ...
/// %2 = llvm.insertvalue %1, %arg0[0 : index] :
///          !llvm.struct<"iterators.reduceState", (...)>
static llvm::SmallVector<Value, 4>
buildOpenBody(ReduceOp op, RewriterBase &rewriter, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos) {
  // Extract upstream state.
  Value initialUpstreamState = rewriter.create<ExtractValueOp>(
      op->getLoc(), upstreamInfos[0].stateType, initialState,
      rewriter.getIndexArrayAttr({0}));

  // Call Open on upstream.
  func::CallOp openCallOp = rewriter.create<func::CallOp>(
      op->getLoc(), upstreamInfos[0].openFunc, upstreamInfos[0].stateType,
      initialUpstreamState);

  // Update upstream state.
  Value updatedState = rewriter.create<InsertValueOp>(
      op->getLoc(), initialState, openCallOp->getResult(0),
      rewriter.getIndexArrayAttr({0}));

  return {updatedState};
}

/// Builds IR that consumes all elements of the upstream iterator and combines
/// them into a single one using the given reduce function. Pseudo-code:
///
/// accumulator = upstream->Next()
/// if !accumulator: return {}
/// while (nextTuple = upstream->Next()):
///     accumulator = reduce(accumulator, nextTuple)
/// return accumulator
///
/// Possible output:
///
/// %0 = llvm.extractvalue %arg0[0 : index] :
///          !llvm.struct<"iterators.reduceState", (...)>
/// %1:3 = call @iterators.sampleInput.Next.0(%0) :
///            (...) -> (..., i1, !llvm.struct<(i32)>)
/// %2:3 = scf.if %1#1 -> (..., i1, !llvm.struct<(i32)>) {
///   %4:3 = scf.while (%arg1 = %1#0, %arg2 = %1#2) :
///              (..., !llvm.struct<(i32)>) ->
///                 (..., !llvm.struct<(i32)>, !llvm.struct<(i32)>) {
///     %5:3 = call @iterators.sampleInput.Next.0(%arg1) :
///                (...) -> (..., i1, !llvm.struct<(i32)>)
///     scf.condition(%5#1) %5#0, %5#2, %arg2 :
///         ..., !llvm.struct<(i32)>, !llvm.struct<(i32)>
///   } do {
///   ^bb0(%arg1: ..., %arg2: !llvm.struct<(i32)>, %arg3: !llvm.struct<(i32)>):
///     // TODO(ingomueller): extend to arbitrary functions
///     %5 = llvm.extractvalue %arg3[0 : index] : !llvm.struct<(i32)>
///     %6 = llvm.extractvalue %arg2[0 : index] : !llvm.struct<(i32)>
///     %7 = arith.addi %6, %5 : i32
///     %8 = llvm.insertvalue %7, %arg2[0 : index] : !llvm.struct<(i32)>
///     scf.yield %arg1, %8 : ..., !llvm.struct<(i32)>
///   }
///   %true = arith.constant true
///   scf.yield %4#0, %true, %4#2 : ..., i1, !llvm.struct<(i32)>
/// } else {
///   scf.yield %1#0, %1#1, %1#2 : ..., i1, !llvm.struct<(i32)>
/// }
/// %3 = llvm.insertvalue %2#0, %arg0[0 : index] :
///          !llvm.struct<"iterators.reduceState", (...)>
static llvm::SmallVector<Value, 4>
buildNextBody(ReduceOp op, RewriterBase &rewriter, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  // Extract upstream state.
  Value initialUpstreamState = rewriter.create<ExtractValueOp>(
      op->getLoc(), upstreamInfos[0].stateType, initialState,
      rewriter.getIndexArrayAttr({0}));

  // Get first result from upstream.
  SmallVector<Type> nextResultTypes = {upstreamInfos[0].stateType,
                                       rewriter.getI1Type(), elementType};
  func::CallOp firstNextCall =
      rewriter.create<func::CallOp>(op->getLoc(), upstreamInfos[0].nextFunc,
                                    nextResultTypes, initialUpstreamState);

  // Check for empty upstream.
  auto ifOp = rewriter.create<scf::IfOp>(
      op->getLoc(), nextResultTypes, firstNextCall->getResult(1),
      /*ifBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        // Create while loop.
        SmallVector<Value> whileInputs = {firstNextCall->getResult(0),
                                          firstNextCall->getResult(2)};
        SmallVector<Type> whileResultTypes = {upstreamInfos[0].stateType,
                                              elementType, elementType};
        scf::WhileOp whileOp = utils::createWhileOp(
            builder, loc, whileResultTypes, whileInputs,
            /*beforeBuilder=*/
            [&](OpBuilder &builder, Location loc,
                Block::BlockArgListType args) {
              Value currentState = args[0];
              Value currentAggregate = args[1];
              func::CallOp nextCall =
                  builder.create<func::CallOp>(loc, upstreamInfos[0].nextFunc,
                                               nextResultTypes, currentState);
              builder.create<scf::ConditionOp>(
                  loc, nextCall->getResult(1),
                  ValueRange{nextCall->getResult(0), nextCall->getResult(2),
                             currentAggregate});
            },
            /*afterBuilder=*/
            [&](OpBuilder &builder, Location loc,
                Block::BlockArgListType args) {
              Value currentState = args[0];
              Value currentAggregate = args[1];
              Value nextElement = args[2];

              ArrayAttr indicesAttr = builder.getIndexArrayAttr(0);
              Value nextValue = builder.create<ExtractValueOp>(
                  loc, builder.getI32Type(), nextElement,
                  builder.getIndexArrayAttr(0));
              Value aggregateValue = builder.create<ExtractValueOp>(
                  loc, builder.getI32Type(), currentAggregate, indicesAttr);
              ArithBuilder ab(builder, loc);
              Value newAggregateValue = ab.add(aggregateValue, nextValue);
              Value newAggregate = builder.create<InsertValueOp>(
                  loc, currentAggregate, newAggregateValue, indicesAttr);

              builder.create<scf::YieldOp>(
                  loc, ValueRange{currentState, newAggregate});
            });

        // The "then" branch of ifOp returns the result of whileOp.
        Value constTrue =
            builder.create<arith::ConstantIntOp>(loc, /*value=*/1, /*width=*/1);
        builder.create<scf::YieldOp>(loc, ValueRange{whileOp->getResult(0),
                                                     constTrue,
                                                     whileOp->getResult(2)});
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        // This branch is taken when the first call to upstream's next does
        // not return anything. In this case, that "end-of-stream" signal is
        // the result of the reduction and the upstream state is final, so we
        // just forward the results of the call to next.
        builder.create<scf::YieldOp>(loc, firstNextCall->getResults());
      });

  // Update state.
  Value finalUpstreamState = ifOp->getResult(0);
  Value finalState = rewriter.create<InsertValueOp>(
      op->getLoc(), initialState.getType(), initialState, finalUpstreamState,
      rewriter.getIndexArrayAttr({0}));

  return {finalState, ifOp->getResult(1), ifOp->getResult(2)};
}

/// Builds IR that closes the nested upstream iterator. Possible output:
///
/// %0 = llvm.extractvalue %arg0[0 : index] :
///          !llvm.struct<"iterators.reduceState", (...)>
/// %1 = call @iterators.sampleInput.Close.0(%0) : (...) -> ...
/// %2 = llvm.insertvalue %1, %arg0[0 : index] :
///          !llvm.struct<"iterators.reduceState", (...)>
static llvm::SmallVector<Value, 4>
buildCloseBody(ReduceOp op, RewriterBase &rewriter, Value initialState,
               ArrayRef<IteratorInfo> upstreamInfos) {
  // Extract upstream state.
  Value initialUpstreamState = rewriter.create<ExtractValueOp>(
      op->getLoc(), upstreamInfos[0].stateType, initialState,
      rewriter.getIndexArrayAttr({0}));

  // Call Open on upstream.
  func::CallOp closeCallOp = rewriter.create<func::CallOp>(
      op->getLoc(), upstreamInfos[0].closeFunc, upstreamInfos[0].stateType,
      initialUpstreamState);

  // Update upstream state.
  Value updatedState = rewriter.create<InsertValueOp>(
      op->getLoc(), initialState, closeCallOp->getResult(0),
      rewriter.getIndexArrayAttr({0}));

  return {updatedState};
}

/// Builds IR that initializes the iterator state with the state of the upstream
/// iterator. Possible output:
///
/// %0 = ...
/// %1 = llvm.mlir.undef : !llvm.struct<"iterators.reduceState", (...)>
/// %2 = llvm.insertvalue %0, %1[0 : index] :
///          !llvm.struct<"iterators.reduceState", (...)>
static Value buildStateCreation(ReduceOp op, RewriterBase &rewriter,
                                LLVM::LLVMStructType stateType,
                                ValueRange upstreamStates) {
  Value undefState = rewriter.create<UndefOp>(op->getLoc(), stateType);
  Value initialState =
      rewriter.create<InsertValueOp>(op.getLoc(), undefState, upstreamStates[0],
                                     rewriter.getIndexArrayAttr({0}));
  return initialState;
}

//===----------------------------------------------------------------------===//
// Helpers for creating Open/Next/Close functions and state creation.
//===----------------------------------------------------------------------===//

using OpenNextCloseBodyBuilder =
    llvm::function_ref<llvm::SmallVector<Value, 4>(RewriterBase &, Value)>;

/// Creates a new Open/Next/Close function at the parent module of originalOp
/// with the given types and name, initializes the function body with a first
/// block, and fills that block with the given builder. Since these functions
/// are only used by the iterators in this module, they are created with private
/// visibility.
static FuncOp
buildOpenNextCloseInParentModule(Operation *originalOp, RewriterBase &rewriter,
                                 Type inputType, TypeRange returnTypes,
                                 SymbolRefAttr funcNameAttr,
                                 OpenNextCloseBodyBuilder bodyBuilder) {
  Location loc = originalOp->getLoc();

  StringRef funcName = funcNameAttr.getLeafReference();
  ModuleOp module = originalOp->getParentOfType<ModuleOp>();
  assert(module);
  MLIRContext *context = rewriter.getContext();

  // Create function op.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  auto visibility = StringAttr::get(context, "private");
  auto funcType = FunctionType::get(context, inputType, returnTypes);
  FuncOp funcOp = rewriter.create<FuncOp>(originalOp->getLoc(), funcName,
                                          funcType, visibility);
  funcOp.setPrivate();

  // Create initial block.
  Block *block =
      rewriter.createBlock(&funcOp.getBody(), funcOp.begin(), inputType, loc);
  rewriter.setInsertionPointToStart(block);

  // Build body.
  Value initialState = block->getArgument(0);
  llvm::SmallVector<Value, 4> returnValues =
      bodyBuilder(rewriter, initialState);
  rewriter.create<func::ReturnOp>(loc, returnValues);

  return funcOp;
}

/// Type-switching proxy for builders of the body of Open functions.
static llvm::SmallVector<Value, 4>
buildOpenBody(Operation *op, RewriterBase &rewriter, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos) {
  return llvm::TypeSwitch<Operation *, llvm::SmallVector<Value, 4>>(op)
      .Case<ReduceOp, SampleInputOp>([&](auto op) {
        return buildOpenBody(op, rewriter, initialState, upstreamInfos);
      });
}

/// Type-switching proxy for builders of the body of Next functions.
static llvm::SmallVector<Value, 4>
buildNextBody(Operation *op, RewriterBase &rewriter, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  return llvm::TypeSwitch<Operation *, llvm::SmallVector<Value, 4>>(op)
      .Case<ReduceOp, SampleInputOp>([&](auto op) {
        return buildNextBody(op, rewriter, initialState, upstreamInfos,
                             elementType);
      });
}

/// Type-switching proxy for builders of the body of Close functions.
static llvm::SmallVector<Value, 4>
buildCloseBody(Operation *op, RewriterBase &rewriter, Value initialState,
               ArrayRef<IteratorInfo> upstreamInfos) {
  return llvm::TypeSwitch<Operation *, llvm::SmallVector<Value, 4>>(op)
      .Case<ReduceOp, SampleInputOp>([&](auto op) {
        return buildCloseBody(op, rewriter, initialState, upstreamInfos);
      });
}

/// Type-switching proxy for builders of iterator state creation.
static Value buildStateCreation(Operation *op, RewriterBase &rewriter,
                                LLVM::LLVMStructType stateType,
                                ValueRange upstreamStates) {
  return llvm::TypeSwitch<Operation *, Value>(op).Case<ReduceOp, SampleInputOp>(
      [&](auto op) {
        return buildStateCreation(op, rewriter, stateType, upstreamStates);
      });
}

/// Creates an Open function for originalOp given the provided opInfo. This
/// function only does plumbing; the actual work is done by
/// `buildOpenNextCloseInParentModule` and `buildOpenBody`.
static FuncOp
buildOpenFuncInParentModule(Operation *originalOp, RewriterBase &rewriter,
                            const IteratorInfo &opInfo,
                            ArrayRef<IteratorInfo> upstreamInfos) {
  return buildOpenNextCloseInParentModule(
      originalOp, rewriter, opInfo.stateType, opInfo.stateType, opInfo.openFunc,
      [&](RewriterBase &rewriter, Value initialState) {
        return buildOpenBody(originalOp, rewriter, initialState, upstreamInfos);
      });
}

/// Creates a Next function for originalOp given the provided opInfo. This
/// function only does plumbing; the actual work is done by
/// `buildOpenNextCloseInParentModule` and `buildNextBody`.
static FuncOp
buildNextFuncInParentModule(Operation *originalOp, RewriterBase &rewriter,
                            const IteratorInfo &opInfo,
                            ArrayRef<IteratorInfo> upstreamInfos) {
  // Compute element type.
  assert(originalOp->getNumResults() == 1);
  StreamType streamType =
      originalOp->getResult(0).getType().dyn_cast<StreamType>();
  assert(streamType);
  Type elementType = streamType.getElementType();

  // Build function.
  return buildOpenNextCloseInParentModule(
      originalOp, rewriter, opInfo.stateType,
      {opInfo.stateType, rewriter.getI1Type(), elementType}, opInfo.nextFunc,
      [&](RewriterBase &rewriter, Value initialState) {
        return buildNextBody(originalOp, rewriter, initialState, upstreamInfos,
                             elementType);
      });
}

/// Creates a Close function for originalOp given the provided opInfo. This
/// function only does plumbing; the actual work is done by
/// `buildOpenNextCloseInParentModule` and `buildCloseBody`.
static FuncOp
buildCloseFuncInParentModule(Operation *originalOp, RewriterBase &rewriter,
                             const IteratorInfo &opInfo,
                             ArrayRef<IteratorInfo> upstreamInfos) {
  return buildOpenNextCloseInParentModule(
      originalOp, rewriter, opInfo.stateType, opInfo.stateType,
      opInfo.closeFunc, [&](RewriterBase &rewriter, Value initialState) {
        return buildCloseBody(originalOp, rewriter, initialState,
                              upstreamInfos);
      });
}

//===----------------------------------------------------------------------===//
// Generic conversion of Iterator ops.
//===----------------------------------------------------------------------===//

/// Converts the given iterator op to LLVM using the converted operands from
/// the upstream iterator. This consists of (1) creating the initial iterator
/// state based on the initial states of the upstream iterators and (2) building
/// the op-specific Open/Next/Close functions.
static FailureOr<Optional<Value>>
convertNonSinkIteratorOp(Operation *op, ArrayRef<Value> operands,
                         RewriterBase &rewriter,
                         const IteratorAnalysis &typeAnalysis) {
  // IteratorInfo for this op.
  auto opInfo = typeAnalysis.getIteratorInfo(op);
  assert(opInfo.hasValue());

  // Assemble IteratorInfo for upstreams.
  llvm::SmallVector<IteratorInfo, 8> upstreamInfos;
  for (Value operand : op->getOperands()) {
    Optional<IteratorInfo> upstreamInfo =
        typeAnalysis.getIteratorInfo(operand.getDefiningOp());
    assert(upstreamInfo.hasValue());
    upstreamInfos.push_back(upstreamInfo.getValue());
  }

  // Build Open/Next/Close functions.
  buildOpenFuncInParentModule(op, rewriter, opInfo.getValue(), upstreamInfos);
  buildNextFuncInParentModule(op, rewriter, opInfo.getValue(), upstreamInfos);
  buildCloseFuncInParentModule(op, rewriter, opInfo.getValue(), upstreamInfos);

  // Create initial state.
  Value initialState =
      buildStateCreation(op, rewriter, opInfo->stateType, operands);

  return {initialState};
}

/// Converts the given sink to LLVM using the converted operands from the root
/// iterator. The current sink consumes the root iterator and prints each
/// element it produces. Pseudo code:
///
/// rootIterator->Open()
/// while (nextTuple = rootIterator->Next())
///   print(nextTuple)
/// rootIterator->Close()
///
/// Possible result:
///
/// %2 = ... // initialize state of root iterator
/// %3 = call @iterators.reduce.Open.1(%2) : (!rootStateType) -> !rootStateType
/// %4:3 = scf.while (%arg0 = %3) :
///            (!rootStateType) -> (!rootStateType, i1, !llvm.struct<(i32)>) {
///   %6:3 = call @iterators.reduce.Next.1(%arg0) :
///              (!rootStateType) -> (!rootStateType, i1, !llvm.struct<(i32)>)
///   scf.condition(%6#1) %6#0, %6#1, %6#2 :
///       !rootStateType, i1, !llvm.struct<(i32)>
/// } do {
/// ^bb0(%arg0: !rootStateType, %arg1: i1, %arg2: !llvm.struct<(i32)>):
///   %6 = llvm.mlir.addressof @frmt_spec.anonymous_tuple :
///            !llvm.ptr<array<6 x i8>>
///   %7 = llvm.mlir.constant(0 : i64) : i64
///   %8 = llvm.getelementptr %6[%7, %7] :
///            (!llvm.ptr<array<6 x i8>>, i64, i64) -> !llvm.ptr<i8>
///   %9 = llvm.extractvalue %arg2[0 : index] : !llvm.struct<(i32)>
///   %10 = llvm.call @printf(%8, %9) : (!llvm.ptr<i8>, i32) -> i32
///   scf.yield %arg0 : !rootStateType
/// }
/// %5 = call @iterators.reduce.Close.1(%4#0) :
///          (!rootStateType) -> !rootStateType
static FailureOr<Optional<Value>>
convertSinkIteratorOp(SinkOp op, ArrayRef<Value> operands,
                      RewriterBase &rewriter,
                      const IteratorAnalysis &typeAnalysis) {
  // Look up IteratorInfo about root iterator.
  assert(operands.size() == 1);
  auto opInfo = typeAnalysis.getIteratorInfo(op->getOperand(0).getDefiningOp());
  assert(opInfo.hasValue());

  // Open root iterator. ------------------------------------------------------
  Value initialState = operands[0];
  func::CallOp openCallOp = rewriter.create<func::CallOp>(
      op->getLoc(), opInfo->openFunc, opInfo->stateType, initialState);

  // Consume root iterator in while loop --------------------------------------
  // Input and return types.
  Type elementType =
      op->getOperand(0).getType().dyn_cast<StreamType>().getElementType();
  SmallVector<Type> nextResultTypes = {opInfo->stateType, rewriter.getI1Type(),
                                       elementType};
  SmallVector<Location> whileResultLocs = {op->getLoc(), op->getLoc(),
                                           op->getLoc()};

  scf::WhileOp whileOp = utils::createWhileOp(
      rewriter, op->getLoc(), nextResultTypes, openCallOp->getOpResult(0),
      /*beforeBuilder=*/
      [&](OpBuilder &builder, Location loc, Block::BlockArgListType args) {
        Value currentState = args[0];
        func::CallOp nextCallOp = builder.create<func::CallOp>(
            loc, opInfo->nextFunc, nextResultTypes, currentState);
        // TODO: Don't pass the boolean to "after"
        builder.create<scf::ConditionOp>(loc, nextCallOp->getResult(1),
                                         nextCallOp->getResults());
      },
      /*afterBuilder=*/
      [&](OpBuilder &builder, Location loc, Block::BlockArgListType args) {
        LLVMStructType structType = elementType.dyn_cast<LLVMStructType>();
        assert(structType && "Only struct types supported for now");

        // Assemble format string in the form `(%i, %i, ...)`.
        std::string format("(%i)\n\0"s);

        // Insert format string as global.
        ModuleOp module = op->getParentOfType<ModuleOp>();
        StringRef tupleName = (structType.isIdentified() ? structType.getName()
                                                         : "anonymous_tuple");
        Value formatSpec = getOrCreateGlobalString(
            builder, Twine("frmt_spec.") + tupleName, format, module);

        // Assemble arguments to printf.
        SmallVector<Value, 8> values = {formatSpec};
        ArrayRef<Type> fieldTypes = structType.getBody();
        for (int i = 0; i < static_cast<int>(fieldTypes.size()); i++) {
          // Create index attribute.
          ArrayAttr indicesAttr = builder.getIndexArrayAttr({i});

          // Extract from into struct.
          Value value = builder.create<ExtractValueOp>(loc, fieldTypes[i],
                                                       args[2], indicesAttr);

          values.push_back(value);
        }

        // Generate call to printf.
        FlatSymbolRefAttr printfRef = lookupOrInsertPrintf(builder, module);
        builder.create<LLVM::CallOp>(loc, builder.getI32Type(), printfRef,
                                     values);

        // Forward iterator state to "before" region.
        Value currentState = args[0];
        builder.create<scf::YieldOp>(loc, currentState);
      });

  Value consumedState = whileOp.getResult(0);

  // Close root iterator. -----------------------------------------------------
  rewriter.create<func::CallOp>(op->getLoc(), opInfo->closeFunc,
                                opInfo->stateType, consumedState);

  return Optional<Value>();
}

/// Converts the given op to LLVM using the converted operands from the upstream
/// iterator. This function is essentially a switch between conversion functions
/// for sink and non-sink iterator ops.
static FailureOr<Optional<Value>>
convertIteratorOp(Operation *op, ArrayRef<Value> operands,
                  RewriterBase &rewriter,
                  const IteratorAnalysis &typeAnalysis) {
  return TypeSwitch<Operation *, FailureOr<Optional<Value>>>(op)
      .Case<SampleInputOp, ReduceOp>([&](auto op) {
        return convertNonSinkIteratorOp(op, operands, rewriter, typeAnalysis);
      })
      .Case<SinkOp>([&](auto op) {
        return convertSinkIteratorOp(op, operands, rewriter, typeAnalysis);
      });
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

/// Converts all iterator ops of a module to LLVM using a custom walker.
static void convertIteratorOps(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  IteratorAnalysis typeAnalysis(module);
  BlockAndValueMapping mapping;

  // Collect all iterator ops in a worklist. Within each block, the iterator
  // ops are seen by the walker in sequential order, so each iterator is added
  // to the worklist *after* all of its upstream iterators.
  SmallVector<Operation *, 16> workList;
  module->walk<WalkOrder::PreOrder>([&](Operation *op) {
    TypeSwitch<Operation *, void>(op).Case<SampleInputOp, ReduceOp, SinkOp>(
        [&](Operation *op) { workList.push_back(op); });
  });

  // Convert iterator ops in worklist order.
  for (Operation *op : workList) {
    // Look up converted operands. The worklist order guarantees that they
    // exist.
    SmallVector<Value, 4> operands;
    for (Value operand : op->getOperands()) {
      Value convertedOperand = mapping.lookup(operand);
      assert(convertedOperand);
      operands.push_back(convertedOperand);
    }

    // Convert this op.
    rewriter.setInsertionPointAfter(op);
    FailureOr<Optional<Value>> conversionResult =
        convertIteratorOp(op, operands, rewriter, typeAnalysis);
    assert(succeeded(conversionResult));

    // Save converted result.
    if (conversionResult->hasValue()) {
      mapping.map(op->getResult(0), conversionResult->getValue());
    }
  }

  // Delete the original, now-converted iterator ops.
  for (auto it = workList.rbegin(); it != workList.rend(); it++) {
    rewriter.eraseOp(*it);
  }
}

void mlir::iterators::populateIteratorsToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConstantTupleLowering, PrintOpLowering>(typeConverter,
                                                       patterns.getContext());
}

void ConvertIteratorsToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Convert iterator ops with custom walker.
  convertIteratorOps(module);

  // Convert the remaining ops of this dialect using dialect conversion.
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithmeticDialect, FuncDialect, LLVMDialect,
                         scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&getContext());
  IteratorsTypeConverter typeConverter;
  populateIteratorsToLLVMConversionPatterns(patterns, typeConverter);
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertIteratorsToLLVMPass() {
  return std::make_unique<ConvertIteratorsToLLVMPass>();
}
