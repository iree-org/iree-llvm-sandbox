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
#include "iterators/Conversion/TabularToLLVM/TabularToLLVM.h"
#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Dialect/Tabular/IR/Tabular.h"
#include "iterators/Utils/MLIRSupport.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::iterators;
using namespace mlir::LLVM;
using namespace mlir::func;
using namespace ::iterators;
using namespace std::string_literals;
using IteratorInfo = mlir::iterators::IteratorInfo;

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
    addConversion(TabularTypeConverter::convertTabularViewType);
  }

private:
  /// Maps a TupleType to a corresponding LLVMStructType.
  static Optional<Type> convertTupleType(Type type) {
    if (auto tupleType = type.dyn_cast<TupleType>()) {
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
  MLIRContext *context = builder.getContext();
  Location loc = module->getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Type i8 = builder.getI8Type();
  Type i32 = builder.getI32Type();

  if (module.lookupSymbol<LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  LLVMPointerType charPointerType = LLVMPointerType::get(i8);
  LLVMFunctionType printfFunctionType =
      LLVMFunctionType::get(i32, charPointerType,
                            /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(module.getBody());
  b.create<LLVMFuncOp>("printf", printfFunctionType);
  return SymbolRefAttr::get(context, "printf");
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(OpBuilder &builder, Twine name,
                                     Twine value, ModuleOp module,
                                     bool makeUnique) {
  Location loc = module->getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Type i8 = b.getI8Type();
  Type i64 = b.getI64Type();

  // Determine name of global.
  llvm::SmallString<64> candidateName;
  name.toStringRef(candidateName);
  if (makeUnique) {
    int64_t uniqueNumber = 0;
    while (module.lookupSymbol(candidateName)) {
      (name + "." + Twine(uniqueNumber)).toStringRef(candidateName);
      uniqueNumber++;
    }
  }

  // Create the global at the entry of the module.
  GlobalOp global;
  StringAttr nameAttr = b.getStringAttr(candidateName);
  if (!(global = module.lookupSymbol<GlobalOp>(nameAttr))) {
    StringAttr valueAttr = b.getStringAttr(value);
    LLVMArrayType globalType = LLVMArrayType::get(i8, valueAttr.size());
    OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(module.getBody());
    global = b.create<GlobalOp>(globalType, /*isConstant=*/true,
                                Linkage::Internal, nameAttr, valueAttr,
                                /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = b.create<AddressOfOp>(global);
  Attribute zeroAttr = builder.getI64IntegerAttr(0);
  Value zero = b.create<LLVM::ConstantOp>(i64, zeroAttr);
  return b.create<GEPOp>(loc, LLVMPointerType::get(i8), globalPtr,
                         ArrayRef<Value>({zero, zero}));
}

struct ConstantTupleLowering : public OpConversionPattern<ConstantTupleOp> {
  ConstantTupleLowering(TypeConverter &typeConverter, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(ConstantTupleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Convert tuple type.
    Type tupleType = op.tuple().getType();
    Type structType = typeConverter->convertType(tupleType);

    // Undef.
    Value structValue = rewriter.create<UndefOp>(loc, structType);

    // Insert values.
    ArrayAttr values = op.values();
    for (int i = 0; i < static_cast<int>(values.size()); i++) {
      // Create constant value op.
      Attribute field = values[i];
      Type fieldType = field.getType();
      auto valueOp = rewriter.create<arith::ConstantOp>(loc, fieldType, field);

      // Insert into struct.
      structValue =
          createInsertValueOp(rewriter, loc, structValue, valueOp, {i});
    }

    rewriter.replaceOp(op, structValue);
    return success();
  }
};

/// Applies of 1-to-1 conversion of the given PrintTupleOp to a PrintOp.
struct PrintTupleOpLowering : public OpConversionPattern<PrintTupleOp> {
  PrintTupleOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(PrintTupleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<PrintOp>(op, adaptor.tuple());
    return success();
  }
};

struct PrintOpLowering : public OpConversionPattern<PrintOp> {
  PrintOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();

    auto structType = adaptor.element().getType().cast<LLVMStructType>();

    // Assemble format string in the form `(%lli, %lg, ...)`.
    std::string format("(");
    llvm::raw_string_ostream formatStream(format);
    llvm::interleaveComma(structType.getBody(), formatStream, [&](Type type) {
      // We extend lower-precision values later, so use the maximum width for
      // each type category.
      StringRef specifier;
      assert(type.isSignlessIntOrFloat());
      if (type.isSignlessInteger()) {
        specifier = "%lli";
      } else {
        specifier = "%lg";
      }
      formatStream << specifier;
    });
    format += ")\n\0"s;

    // Insert format string as global.
    auto module = op->getParentOfType<ModuleOp>();
    StringRef tupleName =
        (structType.isIdentified() ? structType.getName() : "anonymous_tuple");
    Value formatSpec = getOrCreateGlobalString(
        rewriter, Twine("frmt_spec.") + tupleName, format, module,
        /*makeUnique=*/!structType.isIdentified());

    // Assemble arguments to printf.
    SmallVector<Value, 8> values = {formatSpec};
    ArrayRef<Type> fieldTypes = structType.getBody();
    for (int i = 0; i < static_cast<int>(fieldTypes.size()); i++) {
      // Extract from struct.
      Type fieldType = fieldTypes[i];
      Value value = createExtractValueOp(rewriter, loc, fieldType,
                                         adaptor.element(), {i});

      // Extend.
      Value extValue;
      if (fieldType.isSignlessInteger()) {
        Type i64 = rewriter.getI64Type();
        extValue = rewriter.create<ZExtOp>(loc, i64, value);
      } else {
        Type f64 = rewriter.getF64Type();
        extValue = rewriter.create<FPExtOp>(loc, f64, value);
      }

      values.push_back(extValue);
    }

    // Generate call to printf.
    FlatSymbolRefAttr printfRef = lookupOrInsertPrintf(rewriter, module);
    rewriter.create<LLVM::CallOp>(loc, i32, printfRef, values);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConstantStreamOp.
//===----------------------------------------------------------------------===//

/// Builds IR that resets the current index to 0. Possible result:
///
/// %0 = llvm.mlir.constant(0 : i32) : i32
/// %1 = iterators.insertvalue %0 into %arg0[0] : !iterators.state<i32>
static Value buildOpenBody(ConstantStreamOp op, OpBuilder &builder,
                           Value initialState,
                           ArrayRef<IteratorInfo> upstreamInfos) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  // Insert constant zero into state.
  Type i32 = b.getI32Type();
  Attribute zeroAttr = b.getI32IntegerAttr(0);
  Value zeroValue = b.create<arith::ConstantOp>(i32, zeroAttr);
  Value updatedState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), zeroValue);

  return updatedState;
}

/// Creates a constant global array with the constant stream data provided in
/// the $value attribute of the given op.
///
/// Possible result:
///
/// llvm.mlir.global internal constant @iterators.constant_stream_data.0() :
///     !llvm.array<4 x !element_type> {
///   %0 = llvm.mlir.undef : !llvm.array<4 x !element_type>
///   // ...
///   llvm.return %n : !llvm.array<4 x !element_type>
/// }
static GlobalOp buildGlobalData(ConstantStreamOp op, OpBuilder &builder,
                                Type elementType) {
  Location loc = op->getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  // Find unique global name.
  auto module = op->getParentOfType<ModuleOp>();
  llvm::SmallString<64> candidateName;
  int64_t uniqueNumber = 0;
  while (true) {
    (Twine("iterators.constant_stream_data.") + Twine(uniqueNumber))
        .toStringRef(candidateName);
    if (!module.lookupSymbol(candidateName)) {
      break;
    }
    uniqueNumber++;
  }
  StringAttr nameAttr = b.getStringAttr(candidateName);

  // Create global op.
  ArrayAttr valueAttr = op.value();
  LLVMArrayType globalType = LLVMArrayType::get(elementType, valueAttr.size());
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(module.getBody());
  auto globalArray =
      b.create<GlobalOp>(globalType, /*isConstant=*/true, Linkage::Internal,
                         nameAttr, Attribute());

  // Create initializer for global. Since arrays of arrays cannot be passed
  // to GlobalOp as attribute, we need to write an initializer that inserts
  // the data from the $value attribute one by one into the global array.
  b.createBlock(&globalArray.getInitializer());
  Value initValue = b.create<UndefOp>(globalType);

  for (auto &elementAttr :
       llvm::enumerate(valueAttr.getAsValueRange<ArrayAttr>())) {
    Value structValue = b.create<UndefOp>(elementType);
    for (auto &fieldAttr : llvm::enumerate(elementAttr.value())) {
      auto value = b.create<LLVM::ConstantOp>(fieldAttr.value().getType(),
                                              fieldAttr.value());
      structValue = createInsertValueOp(
          b, structValue, value, {static_cast<int64_t>(fieldAttr.index())});
    }
    initValue = createInsertValueOp(
        b, initValue, structValue, {static_cast<int64_t>(elementAttr.index())});
  }

  b.create<LLVM::ReturnOp>(initValue);

  return globalArray;
}

/// Builds IR that produces the element at the current index and increments that
/// index. Also creates a global constant with the data. Pseudo-code:
///
/// if currentIndex > max: return {}
/// return data[currentIndex++]
///
/// Possible result:
///
/// llvm.mlir.global internal constant @iterators.constant_stream_data.0() : ...
/// // ...
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<i32>
/// %c4_i32 = arith.constant 4 : i32
/// %1 = arith.cmpi slt, %0, %c4_i32 : i32
/// %2:2 = scf.if %1 -> (!iterators.state<i32>, !element_type) {
///   %c1_i32 = arith.constant 1 : i32
///   %3 = arith.addi %0, %c1_i32 : i32
///   %4 = iterators.insertvalue %3 into %arg0[0] : !iterators.state<i32>
///   %5 = llvm.mlir.addressof @iterators.constant_stream_data.0 :
///            !llvm.ptr<array<4 x !element_type>>
///   %c0_i32 = arith.constant 0 : i32
///   %6 = llvm.getelementptr %5[%c0_i32, %0] :
///            (!llvm.ptr<array<4 x !element_type>>, i32, i32)
///                -> !llvm.ptr<!element_type>
///   %7 = llvm.load %6 : !llvm.ptr<!element_type>
///   scf.yield %4, %7 : !iterators.state<i32>, !element_type
/// } else {
///   %3 = llvm.mlir.undef : !element_type
///   scf.yield %arg0, %3 : !iterators.state<i32>, !element_type
/// }
static llvm::SmallVector<Value, 4>
buildNextBody(ConstantStreamOp op, OpBuilder &builder, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  Location loc = op->getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Type i32 = b.getI32Type();

  // Extract current index.
  Value currentIndex = b.create<iterators::ExtractValueOp>(
      loc, i32, initialState, b.getIndexAttr(0));

  // Test if we have reached the end of the range.
  int64_t numElements = op.value().size();
  Value lastIndex = b.create<arith::ConstantIntOp>(/*value=*/numElements,
                                                   /*width=*/32);
  ArithBuilder ab(b, b.getLoc());
  Value hasNext = ab.slt(currentIndex, lastIndex);
  auto ifOp = b.create<scf::IfOp>(
      TypeRange{initialState.getType(), elementType}, hasNext,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);

        // Increment index and update state.
        Value one = b.create<arith::ConstantIntOp>(/*value=*/1,
                                                   /*width=*/32);
        ArithBuilder ab(b, b.getLoc());
        Value updatedCurrentIndex = ab.add(currentIndex, one);
        Value updatedState = b.create<iterators::InsertValueOp>(
            initialState, b.getIndexAttr(0), updatedCurrentIndex);

        // Load element from global data at current index.
        GlobalOp globalArray = buildGlobalData(op, b, elementType);
        Value globalPtr = b.create<AddressOfOp>(globalArray);
        Value zero = b.create<arith::ConstantIntOp>(/*value=*/0,
                                                    /*width=*/32);
        Value gep = b.create<GEPOp>(LLVMPointerType::get(elementType),
                                    globalPtr, ValueRange{zero, currentIndex});
        Value nextElement = b.create<LoadOp>(gep);

        b.create<scf::YieldOp>(ValueRange{updatedState, nextElement});
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);

        // Don't modify state; return undef element.
        Value nextElement = b.create<UndefOp>(elementType);
        b.create<scf::YieldOp>(ValueRange{initialState, nextElement});
      });

  Value finalState = ifOp->getResult(0);
  Value nextElement = ifOp.getResult(1);
  return {finalState, hasNext, nextElement};
}

/// Forwards the initial state. The ConstantStreamOp doesn't do anything on
/// Close.
static Value buildCloseBody(ConstantStreamOp /*op*/, OpBuilder & /*builder*/,
                            Value initialState,
                            ArrayRef<IteratorInfo> /*upstreamInfos*/) {
  return initialState;
}

/// Builds IR that creates an initial iterator state consisting of an
/// (uninitialized) current index. Possible result:
///
/// %0 = llvm.mlir.constant(0 : i32) : i32
/// %1 = iterators.insertvalue %0 into %arg0[0] : !iterators.state<i32>
static Value buildStateCreation(ConstantStreamOp op,
                                ConstantStreamOp::Adaptor /*adaptor*/,
                                OpBuilder &builder, StateType stateType) {
  return builder.create<UndefStateOp>(op.getLoc(), stateType);
}

//===----------------------------------------------------------------------===//
// FilterOp.
//===----------------------------------------------------------------------===//

/// Builds IR that opens the nested upstream iterator. Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1 = call @iterators.upstream.open.0(%0) : (!nested_state) -> !nested_state
/// %2 = iterators.insertvalue %1 into %arg0[0] :
///          !iterators.state<!nested_state>
static Value buildOpenBody(FilterOp op, OpBuilder &builder, Value initialState,
                           ArrayRef<IteratorInfo> upstreamInfos) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  Type upstreamStateType = upstreamInfos[0].stateType;

  // Extract upstream state.
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Call Open on upstream.
  SymbolRefAttr openFunc = upstreamInfos[0].openFunc;
  auto openCallOp =
      b.create<func::CallOp>(openFunc, upstreamStateType, initialUpstreamState);

  // Update upstream state.
  Value updatedUpstreamState = openCallOp->getResult(0);
  Value updatedState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), updatedUpstreamState);

  return updatedState;
}

/// Builds IR that consumes all elements of the upstream iterator and returns
/// a stream of those that pass the given precicate. Pseudo-code:
///
/// while (nextTuple = upstream->Next()):
///     if precicate(nextTuple):
///         return nextTuple
/// return {}
///
/// Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1:3 = scf.while (%arg1 = %0) :
///            (!nested_state) -> (!nested_state, i1, !element_type) {
///   %3:3 = func.call @iterators.upstream.next.0(%arg1) :
///              (!nested_state) -> (!nested_state, i1, !element_type)
///   %4 = scf.if %3#1 -> (i1) {
///     %7 = func.call @predicate(%3#2) : (!element_type) -> i1
///     scf.yield %7 : i1
///   } else {
///     scf.yield %3#1 : i1
///   }
///   %true = arith.constant true
///   %5 = arith.xori %4, %true : i1
///   %6 = arith.andi %3#1, %5 : i1
///   scf.condition(%6) %3#0, %3#1, %3#2 : !nested_state, i1, !element_type
/// } do {
/// ^bb0(%arg1: !nested_state, %arg2: i1, %arg3: !element_type):
///   scf.yield %arg1 : !nested_state
/// }
/// %2 = iterators.insertvalue %1#0 into %arg0[0] :
///          !iterators.state<!nested_state>
static llvm::SmallVector<Value, 4>
buildNextBody(FilterOp op, OpBuilder &builder, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  // Extract upstream state.
  Type upstreamStateType = upstreamInfos[0].stateType;
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Main while loop.
  Type i1 = b.getI1Type();
  SmallVector<Type> nextResultTypes = {upstreamStateType, i1, elementType};
  scf::WhileOp whileOp = scf::createWhileOp(
      b, nextResultTypes, initialUpstreamState,
      /*beforeBuilder=*/
      [&](OpBuilder &builder, Location loc, Block::BlockArgListType args) {
        ImplicitLocOpBuilder b(loc, builder);

        Value upstreamState = args[0];
        SymbolRefAttr nextFunc = upstreamInfos[0].nextFunc;
        auto nextCall = builder.create<func::CallOp>(
            loc, nextFunc, nextResultTypes, upstreamState);
        Value hasNext = nextCall->getResult(1);
        Value nextElement = nextCall->getResult(2);

        // If we got an element, apply predicate.
        auto ifOp = b.create<scf::IfOp>(
            i1, hasNext,
            /*ifBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              ImplicitLocOpBuilder b(loc, builder);

              // Call predicate.
              auto predicateCall = b.create<func::CallOp>(
                  i1, op.predicateRef(), ValueRange{nextElement});
              Value isMatch = predicateCall->getResult(0);
              b.create<scf::YieldOp>(loc, isMatch);
            },
            /*elseBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              // Note: hasNext is false in this branch.
              builder.create<scf::YieldOp>(loc, hasNext);
            });
        Value hasMatchingNext = ifOp->getResult(0);

        // Build condition: continue loop if (1) we did get an element from
        // upstream (i.e., hasNext) and (2) that element did not match,i.e.,
        // !hasMatchingNext = xor(hasMatchingNext, true).
        Value constTrue =
            b.create<arith::ConstantIntOp>(/*value=*/1, /*width=*/1);
        Value hasNoMatchingNext =
            b.create<arith::XOrIOp>(hasMatchingNext, constTrue);
        Value loopCondition =
            b.create<arith::AndIOp>(hasNext, hasNoMatchingNext);

        b.create<scf::ConditionOp>(loopCondition, nextCall->getResults());
      },
      /*afterBuilder=*/
      [&](OpBuilder &builder, Location loc, Block::BlockArgListType args) {
        Value upstreamState = args[0];
        builder.create<scf::YieldOp>(loc, upstreamState);
      });

  // Update state.
  Value finalUpstreamState = whileOp->getResult(0);
  Value finalState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), finalUpstreamState);
  Value hasNext = whileOp->getResult(1);
  Value nextElement = whileOp->getResult(2);

  return {finalState, hasNext, nextElement};
}

/// Builds IR that closes the nested upstream iterator. Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1 = call @iterators.upstream.close.0(%0) : (!nested_state) -> !nested_state
/// %2 = iterators.insertvalue %1 into %arg0[0] :
///          !iterators.state<!nested_state>
static Value buildCloseBody(FilterOp op, OpBuilder &builder, Value initialState,
                            ArrayRef<IteratorInfo> upstreamInfos) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  Type upstreamStateType = upstreamInfos[0].stateType;

  // Extract upstream state.
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Call Close on upstream.
  SymbolRefAttr closeFunc = upstreamInfos[0].closeFunc;
  auto closeCallOp = b.create<func::CallOp>(closeFunc, upstreamStateType,
                                            initialUpstreamState);

  // Update upstream state.
  Value updatedUpstreamState = closeCallOp->getResult(0);
  return b
      .create<iterators::InsertValueOp>(initialState, b.getIndexAttr(0),
                                        updatedUpstreamState)
      .getResult();
}

/// Builds IR that initializes the iterator state with the state of the upstream
/// iterator. Possible output:
///
/// %0 = ...
/// %1 = iterators.undefstate : !iterators.state<!nested_state>
/// %2 = iterators.insertvalue %0 into %1[0] : !iterators.state<!nested_state>
static Value buildStateCreation(FilterOp op, FilterOp::Adaptor adaptor,
                                OpBuilder &builder, StateType stateType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Value undefState = b.create<UndefStateOp>(stateType);
  Value upstreamState = adaptor.input();
  return b.create<iterators::InsertValueOp>(undefState, b.getIndexAttr(0),
                                            upstreamState);
}

//===----------------------------------------------------------------------===//
// MapOp.
//===----------------------------------------------------------------------===//

/// Builds IR that opens the nested upstream iterator. Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1 = call @iterators.upstream.open.0(%0) : (!nested_state) -> !nested_state
/// %2 = iterators.insertvalue %1 into %arg0[0] :
///          !iterators.state<!nested_state>
static Value buildOpenBody(MapOp op, OpBuilder &builder, Value initialState,
                           ArrayRef<IteratorInfo> upstreamInfos) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  Type upstreamStateType = upstreamInfos[0].stateType;

  // Extract upstream state.
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Call Open on upstream.
  SymbolRefAttr openFunc = upstreamInfos[0].openFunc;
  auto openCallOp =
      b.create<func::CallOp>(openFunc, upstreamStateType, initialUpstreamState);

  // Update upstream state.
  Value updatedUpstreamState = openCallOp->getResult(0);
  Value updatedState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), updatedUpstreamState);

  return updatedState;
}

/// Builds IR that consumes all elements of the upstream iterator and returns
/// a stream where each original element is mapped to/transformed into a new
/// element using the given map function. Pseudo-code:
///
/// if (nextTuple = upstream->Next()):
///     return mapFunc(nextTuple)
/// return {}
///
/// Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1:3 = call @iterators.upstream.next.0(%0) :
///            (!nested_state) -> (!nested_state, i1, !element_type)
/// %2 = scf.if %1#1 -> (!element_type) {
///   %4 = func.call @map_function(%1#2) : (!element_type) -> !element_type
///   scf.yield %4 : !element_type
/// } else {
///   %4 = llvm.mlir.undef : !element_type
///   scf.yield %4 : !element_type
/// }
/// %3 = iterators.insertvalue %1#0 into %arg0[0] :
///          !iterators.state<!nested_state>
static llvm::SmallVector<Value, 4>
buildNextBody(MapOp op, OpBuilder &builder, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  // Extract upstream state.
  Type upstreamStateType = upstreamInfos[0].stateType;
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Extract input element type.
  StreamType inputStreamType = op.input().getType().cast<StreamType>();
  Type inputElementType = inputStreamType.getElementType();

  // Call next.
  Type i1 = b.getI1Type();
  SmallVector<Type> nextResultTypes = {upstreamStateType, i1, inputElementType};
  SymbolRefAttr nextFunc = upstreamInfos[0].nextFunc;
  auto nextCall =
      b.create<func::CallOp>(nextFunc, nextResultTypes, initialUpstreamState);
  Value hasNext = nextCall->getResult(1);
  Value nextElement = nextCall->getResult(2);

  // If we got an element, apply map function.
  auto ifOp = b.create<scf::IfOp>(
      elementType, hasNext,
      /*ifBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        // Apply map function.
        ImplicitLocOpBuilder b(loc, builder);
        auto mapCall = b.create<func::CallOp>(elementType, op.mapFuncRef(),
                                              ValueRange{nextElement});
        Value mappedElement = mapCall->getResult(0);
        b.create<scf::YieldOp>(mappedElement);
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        // Return undefined value.
        ImplicitLocOpBuilder b(loc, builder);
        Value undef = b.create<LLVM::UndefOp>(elementType);
        b.create<scf::YieldOp>(undef);
      });
  Value mappedElement = ifOp.getResult(0);

  // Update state.
  Value finalUpstreamState = nextCall.getResult(0);
  Value finalState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), finalUpstreamState);

  return {finalState, hasNext, mappedElement};
}

/// Builds IR that closes the nested upstream iterator. Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1 = call @iterators.upstream.close.0(%0) : (!nested_state) -> !nested_state
/// %2 = iterators.insertvalue %1 into %arg0[0] :
///          !iterators.state<!nested_state>
static Value buildCloseBody(MapOp op, OpBuilder &builder, Value initialState,
                            ArrayRef<IteratorInfo> upstreamInfos) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  Type upstreamStateType = upstreamInfos[0].stateType;

  // Extract upstream state.
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Call Close on upstream.
  SymbolRefAttr closeFunc = upstreamInfos[0].closeFunc;
  auto closeCallOp = b.create<func::CallOp>(closeFunc, upstreamStateType,
                                            initialUpstreamState);

  // Update upstream state.
  Value updatedUpstreamState = closeCallOp->getResult(0);
  return b.create<iterators::InsertValueOp>(initialState, b.getIndexAttr(0),
                                            updatedUpstreamState);
}

/// Builds IR that initializes the iterator state with the state of the upstream
/// iterator. Possible output:
///
/// %0 = ...
/// %1 = iterators.undefstate : !iterators.state<!nested_state>
/// %2 = iterators.insertvalue %0 into %1[0] : !iterators.state<!nested_state>
static Value buildStateCreation(MapOp op, MapOp::Adaptor adaptor,
                                OpBuilder &builder, StateType stateType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Value undefState = b.create<UndefStateOp>(stateType);
  Value upstreamState = adaptor.input();
  return b.create<iterators::InsertValueOp>(undefState, b.getIndexAttr(0),
                                            upstreamState);
}

//===----------------------------------------------------------------------===//
// ReduceOp.
//===----------------------------------------------------------------------===//

/// Builds IR that opens the nested upstream iterator. Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1 = call @iterators.upstream.open.0(%0) : (!nested_state) -> !nested_state
/// %2 = iterators.insertvalue %1 into %arg0[0] :
///          !iterators.state<!nested_state>
static Value buildOpenBody(ReduceOp op, OpBuilder &builder, Value initialState,
                           ArrayRef<IteratorInfo> upstreamInfos) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  Type upstreamStateType = upstreamInfos[0].stateType;

  // Extract upstream state.
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Call Open on upstream.
  SymbolRefAttr openFunc = upstreamInfos[0].openFunc;
  auto openCallOp =
      b.create<func::CallOp>(openFunc, upstreamStateType, initialUpstreamState);

  // Update upstream state.
  Value updatedUpstreamState = openCallOp->getResult(0);
  Value updatedState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), updatedUpstreamState);

  return updatedState;
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
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1:3 = call @iterators.upstream.next.0(%0) :
///            (!nested_state) -> (!nested_state, i1, !element_type)
/// %2:3 = scf.if %1#1 -> (!nested_state, i1, !element_type) {
///   %4:3 = scf.while (%arg1 = %1#0, %arg2 = %1#2) :
///              (!nested_state, !element_type) ->
///                  (!nested_state, !element_type, !element_type) {
///     %5:3 = func.call @iterators.upstream.next.0(%arg1) :
///                (!nested_state) -> (!nested_state, i1, !element_type)
///     scf.condition(%5#1) %5#0, %arg2, %5#2 :
///         !nested_state, !element_type, !element_type
///   } do {
///   ^bb0(%arg1: !nested_state, %arg2: !element_type, %arg3: !element_type):
///     %5 = func.call @reduce_func(%arg2, %arg3) :
///              (!element_type, !element_type) -> !element_type
///     scf.yield %arg1, %5 : !nested_state, !element_type
///   }
///   %true = arith.constant true
///   scf.yield %4#0, %true, %4#1 : !nested_state, i1, !element_type
/// } else {
///   scf.yield %1#0, %1#1, %1#2 : !nested_state, i1, !element_type
/// }
/// %3 = iterators.insertvalue %2#0 into %arg0[0] :
///          !iterators.state<!nested_state>
static llvm::SmallVector<Value, 4>
buildNextBody(ReduceOp op, OpBuilder &builder, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  // Extract upstream state.
  Type upstreamStateType = upstreamInfos[0].stateType;
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Get first result from upstream.
  Type i1 = b.getI1Type();
  SmallVector<Type> nextResultTypes = {upstreamStateType, i1, elementType};
  SymbolRefAttr nextFunc = upstreamInfos[0].nextFunc;
  auto firstNextCall =
      b.create<func::CallOp>(nextFunc, nextResultTypes, initialUpstreamState);

  // Check for empty upstream.
  Value firstHasNext = firstNextCall->getResult(1);
  auto ifOp = b.create<scf::IfOp>(
      loc, nextResultTypes, firstHasNext,
      /*ifBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);

        // Create while loop.
        Value firstCallUpstreamState = firstNextCall->getResult(0);
        Value firstCallElement = firstNextCall->getResult(2);
        SmallVector<Value> whileInputs = {firstCallUpstreamState,
                                          firstCallElement};
        SmallVector<Type> whileResultTypes = {
            upstreamStateType, // Updated upstream state.
            elementType,       // Accumulator.
            elementType        // Element from last next call.
        };
        scf::WhileOp whileOp = scf::createWhileOp(
            b, whileResultTypes, whileInputs,
            /*beforeBuilder=*/
            [&](OpBuilder &builder, Location loc,
                Block::BlockArgListType args) {
              ImplicitLocOpBuilder b(loc, builder);

              Value upstreamState = args[0];
              Value accumulator = args[1];
              auto nextCall = b.create<func::CallOp>(nextFunc, nextResultTypes,
                                                     upstreamState);

              Value updatedUpstreamState = nextCall->getResult(0);
              Value hasNext = nextCall->getResult(1);
              Value maybeNextElement = nextCall->getResult(2);
              b.create<scf::ConditionOp>(
                  hasNext, ValueRange{updatedUpstreamState, accumulator,
                                      maybeNextElement});
            },
            /*afterBuilder=*/
            [&](OpBuilder &builder, Location loc,
                Block::BlockArgListType args) {
              ImplicitLocOpBuilder b(loc, builder);

              Value upstreamState = args[0];
              Value accumulator = args[1];
              Value nextElement = args[2];

              // Call reduce function.
              auto reduceCall =
                  b.create<func::CallOp>(elementType, op.reduceFuncRef(),
                                         ValueRange{accumulator, nextElement});
              Value newAccumulator = reduceCall->getResult(0);

              b.create<scf::YieldOp>(ValueRange{upstreamState, newAccumulator});
            });

        // The "then" branch of ifOp returns the result of whileOp.
        Value constTrue =
            b.create<arith::ConstantIntOp>(/*value=*/1, /*width=*/1);
        Value updatedUpstreamState = whileOp->getResult(0);
        Value accumulator = whileOp->getResult(1);
        b.create<scf::YieldOp>(
            ValueRange{updatedUpstreamState, constTrue, accumulator});
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
  Value finalState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), finalUpstreamState);
  Value hasNext = ifOp->getResult(1);
  Value nextElement = ifOp->getResult(2);

  return {finalState, hasNext, nextElement};
}

/// Builds IR that closes the nested upstream iterator. Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<!nested_state>
/// %1 = call @iterators.upstream.close.0(%0) : (!nested_state) -> !nested_state
/// %2 = iterators.insertvalue %1 into %arg0[0] :
///          !iterators.state<!nested_state>
static Value buildCloseBody(ReduceOp op, OpBuilder &builder, Value initialState,
                            ArrayRef<IteratorInfo> upstreamInfos) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  Type upstreamStateType = upstreamInfos[0].stateType;

  // Extract upstream state.
  Value initialUpstreamState = b.create<iterators::ExtractValueOp>(
      upstreamStateType, initialState, b.getIndexAttr(0));

  // Call Close on upstream.
  SymbolRefAttr closeFunc = upstreamInfos[0].closeFunc;
  auto closeCallOp = b.create<func::CallOp>(closeFunc, upstreamStateType,
                                            initialUpstreamState);

  // Update upstream state.
  Value updatedUpstreamState = closeCallOp->getResult(0);
  return b
      .create<iterators::InsertValueOp>(initialState, b.getIndexAttr(0),
                                        updatedUpstreamState)
      .getResult();
}

/// Builds IR that initializes the iterator state with the state of the upstream
/// iterator. Possible output:
///
/// %0 = ...
/// %1 = iterators.undefstate : !iterators.state<!nested_state>
/// %2 = iterators.insertvalue %0 into %1[0] : !iterators.state<!nested_state>
static Value buildStateCreation(ReduceOp op, ReduceOp::Adaptor adaptor,
                                OpBuilder &builder, StateType stateType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Value undefState = b.create<UndefStateOp>(loc, stateType);
  Value upstreamState = adaptor.input();
  return b.create<iterators::InsertValueOp>(undefState, b.getIndexAttr(0),
                                            upstreamState);
}

//===----------------------------------------------------------------------===//
// TabularViewToStreamOp.
//===----------------------------------------------------------------------===//

/// Builds IR that (re) sets the current index to zero. Possible output:
///
/// %0 = llvm.mlir.constant(0 : i64) : i64
/// %1 = iterators.insertvalue %0 into %arg0[0] :
///          !iterators.state<i64, !tabular_view_type>
static Value buildOpenBody(TabularViewToStreamOp op, OpBuilder &builder,
                           Value initialState,
                           ArrayRef<IteratorInfo> upstreamInfos) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  // Insert constant zero into state.
  Type i64 = b.getI64Type();
  Attribute zeroAttr = b.getI64IntegerAttr(0);
  Value zeroValue = b.create<arith::ConstantOp>(i64, zeroAttr);
  return b.create<iterators::InsertValueOp>(initialState, b.getIndexAttr(0),
                                            zeroValue);
}

/// Builds IR that assembles an element from the values in the buffers at the
/// current index and increments that index. Pseudo-code: Pseudo-code:
///
/// tuple = (buffer[current_index] for buffer in input)
/// current_index++
/// return tuple
///
/// Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] :
///          !iterators.state<i64, !tabular_view_type>
/// %1 = iterators.extractvalue %arg0[1] :
///          !iterators.state<i64, !tabular_view_type>
/// %2 = llvm.extractvalue %1[0 : index] : !tabular_view_type
/// %3 = arith.cmpi slt, %0, %2 : i64
/// %4:2 = scf.if %3 -> (!iterators.state<i64, !tabular_view_type>,
///                      !element_type) {
///   %c1_i64 = arith.constant 1 : i64
///   %5 = arith.addi %0, %c1_i64 : i64
///   %6 = iterators.insertvalue %5 into %arg0[0] :
///            !iterators.state<i64, !tabular_view_type>
///   %7 = llvm.mlir.undef : !element_type
///   %8 = llvm.extractvalue %1[1 : index] : !tabular_view_type
///   %9 = llvm.getelementptr %8[%0] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
///   %10 = llvm.load %9 : !llvm.ptr<i32>
///   %11 = llvm.insertvalue %10, %7[0 : index] : !element_type
///   scf.yield %6, %11 :
///       !iterators.state<i64, !tabular_view_type>, !element_type
/// } else {
///   %5 = llvm.mlir.undef : !element_type
///   scf.yield %arg0, %5 :
///       !iterators.state<i64, !tabular_view_type>, !element_type
/// }
static llvm::SmallVector<Value, 4>
buildNextBody(TabularViewToStreamOp op, OpBuilder &builder, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  Location loc = op->getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Type i64 = b.getI64Type();

  auto elementStructType = elementType.cast<LLVMStructType>();

  // Extract current index.
  Value currentIndex =
      b.create<iterators::ExtractValueOp>(i64, initialState, b.getIndexAttr(0));

  // Extract input column buffers.
  auto stateType = initialState.getType().cast<StateType>();
  Type structOfInputBuffersType = stateType.getFieldTypes()[1];
  Value structOfInputBuffers = b.create<iterators::ExtractValueOp>(
      structOfInputBuffersType, initialState, b.getIndexAttr(1));

  // Test if we have reached the end of the range.
  Value lastIndex = createExtractValueOp(b, i64, structOfInputBuffers, {0});

  ArithBuilder ab(b, b.getLoc());
  Value hasNext = ab.slt(currentIndex, lastIndex);
  auto ifOp = b.create<scf::IfOp>(
      TypeRange{initialState.getType(), elementType}, hasNext,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);

        // Increment index and update state.
        Value one = b.create<arith::ConstantIntOp>(/*value=*/1,
                                                   /*width=*/64);
        ArithBuilder ab(b, b.getLoc());
        Value updatedCurrentIndex = ab.add(currentIndex, one);
        Value updatedState = b.create<iterators::InsertValueOp>(
            initialState, b.getIndexAttr(0), updatedCurrentIndex);

        // Assemble field values from values at current index of column
        // buffers.
        Value nextElement = b.create<UndefOp>(elementType);
        for (const auto &indexedFieldType :
             llvm::enumerate(elementStructType.getBody())) {
          auto fieldIndex = static_cast<int64_t>(indexedFieldType.index());
          Type fieldType = indexedFieldType.value();
          Type columnPointerType = LLVMPointerType::get(fieldType);

          // Extract column pointer.
          Value columnPtr = createExtractValueOp(
              b, columnPointerType, structOfInputBuffers, {fieldIndex + 1});

          // Get element pointer.
          Value gep = b.create<GEPOp>(columnPointerType, columnPtr,
                                      ValueRange{currentIndex});

          // Load.
          Value fieldValue = b.create<LoadOp>(gep);

          // Insert into next element struct.
          nextElement =
              createInsertValueOp(b, nextElement, fieldValue, {fieldIndex});
        }

        b.create<scf::YieldOp>(ValueRange{updatedState, nextElement});
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        // Don't modify state; return undef element.
        ImplicitLocOpBuilder b(loc, builder);
        Value nextElement = b.create<UndefOp>(elementType);
        b.create<scf::YieldOp>(ValueRange{initialState, nextElement});
      });

  Value finalState = ifOp->getResult(0);
  Value nextElement = ifOp.getResult(1);
  return {finalState, hasNext, nextElement};
}

/// Builds IR that does nothing. The TabularViewToStreamOp does not need to do
/// anything on close.
static Value buildCloseBody(TabularViewToStreamOp /*op*/,
                            OpBuilder & /*rewriter*/, Value initialState,
                            ArrayRef<IteratorInfo> /*upstreamInfos*/) {
  return initialState;
}

/// Builds IR that initializes the iterator state with the columnar input
/// buffers and an undefined current index. Possible output:
///
/// %0 = ...
/// %1 = iterators.undefstate : !iterators.state<i64, !tabular_view_type>
/// %2 = iterators.insertvalue %1[1] (%0 : !tabular_view_type) :
///          <i64, !tabular_view_type>
static Value buildStateCreation(TabularViewToStreamOp op,
                                TabularViewToStreamOp::Adaptor adaptor,
                                OpBuilder &builder, StateType stateType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  // Insert input into iterator state.
  Value iteratorState = b.create<UndefStateOp>(stateType);
  Value input = adaptor.input();
  return b.create<iterators::InsertValueOp>(iteratorState, b.getIndexAttr(1),
                                            input);
}

//===----------------------------------------------------------------------===//
// ValueToStreamOp.
//===----------------------------------------------------------------------===//

/// Builds IR that sets `hasReturned` to false. Possible output:
///
/// %3 = iterators.insertvalue %false into %arg0[1] : !iterators.state<i1, i32>
static Value buildOpenBody(ValueToStreamOp op, OpBuilder &builder,
                           Value initialState,
                           ArrayRef<IteratorInfo> /*upstreamInfos*/) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  // Reset hasReturned to false.
  Value constFalse = b.create<arith::ConstantIntOp>(/*value=*/0, /*width=*/1);
  Value updatedState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), constFalse);

  return updatedState;
}

/// Builds IR that returns the value in the first call and end-of-stream
/// otherwise. Pseudo-code:
///
/// if hasReturned: return {}
/// return value
///
/// Possible output:
///
/// %0 = iterators.extractvalue %arg0[0] : !iterators.state<i1, i32>
/// %true = arith.constant true
/// %1 = arith.xori %true, %0 : i1
/// %2 = iterators.extractvalue %arg0[1] : !iterators.state<i1, i32>
/// %3 = iterators.insertvalue %true into %arg0[0] : !iterators.state<i1, i32>
static llvm::SmallVector<Value, 4>
buildNextBody(ValueToStreamOp op, OpBuilder &builder, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Type i1 = b.getI1Type();

  // Check if the iterator has returned an element already (since it should
  // return one only in the first call to next).
  Value hasReturned =
      b.create<iterators::ExtractValueOp>(i1, initialState, b.getIndexAttr(0));

  // Compute hasNext: we have an element iff we have not returned before, i.e.,
  // iff "not hasReturend". We simulate "not" with "xor true".
  Value constTrue = b.create<arith::ConstantIntOp>(/*value=*/1, /*width=*/1);
  Value hasNext = b.create<arith::XOrIOp>(constTrue, hasReturned);

  // Extract value as next element.
  Value nextElement = b.create<iterators::ExtractValueOp>(
      elementType, initialState, b.getIndexAttr(1));

  // Update state.
  Value finalState = b.create<iterators::InsertValueOp>(
      initialState, b.getIndexAttr(0), constTrue);

  return {finalState, hasNext, nextElement};
}

/// Forwards the initial state. The ValueToStreamOp doesn't do anything on
/// Close.
static Value buildCloseBody(ValueToStreamOp /*op*/, OpBuilder & /*builder*/,
                            Value initialState,
                            ArrayRef<IteratorInfo> /*upstreamInfos*/) {
  return initialState;
}

/// Builds IR that initializes the iterator state with value. Possible output:
///
/// %0 = ...
/// %1 = iterators.undefstate : !iterators.state<i1, i32>
/// %2 = iterators.insertvalue %0 into %1[1] : !iterators.state<i1, i32>
static Value buildStateCreation(ValueToStreamOp op,
                                ValueToStreamOp::Adaptor adaptor,
                                OpBuilder &builder, StateType stateType) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  Value undefState = b.create<UndefStateOp>(loc, stateType);
  Value value = adaptor.input();
  return b.create<iterators::InsertValueOp>(undefState, b.getIndexAttr(1),
                                            value);
}

//===----------------------------------------------------------------------===//
// Helpers for creating Open/Next/Close functions and state creation.
//===----------------------------------------------------------------------===//

using OpenNextCloseBodyBuilder =
    llvm::function_ref<llvm::SmallVector<Value, 4>(OpBuilder &, Value)>;

/// Creates a new Open/Next/Close function at the parent module of originalOp
/// with the given types and name, initializes the function body with a first
/// block, and fills that block with the given builder. Since these functions
/// are only used by the iterators in this module, they are created with private
/// visibility.
static FuncOp
buildOpenNextCloseInParentModule(Operation *originalOp, OpBuilder &builder,
                                 Type inputType, TypeRange returnTypes,
                                 SymbolRefAttr funcNameAttr,
                                 OpenNextCloseBodyBuilder bodyBuilder) {
  Location loc = originalOp->getLoc();
  ImplicitLocOpBuilder b(loc, builder);

  StringRef funcName = funcNameAttr.getLeafReference();
  ModuleOp module = originalOp->getParentOfType<ModuleOp>();
  assert(module);
  MLIRContext *context = b.getContext();

  // Create function op.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());

  auto visibility = StringAttr::get(context, "private");
  auto funcType = FunctionType::get(context, inputType, returnTypes);
  FuncOp funcOp = b.create<FuncOp>(funcName, funcType, visibility);
  funcOp.setPrivate();

  // Create initial block.
  Block *block =
      b.createBlock(&funcOp.getBody(), funcOp.begin(), inputType, loc);
  b.setInsertionPointToStart(block);

  // Build body.
  Value initialState = block->getArgument(0);
  llvm::SmallVector<Value, 4> returnValues = bodyBuilder(b, initialState);
  b.create<func::ReturnOp>(returnValues);

  return funcOp;
}

/// Type-switching proxy for builders of the body of Open functions.
static Value buildOpenBody(Operation *op, OpBuilder &builder,
                           Value initialState,
                           ArrayRef<IteratorInfo> upstreamInfos) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<
          // clang-format off
          ConstantStreamOp,
          FilterOp,
          MapOp,
          ReduceOp,
          TabularViewToStreamOp,
          ValueToStreamOp
          // clang-format on
          >([&](auto op) {
        return buildOpenBody(op, builder, initialState, upstreamInfos);
      });
}

/// Type-switching proxy for builders of the body of Next functions.
static llvm::SmallVector<Value, 4>
buildNextBody(Operation *op, OpBuilder &builder, Value initialState,
              ArrayRef<IteratorInfo> upstreamInfos, Type elementType) {
  return llvm::TypeSwitch<Operation *, llvm::SmallVector<Value, 4>>(op)
      .Case<
          // clang-format off
          ConstantStreamOp,
          FilterOp,
          MapOp,
          ReduceOp,
          TabularViewToStreamOp,
          ValueToStreamOp
          // clang-format on
          >([&](auto op) {
        return buildNextBody(op, builder, initialState, upstreamInfos,
                             elementType);
      });
}

/// Type-switching proxy for builders of the body of Close functions.
static Value buildCloseBody(Operation *op, OpBuilder &builder,
                            Value initialState,
                            ArrayRef<IteratorInfo> upstreamInfos) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<
          // clang-format off
          ConstantStreamOp,
          FilterOp,
          MapOp,
          ReduceOp,
          TabularViewToStreamOp,
          ValueToStreamOp
          // clang-format on
          >([&](auto op) {
        return buildCloseBody(op, builder, initialState, upstreamInfos);
      });
}

/// Type-switching proxy for builders of iterator state creation.
static Value buildStateCreation(IteratorOpInterface op, OpBuilder &builder,
                                StateType stateType, ValueRange operands) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<
          // clang-format off
          ConstantStreamOp,
          FilterOp,
          MapOp,
          ReduceOp,
          TabularViewToStreamOp,
          ValueToStreamOp
          // clang-format on
          >([&](auto op) {
        using OpAdaptor = typename decltype(op)::Adaptor;
        OpAdaptor adaptor(operands, op->getAttrDictionary());
        return buildStateCreation(op, adaptor, builder, stateType);
      });
}

/// Creates an Open function for originalOp given the provided opInfo. This
/// function only does plumbing; the actual work is done by
/// `buildOpenNextCloseInParentModule` and `buildOpenBody`.
static FuncOp
buildOpenFuncInParentModule(Operation *originalOp, OpBuilder &builder,
                            const IteratorInfo &opInfo,
                            ArrayRef<IteratorInfo> upstreamInfos) {
  Type inputType = opInfo.stateType;
  Type returnType = opInfo.stateType;
  SymbolRefAttr funcName = opInfo.openFunc;

  return buildOpenNextCloseInParentModule(
      originalOp, builder, inputType, returnType, funcName,
      [&](OpBuilder &builder,
          Value initialState) -> llvm::SmallVector<Value, 4> {
        return {
            buildOpenBody(originalOp, builder, initialState, upstreamInfos)};
      });
}

/// Creates a Next function for originalOp given the provided opInfo. This
/// function only does plumbing; the actual work is done by
/// `buildOpenNextCloseInParentModule` and `buildNextBody`.
static FuncOp
buildNextFuncInParentModule(Operation *originalOp, OpBuilder &builder,
                            const IteratorInfo &opInfo,
                            ArrayRef<IteratorInfo> upstreamInfos) {
  // Compute element type.
  assert(originalOp->getNumResults() == 1);
  StreamType streamType = originalOp->getResult(0).getType().cast<StreamType>();
  Type elementType = streamType.getElementType();

  // Build function.
  Type i1 = builder.getI1Type();
  Type inputType = opInfo.stateType;
  SymbolRefAttr funcName = opInfo.nextFunc;

  return buildOpenNextCloseInParentModule(
      originalOp, builder, inputType, {opInfo.stateType, i1, elementType},
      funcName, [&](OpBuilder &builder, Value initialState) {
        return buildNextBody(originalOp, builder, initialState, upstreamInfos,
                             elementType);
      });
}

/// Creates a Close function for originalOp given the provided opInfo. This
/// function only does plumbing; the actual work is done by
/// `buildOpenNextCloseInParentModule` and `buildCloseBody`.
static FuncOp
buildCloseFuncInParentModule(Operation *originalOp, OpBuilder &builder,
                             const IteratorInfo &opInfo,
                             ArrayRef<IteratorInfo> upstreamInfos) {
  Type inputType = opInfo.stateType;
  Type returnType = opInfo.stateType;
  SymbolRefAttr funcName = opInfo.closeFunc;

  return buildOpenNextCloseInParentModule(
      originalOp, builder, inputType, returnType, funcName,
      [&](OpBuilder &builder,
          Value initialState) -> llvm::SmallVector<Value, 4> {
        return {
            buildCloseBody(originalOp, builder, initialState, upstreamInfos)};
      });
}

//===----------------------------------------------------------------------===//
// Generic conversion of Iterator ops.
//===----------------------------------------------------------------------===//

/// Converts the given iterator op to LLVM using the converted operands from
/// the upstream iterator. This consists of (1) creating the initial iterator
/// state based on the initial states of the upstream iterators and (2) building
/// the op-specific Open/Next/Close functions.
static Value convert(IteratorOpInterface op, ValueRange operands,
                     IteratorInfo opInfo, ArrayRef<IteratorInfo> upstreamInfos,
                     OpBuilder &builder) {
  // Build Open/Next/Close functions.
  buildOpenFuncInParentModule(op, builder, opInfo, upstreamInfos);
  buildNextFuncInParentModule(op, builder, opInfo, upstreamInfos);
  buildCloseFuncInParentModule(op, builder, opInfo, upstreamInfos);

  // Create initial state.
  StateType stateType = opInfo.stateType;
  return buildStateCreation(op, builder, stateType, operands);
}

/// Converts the given sink to LLVM using the converted input iterator. The
/// current sink consumes the input iterator and prints each element it
/// produces. Pseudo code:
///
/// input->Open()
/// while (nextTuple = input->Next())
///   print(nextTuple)
/// input->Close()
///
/// Possible result:
///
/// %2 = ... // initialize state of input iterator
/// %3 = call @iterators.upstream.open.1(%2) :
///          (!input_state_type) -> !input_state_type
/// %4:3 = scf.while (%arg0 = %3) :
///            (!input_state_type) -> (!input_state_type, i1, !element_type) {
///   %6:3 = call @iterators.upstream.next.1(%arg0) :
///              (!input_state_type) -> (!input_state_type, i1, !element_type)
///   scf.condition(%6#1) %6#0, %6#1, %6#2 :
///       !input_state_type, i1, !element_type
/// } do {
/// ^bb0(%arg0: !input_state_type, %arg1: i1, %arg2: !element_type):
///   "iterators.print"(%arg1) : (!element_type) -> ()
///   scf.yield %arg0 : !input_state_type
/// }
/// %5 = call @iterators.upstream.close.1(%4#0) :
///          (!input_state_type) -> !input_state_type
static SmallVector<Value> convert(SinkOp op, SinkOpAdaptor adaptor,
                                  ArrayRef<IteratorInfo> upstreamInfos,
                                  OpBuilder &rewriter) {
  Location loc = op->getLoc();
  ImplicitLocOpBuilder builder(loc, rewriter);

  // Look up IteratorInfo about input iterator.
  IteratorInfo upstreamInfo = upstreamInfos[0];

  Type stateType = upstreamInfo.stateType;
  SymbolRefAttr openFunc = upstreamInfo.openFunc;
  SymbolRefAttr nextFunc = upstreamInfo.nextFunc;
  SymbolRefAttr closeFunc = upstreamInfo.closeFunc;

  // Open input iterator. ------------------------------------------------------
  Value initialState = adaptor.input();
  auto openCallOp =
      builder.create<func::CallOp>(openFunc, stateType, initialState);
  Value openedUpstreamState = openCallOp->getResult(0);

  // Consume input iterator in while loop. -------------------------------------
  // Input and return types.
  Type elementType = op.input().getType().cast<StreamType>().getElementType();
  Type i1 = builder.getI1Type();
  SmallVector<Type> nextResultTypes = {stateType, i1, elementType};
  SmallVector<Type> whileResultTypes = {stateType, elementType};

  scf::WhileOp whileOp = scf::createWhileOp(
      builder, whileResultTypes, openedUpstreamState,
      /*beforeBuilder=*/
      [&](OpBuilder &builder, Location loc, Block::BlockArgListType args) {
        ImplicitLocOpBuilder b(loc, builder);

        Value currentState = args[0];
        func::CallOp nextCallOp =
            b.create<func::CallOp>(nextFunc, nextResultTypes, currentState);

        Value updatedState = nextCallOp->getResult(0);
        Value hasNext = nextCallOp->getResult(1);
        Value nextElement = nextCallOp->getResult(2);
        b.create<scf::ConditionOp>(hasNext,
                                   ValueRange{updatedState, nextElement});
      },
      /*afterBuilder=*/
      [&](OpBuilder &builder, Location loc, Block::BlockArgListType args) {
        ImplicitLocOpBuilder b(loc, builder);

        Value currentState = args[0];
        Value nextElement = args[1];

        // Print next element.
        b.create<PrintOp>(nextElement);

        // Forward iterator state to "before" region.
        b.create<scf::YieldOp>(currentState);
      });

  Value consumedState = whileOp.getResult(0);

  // Close input iterator. -----------------------------------------------------
  builder.create<func::CallOp>(closeFunc, stateType, consumedState);

  return {};
}

/// Converts the given StreamToValueOp to LLVM using the converted operands.
/// This consists of opening the input iterator, consuming one element (which is
/// the result of this op), and closing it again. Pseudo code:
///
/// upstream->Open()
/// value = upstream->Next()
/// upstream->Close()
///
/// Possible result:
///
/// %0 = ...
/// %1 = call @iterators.upstream.open.0(%0) : (!nested_state) -> !nested_state
/// %2:3 = call @iterators.upstream.next.0(%1) :
///            (!nested_state) -> (!nested_state, i1, !element_type)
/// %3 = call @iterators.upstream.close.0(%2#0) :
///          (!nested_state) -> !nested_state
static SmallVector<Value> convert(StreamToValueOp op,
                                  StreamToValueOpAdaptor adaptor,
                                  ArrayRef<IteratorInfo> upstreamInfos,
                                  OpBuilder &rewriter) {
  Location loc = op->getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);

  // Look up IteratorInfo about the upstream iterator.
  IteratorInfo upstreamInfo = upstreamInfos[0];

  Type stateType = upstreamInfo.stateType;
  SymbolRefAttr openFunc = upstreamInfo.openFunc;
  SymbolRefAttr nextFunc = upstreamInfo.nextFunc;
  SymbolRefAttr closeFunc = upstreamInfo.closeFunc;

  // Open upstream iterator. ---------------------------------------------------
  Value initialState = adaptor.input();
  auto openCallOp = b.create<func::CallOp>(openFunc, stateType, initialState);
  Value openedUpstreamState = openCallOp->getResult(0);

  // Consume one element from upstream iterator --------------------------------
  // Input and return types.
  auto elementType = op.input().getType().cast<StreamType>().getElementType();
  Type i1 = b.getI1Type();
  SmallVector<Type> nextResultTypes = {stateType, i1, elementType};

  func::CallOp nextCallOp =
      b.create<func::CallOp>(nextFunc, nextResultTypes, openedUpstreamState);

  Value consumedUpstreamState = nextCallOp->getResult(0);
  Value hasValue = nextCallOp->getResult(1);
  Value value = nextCallOp->getResult(2);

  // Close upstream iterator. --------------------------------------------------
  b.create<func::CallOp>(closeFunc, stateType, consumedUpstreamState);

  return {value, hasValue};
}

/// Converts the given op to LLVM using the converted operands from the upstream
/// iterator. This function is essentially a switch between conversion functions
/// for sink and non-sink iterator ops.
static SmallVector<Value>
convertIteratorOp(Operation *op, ValueRange operands, OpBuilder &builder,
                  const IteratorAnalysis &iteratorAnalysis) {
  // Look up IteratorInfo for this op.
  IteratorInfo opInfo;
  if (isa<IteratorOpInterface>(op))
    opInfo = iteratorAnalysis.getExpectedIteratorInfo(op);

  // Look up IteratorInfo for all the upstream iterators (i.e., all the defs).
  SmallVector<IteratorInfo> upstreamInfos;
  for (Value operand : op->getOperands()) {
    IteratorInfo upstreamInfo;

    // Get info about operand *iff* it is defined by an iterator op;
    // otherwise, leave IteratorInfo empty.
    if (operand.getDefiningOp())
      if (auto definingOp =
              dyn_cast<IteratorOpInterface>(operand.getDefiningOp()))
        upstreamInfo = iteratorAnalysis.getExpectedIteratorInfo(definingOp);

    upstreamInfos.push_back(upstreamInfo);
  }

  // Call op-specific conversion.
  return TypeSwitch<Operation *, SmallVector<Value>>(op)
      .Case<IteratorOpInterface>([&](auto op) {
        return SmallVector<Value>{
            convert(op, operands, opInfo, upstreamInfos, builder)};
      })
      .Case<SinkOp, StreamToValueOp>([&](auto op) {
        using OpAdaptor = typename decltype(op)::Adaptor;
        OpAdaptor adaptor(operands, op->getAttrDictionary());
        return convert(op, adaptor, upstreamInfos, builder);
      });
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

/// Converts all iterator ops of a module to LLVM. The lowering converts each
/// connected component of iterators to logic that co-executes all iterators in
/// that component. Currently, these connected components have to be shaped as a
/// tree. Roughly speaking, the lowering result consists of three things:
///
/// 1. Nested iterator state types that allow iterators to give up and resume
///    execution by passing control flow between them.
/// 2. Operations that create the initial states.
/// 3. Functions and control flow that executes the logic of one connected
///    component while updating the states.
///
/// Each iterator state is represented as an `!llvm.struct<...>` that constists
/// of an iterator-specific part plus the state of all upstream iterators, i.e.,
/// of all iterators in the transitive use-def chain of the operands of the
/// current iterator, which is inlined into its own state.
///
/// The lowering replaces the original iterator ops with ops that create the
/// initial operator states (which is usually short) plus the logic of the sink
/// operator, which drives the computation of the entire connected components.
/// The logic of the iterators is represented by functions that are added to
/// the current module, and which are called initially by the sink logic and
/// then transitively by the iterators in the connected component.
///
/// More precisely, the computations of each non-sink iterator are expressed as
/// the three functions, `Open`, `Next`, and `Close`, which operate on the
/// iterator's state and which continuously pass control flow between the logic
/// of to and from other iterators:
///
/// * `Open` initializes the computations, typically calling `Open` on the
///   nested states of the current iterator;
/// * `Next` produces the next element in the stream or signals "end of stream",
///   making zero, one, or more calls to `Next` on any of the nested states as
///   required by the logic of the current iterator; and
/// * `Close` cleans up the state if necessary, typically calling `Close` on the
///   nested states of the current iterator.
///
/// The three functions take the current iterator state as an input and return
/// the updated state. (Subsequent bufferization within LLVM presumably converts
/// this to in-place updates.) `Next` also returns the next element in the
/// stream, plus a Boolean that signals whether the element is valid or the end
/// of the stream was reached.
///
/// Iterator states inherently require to contain the states of the all
/// iterators they transitively consume from: Since producing an element for
/// the result stream of an iterator eventually requires to consume elements
/// from the operand streams and the consumption of those elements updates the
/// state of the iterators that produce them, the states of these iterators
/// need to be updated and hence known. If these updates should not be side
/// effects, they have to be reflected in the updated state of the current
/// iterator.
///
/// This means that the lowering of any particular iterator op depends on the
/// transitive use-def chain of its operands. However, it is sufficient to know
/// the state *types*, and, since these states are only forwarded to functions
/// from the upstream iterators (i.e., the iterators that produce the operand
/// streams), they are treated as blackboxes.
///
/// The lowering is done using a custom walker. The conversion happens in two
/// steps:
///
/// 1. The `IteratorAnalysis` computes the nested state of each iterator op
///    and pre-assigns the names of the Open/Next/Close functions of each
///    iterator op. The former provides all information needed from the use-def
///    chain and hences removes need to traverse it for the conversion of each
///    iterator op; the latter establishes the interfaces between each iterator
///    and its downstreams (i.e., consumers).
/// 2. The custom walker traverses the iterator ops in use-def order, converting
///    each iterator in an op-specific way providing the converted operands
///    (which it has walked before) to the conversion logic.
static void convertIteratorOps(ModuleOp module, TypeConverter &typeConverter) {
  IRRewriter rewriter(module.getContext());
  IteratorAnalysis analysis(module, typeConverter);
  BlockAndValueMapping mapping;

  // Collect all iterator ops in a worklist. Within each block, the iterator
  // ops are seen by the walker in sequential order, so each iterator is added
  // to the worklist *after* all of its upstream iterators.
  SmallVector<Operation *, 16> workList;
  module->walk<WalkOrder::PreOrder>([&](Operation *op) {
    TypeSwitch<Operation *, void>(op)
        .Case<IteratorOpInterface, SinkOp, StreamToValueOp>(
            [&](Operation *op) { workList.push_back(op); });
  });

  // Convert iterator ops in worklist order.
  for (Operation *op : workList) {
    rewriter.setInsertionPoint(op);

    // Look up converted operands. The worklist order guarantees that they
    // exist. Use type converter if it isn't produced by an iterator op.
    SmallVector<Value> mappedOperands;
    for (Value operand : op->getOperands()) {
      // Try mapping produced by analysis. This works for operands produced by
      // iterator ops.
      Value mappedOperand = mapping.lookupOrNull(operand);

      // In the other cases (i.e., non-iterator operands), insert unrealized
      // conversion cast to provide conversion with an operand of the converted
      // type similar to the standard dialect conversion.
      if (!mappedOperand) {
        Location loc = op->getLoc();
        Type convertedType = typeConverter.convertType(operand.getType());
        if (convertedType != operand.getType()) {
          mappedOperand = rewriter
                              .create<UnrealizedConversionCastOp>(
                                  loc, convertedType, operand)
                              .getResult(0);
        } else {
          mappedOperand = operand;
        }
      }

      mappedOperands.push_back(mappedOperand);
    }

    // Convert this op.
    SmallVector<Value> converted =
        convertIteratorOp(op, mappedOperands, rewriter, analysis);
    TypeSwitch<Operation *>(op)
        .Case<IteratorOpInterface>([&](auto op) {
          // Iterator op: remember result for conversion of later ops.
          assert(converted.size() == 1 &&
                 "Expected iterator op to be converted to one value.");
          mapping.map(op->getResult(0), converted[0]);
        })
        .Case<StreamToValueOp>([&](auto op) {
          // Special case: uses will not be converted, so replace them.
          assert(converted.size() == 2 &&
                 "Expected StreamToValueOp to be converted to two values.");
          op->getResult(0).replaceAllUsesWith(converted[0]);
          op->getResult(1).replaceAllUsesWith(converted[1]);
        })
        .Case<SinkOp>([&](auto op) {
          // Special case: no result, nothing to do.
          assert(converted.empty() &&
                 "Expected sink op to be converted to no value.");
        });
  }

  // Delete the original, now-converted iterator ops.
  for (auto it = workList.rbegin(); it != workList.rend(); it++)
    rewriter.eraseOp(*it);
}

void mlir::iterators::populateIteratorsToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<
      // clang-format off
      ConstantTupleLowering,
      PrintTupleOpLowering,
      PrintOpLowering
      // clang-format on
      >(typeConverter, patterns.getContext());
}

void ConvertIteratorsToLLVMPass::runOnOperation() {
  auto module = getOperation();
  IteratorsTypeConverter typeConverter;

  // Convert iterator ops with custom walker.
  convertIteratorOps(module, typeConverter);

  // Convert the remaining ops of this dialect using dialect conversion.
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithmeticDialect, LLVMDialect,
                         scf::SCFDialect>();
  target.addLegalOp<ModuleOp, UndefStateOp, iterators::ExtractValueOp,
                    iterators::InsertValueOp>();
  RewritePatternSet patterns(&getContext());

  populateIteratorsToLLVMConversionPatterns(patterns, typeConverter);

  // Add patterns that convert function signature and calls.
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                           typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  // Force application of that pattern if signature is not legal yet.
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return typeConverter.isLegal(op.getOperandTypes());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return typeConverter.isSignatureLegal(op.getCalleeType());
  });

  // Use UnrealizedConversionCast as materializations, which have to be cleaned
  // up by later passes.
  auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return Optional<Value>(cast.getResult(0));
  };
  typeConverter.addSourceMaterialization(addUnrealizedCast);
  typeConverter.addTargetMaterialization(addUnrealizedCast);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertIteratorsToLLVMPass() {
  return std::make_unique<ConvertIteratorsToLLVMPass>();
}
