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
using namespace std::string_literals;

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
static FlatSymbolRefAttr lookupOrInsertPrintf(RewriterBase &rewriter,
                                              ModuleOp module) {
  if (module.lookupSymbol<LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(rewriter.getContext(), "printf");

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  LLVMPointerType charPointerType = LLVMPointerType::get(rewriter.getI8Type());
  LLVMFunctionType printfFunctionType =
      LLVMFunctionType::get(rewriter.getI32Type(), charPointerType,
                            /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVMFuncOp>(module.getLoc(), "printf", printfFunctionType);
  return SymbolRefAttr::get(rewriter.getContext(), "printf");
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
// Helpers for creating Open/Next/Close functions
//===----------------------------------------------------------------------===//

/// Creates a new Open/Next/Close function at the parent module of originalOp
/// with the given types and name and initializes the function body with a first
/// block. Since these functions are only used by the iterators in this module,
/// they are created with private visibility.
static FuncOp createOpenNextCloseInParentModule(Operation *originalOp,
                                                RewriterBase &rewriter,
                                                TypeRange inputTypes,
                                                TypeRange returnTypes,
                                                SymbolRefAttr funcNameAttr) {
  StringRef funcName = funcNameAttr.getLeafReference();
  ModuleOp module = originalOp->getParentOfType<ModuleOp>();
  assert(module);
  MLIRContext *context = rewriter.getContext();

  // Create function op.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  auto visibility = StringAttr::get(context, "private");
  auto funcType = FunctionType::get(context, inputTypes, returnTypes);
  FuncOp funcOp = rewriter.create<FuncOp>(originalOp->getLoc(), funcName,
                                          funcType, visibility);
  funcOp.setPrivate();

  // Create initial block.
  SmallVector<Location, 4> locs(inputTypes.size(), originalOp->getLoc());
  rewriter.createBlock(&funcOp.getBody(), funcOp.begin(), inputTypes, locs);

  return funcOp;
}

/// Creates an Open function for originalOp given the provided opInfo.
static FuncOp
createOpenFuncInParentModule(Operation *originalOp, RewriterBase &rewriter,
                             const IteratorAnalysis::IteratorInfo &opInfo) {
  return createOpenNextCloseInParentModule(originalOp, rewriter,
                                           opInfo.stateType, opInfo.stateType,
                                           opInfo.openFunc);
}

/// Creates an Next function for originalOp given the provided opInfo.
static FuncOp
createNextFuncInParentModule(Operation *originalOp, RewriterBase &rewriter,
                             const IteratorAnalysis::IteratorInfo &opInfo,
                             Type elementType) {
  return createOpenNextCloseInParentModule(
      originalOp, rewriter, opInfo.stateType,
      {opInfo.stateType, rewriter.getI1Type(), elementType}, opInfo.nextFunc);
}

/// Creates an Close function for originalOp given the provided opInfo.
static FuncOp
createCloseFuncInParentModule(Operation *originalOp, RewriterBase &rewriter,
                              const IteratorAnalysis::IteratorInfo &opInfo) {
  return createOpenNextCloseInParentModule(originalOp, rewriter,
                                           opInfo.stateType, opInfo.stateType,
                                           opInfo.closeFunc);
}

//===----------------------------------------------------------------------===//
// Conversion of iterator ops to LLVM
//===----------------------------------------------------------------------===//

/// Converts the given SampleInputOp to LLVM using the converted operands from
/// the upstream iterator.
static FailureOr<Optional<Value>>
convertIteratorOp(SampleInputOp op, ArrayRef<Value> operands,
                  RewriterBase &rewriter,
                  const IteratorAnalysis &typeAnalysis) {
  assert(op->getNumResults() == 1);
  StreamType streamType = op->getResult(0).getType().dyn_cast<StreamType>();
  assert(streamType);
  Type elementType = streamType.getElementType();

  auto maybeOpInfo = typeAnalysis.getIteratorInfo(op);
  assert(maybeOpInfo.hasValue());
  auto opInfo = maybeOpInfo.getValue();

  // Create converted state.
  Value convertedState =
      rewriter.create<UndefOp>(op->getLoc(), opInfo.stateType);

  // Create Open function. --------------------------------------------------
  {
    FuncOp funcOp = createOpenFuncInParentModule(op, rewriter, opInfo);

    Block *block = &funcOp.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    // Insert constant zero into state.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    Value updatedState =
        rewriter.create<InsertValueOp>(op->getLoc(), block->getArgument(0),
                                       zero, rewriter.getIndexArrayAttr({0}));

    // Return.
    rewriter.create<func::ReturnOp>(op->getLoc(), updatedState);
  }

  // Create Next function. --------------------------------------------------
  {
    FuncOp funcOp =
        createNextFuncInParentModule(op, rewriter, opInfo, elementType);

    Block *block = &funcOp.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    Value originalState = block->getArgument(0);
    Value currentIndex = rewriter.create<ExtractValueOp>(
        op->getLoc(), rewriter.getI32Type(), originalState,
        rewriter.getIndexArrayAttr({0}));

    // Test if we have reached the end of the range.
    Value four =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), /*value=*/4,
                                              /*width=*/32);
    ArithBuilder ab(rewriter, op.getLoc());
    Value hasNext = ab.slt(currentIndex, four);
    auto ifOp = rewriter.create<scf::IfOp>(
        op->getLoc(), opInfo.stateType, hasNext,
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          Value one = builder.create<arith::ConstantIntOp>(loc, /*value=*/1,
                                                           /*width=*/32);
          ArithBuilder ab(builder, loc);
          Value updatedCurrentIndex = ab.add(currentIndex, one);
          Value updatedState = builder.create<InsertValueOp>(
              loc, originalState, updatedCurrentIndex,
              builder.getIndexArrayAttr({0}));
          builder.create<scf::YieldOp>(loc, updatedState);
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::YieldOp>(loc, originalState);
        });

    // Assemble element that will be returned.
    Value emptyNextElement =
        rewriter.create<UndefOp>(op->getLoc(), elementType);
    Value nextElement = rewriter.create<InsertValueOp>(
        op->getLoc(), emptyNextElement, currentIndex,
        rewriter.getIndexArrayAttr({0}));

    // Return.
    rewriter.create<func::ReturnOp>(
        op->getLoc(), ValueRange{ifOp->getResult(0), hasNext, nextElement});
  }

  // Create Close function. -------------------------------------------------
  {
    FuncOp funcOp = createCloseFuncInParentModule(op, rewriter, opInfo);

    Block *block = &funcOp.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    // Nothing to do, return.
    rewriter.create<func::ReturnOp>(op->getLoc(), block->getArgument(0));
  }

  return {convertedState};
}

/// Converts the given ReduceOp to LLVM using the converted operands from the
/// upstream iterator.
static FailureOr<Optional<Value>>
convertIteratorOp(ReduceOp op, ArrayRef<Value> operands, RewriterBase &rewriter,
                  const IteratorAnalysis &typeAnalysis) {
  // Convert upstream operation.
  Value upstreamState = operands[0];

  // Collect types.
  assert(op->getNumResults() == 1);
  StreamType streamType = op.result().getType().dyn_cast<StreamType>();
  assert(streamType);
  Type elementType = streamType.getElementType();

  auto maybeUpstreamInfo =
      typeAnalysis.getIteratorInfo(op->getOperand(0).getDefiningOp());
  assert(maybeUpstreamInfo.hasValue());
  auto upstreamInfo = maybeUpstreamInfo.getValue();

  auto maybeOpInfo = typeAnalysis.getIteratorInfo(op);
  assert(maybeOpInfo.hasValue());
  auto opInfo = maybeOpInfo.getValue();

  // Create converted state.
  Value undefState = rewriter.create<UndefOp>(op->getLoc(), opInfo.stateType);
  Value convertedState = rewriter.create<InsertValueOp>(
      op.getLoc(), undefState, upstreamState, rewriter.getIndexArrayAttr({0}));

  // Create Open function. --------------------------------------------------
  {
    FuncOp funcOp = createOpenFuncInParentModule(op, rewriter, opInfo);
    Block *block = &funcOp.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    // Extract upstream state.
    Value initialState = block->getArgument(0);
    Value initialUpstreamState = rewriter.create<ExtractValueOp>(
        op->getLoc(), upstreamInfo.stateType, initialState,
        rewriter.getIndexArrayAttr({0}));

    // Call Open on upstream.
    func::CallOp openCallOp = rewriter.create<func::CallOp>(
        op->getLoc(), upstreamInfo.openFunc, upstreamInfo.stateType,
        initialUpstreamState);

    // Update upstream state.
    Value updatedState = rewriter.create<InsertValueOp>(
        op->getLoc(), initialState, openCallOp->getResult(0),
        rewriter.getIndexArrayAttr({0}));

    // Return.
    rewriter.create<func::ReturnOp>(op->getLoc(), updatedState);
  }

  // Create Next function. --------------------------------------------------
  {
    FuncOp funcOp =
        createNextFuncInParentModule(op, rewriter, opInfo, elementType);
    Block *block = &funcOp.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    // Extract upstream state.
    Value initialState = block->getArgument(0);
    Value initialUpstreamState = rewriter.create<ExtractValueOp>(
        op->getLoc(), upstreamInfo.stateType, initialState,
        rewriter.getIndexArrayAttr({0}));

    // Get first result from upstream.
    SmallVector<Type> nextResultTypes = {upstreamInfo.stateType,
                                         rewriter.getI1Type(), elementType};
    func::CallOp firstNextCall =
        rewriter.create<func::CallOp>(op->getLoc(), upstreamInfo.nextFunc,
                                      nextResultTypes, initialUpstreamState);

    // Check for empty upstream.
    auto ifOp = rewriter.create<scf::IfOp>(
        op->getLoc(), nextResultTypes, firstNextCall->getResult(1),
        /*ifBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          // Create while loop.
          SmallVector<Value> whileInputs = {firstNextCall->getResult(0),
                                            firstNextCall->getResult(2)};
          SmallVector<Type> whileInputTypes = {
              firstNextCall->getResult(0).getType(),
              firstNextCall->getResult(2).getType()};
          SmallVector<Type> whileResultTypes = {upstreamInfo.stateType,
                                                elementType, elementType};
          SmallVector<Location> whileResultLocs = {loc, loc, loc};
          scf::WhileOp whileOp =
              builder.create<scf::WhileOp>(loc, whileResultTypes, whileInputs);

          // Fill the "before" region of whileOp.
          {
            OpBuilder::InsertionGuard guard(builder);
            Block *before = builder.createBlock(&whileOp.getBefore(), {},
                                                whileInputTypes, {loc, loc});
            Value currentState = before->getArgument(0);
            Value currentAggregate = before->getArgument(1);
            func::CallOp nextCall = builder.create<func::CallOp>(
                loc, upstreamInfo.nextFunc, nextResultTypes, currentState);
            builder.create<scf::ConditionOp>(loc, nextCall->getResult(1),
                                             ValueRange{nextCall->getResult(0),
                                                        nextCall->getResult(2),
                                                        currentAggregate});
          } // "before" region

          // Fill the "after" region of whileOp.
          {
            OpBuilder::InsertionGuard guard(builder);
            Block *after = builder.createBlock(
                &whileOp.getAfter(), {}, whileResultTypes, whileResultLocs);

            Value currentState = after->getArgument(0);
            Value currentAggregate = after->getArgument(1);
            Value nextElement = after->getArgument(2);

            // TODO(ingomueller): extend to arbitrary functions
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
          } // "after" region

          // The "then" branch of ifOp returns the result of whileOp.
          Value constTrue = builder.create<arith::ConstantIntOp>(
              loc, /*value=*/1, /*width=*/1);
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
        op->getLoc(), opInfo.stateType, initialState, finalUpstreamState,
        rewriter.getIndexArrayAttr({0}));

    // Return.
    rewriter.create<func::ReturnOp>(
        op->getLoc(),
        ValueRange{finalState, ifOp->getResult(1), ifOp->getResult(2)});
  }

  // Create Close function. -------------------------------------------------
  {
    FuncOp funcOp = createCloseFuncInParentModule(op, rewriter, opInfo);
    Block *block = &funcOp.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    // Extract upstream state.
    Value initialState = block->getArgument(0);
    Value initialUpstreamState = rewriter.create<ExtractValueOp>(
        op->getLoc(), upstreamInfo.stateType, initialState,
        rewriter.getIndexArrayAttr({0}));

    // Call Open on upstream.
    func::CallOp closeCallOp = rewriter.create<func::CallOp>(
        op->getLoc(), upstreamInfo.closeFunc, upstreamInfo.stateType,
        initialUpstreamState);

    // Update upstream state.
    Value updatedState = rewriter.create<InsertValueOp>(
        op->getLoc(), initialState, closeCallOp->getResult(0),
        rewriter.getIndexArrayAttr({0}));

    // Return.
    rewriter.create<func::ReturnOp>(op->getLoc(), updatedState);
  }

  return {convertedState};
}

/// Converts the given sink to LLVM using the converted operands from the root
/// iterator.
static FailureOr<Optional<Value>>
convertIteratorOp(SinkOp op, ArrayRef<Value> operands, RewriterBase &rewriter,
                  const IteratorAnalysis &typeAnalysis) {
  // Convert upstream operation.
  Value initialRootState = operands[0];

  assert(operands.size() == 1);
  auto maybeUpstreamInfo =
      typeAnalysis.getIteratorInfo(op->getOperand(0).getDefiningOp());
  assert(maybeUpstreamInfo.hasValue());
  auto upstreamInfo = maybeUpstreamInfo.getValue();

  // Open root iterator. -----------------------------------------------------
  func::CallOp openCallOp =
      rewriter.create<func::CallOp>(op->getLoc(), upstreamInfo.openFunc,
                                    upstreamInfo.stateType, initialRootState);

  // Consume root operator in while loop -------------------------------------
  // Input and return types.
  Type elementType =
      op->getOperand(0).getType().dyn_cast<StreamType>().getElementType();
  SmallVector<Type> nextResultTypes = {upstreamInfo.stateType,
                                       rewriter.getI1Type(), elementType};
  SmallVector<Location> whileResultLocs = {op->getLoc(), op->getLoc(),
                                           op->getLoc()};

  scf::WhileOp whileOp = rewriter.create<scf::WhileOp>(
      op->getLoc(), nextResultTypes, openCallOp->getOpResult(0));

  // Before region.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *before = rewriter.createBlock(&whileOp.getBefore(), {},
                                         upstreamInfo.stateType, op->getLoc());
    Value currentState = before->getArgument(0);
    func::CallOp nextCallOp = rewriter.create<func::CallOp>(
        op->getLoc(), upstreamInfo.nextFunc, nextResultTypes, currentState);
    // TODO: Don't pass the boolean to "after"
    rewriter.create<scf::ConditionOp>(op->getLoc(), nextCallOp->getResult(1),
                                      nextCallOp->getResults());
  }

  // After region.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *after = rewriter.createBlock(&whileOp.getAfter(), {},
                                        nextResultTypes, whileResultLocs);

    LLVMStructType structType = elementType.dyn_cast<LLVMStructType>();
    assert(structType && "Only struct types supported for now");

    // Assemble format string in the form `(%i, %i, ...)`.
    std::string format("(%i)\n\0"s);

    // Insert format string as global.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    StringRef tupleName =
        (structType.isIdentified() ? structType.getName() : "anonymous_tuple");
    Value formatSpec = getOrCreateGlobalString(
        rewriter, Twine("frmt_spec.") + tupleName, format, module);

    // Assemble arguments to printf.
    SmallVector<Value, 8> values = {formatSpec};
    ArrayRef<Type> fieldTypes = structType.getBody();
    for (int i = 0; i < static_cast<int>(fieldTypes.size()); i++) {
      // Create index attribute.
      ArrayAttr indicesAttr = rewriter.getIndexArrayAttr({i});

      // Extract from into struct.
      Value value = rewriter.create<ExtractValueOp>(
          op->getLoc(), fieldTypes[i], after->getArgument(2), indicesAttr);

      values.push_back(value);
    }

    // Generate call to printf.
    FlatSymbolRefAttr printfRef = lookupOrInsertPrintf(rewriter, module);
    rewriter.create<LLVM::CallOp>(op->getLoc(), rewriter.getI32Type(),
                                  printfRef, values);

    Value currentState = after->getArgument(0);
    rewriter.create<scf::YieldOp>(op->getLoc(), currentState);
  }

  Value consumedState = whileOp.getResult(0);

  // Close root iterator. ----------------------------------------------------
  rewriter.create<func::CallOp>(op->getLoc(), upstreamInfo.closeFunc,
                                upstreamInfo.stateType, consumedState);

  return Optional<Value>();
}

/// Converts the given op to LLVM using the converted operands from the upstream
/// iterator. This function is essentially a type switch to op-specific
/// conversion functions.
static FailureOr<Optional<Value>>
convertIteratorOp(Operation *op, ArrayRef<Value> operands,
                  RewriterBase &rewriter,
                  const IteratorAnalysis &typeAnalysis) {
  return TypeSwitch<Operation *, FailureOr<Optional<Value>>>(op)
      .Case<SampleInputOp, ReduceOp, SinkOp>([&](auto op) {
        return convertIteratorOp(op, operands, rewriter, typeAnalysis);
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

  // Collect all iterator ops in a worklist. Within each block, the iterator ops
  // are seen by the walker in sequential order, so each iterator is added to
  // the worklist *after* all of its upstream iterators.
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
