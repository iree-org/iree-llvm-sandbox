//===-- IteratorsOps.td - Iterators operations definitions -*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSOPS
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSOPS

include "iterators/Dialect/Iterators/IR/IteratorsDialect.td"
include "iterators/Dialect/Iterators/IR/IteratorsTypes.td"
include "iterators/Dialect/Tabular/IR/TabularTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class Iterators_Base_Op<string mnemonic, list<Trait> traits = []> :
    Op<Iterators_Dialect, mnemonic, traits> {
}

class Iterators_Op<string mnemonic, list<Trait> traits = []> :
    Iterators_Base_Op<mnemonic,  traits # [Iterators_IteratorOpInterface]> {
}

//===----------------------------------------------------------------------===//
// Debugging/testing utilities
//===----------------------------------------------------------------------===//

def MatchingFieldCountsPred
    : CPred<[{$tuple.getType().dyn_cast<TupleType>().size() ==
              $values.dyn_cast<ArrayAttr>().size()}]>;
def MatchingFieldCountsConstraint
    : PredOpTrait<"number of values matches tuple size",
                  MatchingFieldCountsPred>;

def Iterators_ConstantTupleOp : Iterators_Base_Op<"constanttuple",
    [MatchingFieldCountsConstraint,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Creates a tuple from the given values";
  let arguments = (ins Iterators_PrintableNumericArrayAttr:$values);
  let results = (outs Iterators_TupleOfPrintableTypes:$tuple);
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "tuple");
    }
  }];
}

def Iterators_PrintOp : Iterators_Base_Op<"print", [
    PredOpTrait<"only one of 'constant' and 'value' may be set",
      CPred<[{($constant.empty() || !llvm::cast<PrintOp>($_op).getValue())}]>>
    ]> {
  let summary = "Prints the given value or string";
  let arguments = (ins
    Optional<Iterators_PrintableType>:$value,
    DefaultValuedAttr<StrAttr, [{""}]>:$constant,
    UnitAttr:$nonl
  );
  let assemblyFormat = [{
    `(` ($constant^):($value)? (`nonl` $nonl^)? `)`
      attr-dict (`:` type($value)^)?
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Value":$value), [{
      $_state.addOperands(value);
    }]>,
    OpBuilder<(ins "::llvm::StringRef": $constant), [{
      $_state.addAttribute("constant", $_builder.getStringAttr(constant));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// High-level iterators
//===----------------------------------------------------------------------===//

/// Verifies that the element types of nested arrays in the $value array
/// correspond to the types of the LLVM-struct element type of the $result
/// Stream.
def Iterators_ValueMatchesElementTypePred
  : CPred<[{$value.dyn_cast<ArrayAttr>().size() == 0 ||
            $result.getType().dyn_cast<StreamType>().getElementType() ==
              ::mlir::LLVM::LLVMStructType::getLiteral(
                $result.getType().getContext(),
                ::llvm::SmallVector<Type>(
                  ::llvm::map_range(
                    $value.dyn_cast<::mlir::ArrayAttr>().begin()->dyn_cast<::mlir::ArrayAttr>(),
                    [](Attribute attr) { return attr.cast<TypedAttr>().getType(); }
                  )
                )
              )}]>;
def Iterators_ValueMatchesElementType
  : PredOpTrait<"value type matches return type",
                Iterators_ValueMatchesElementTypePred>;

def Iterators_ConstantStreamOp : Iterators_Op<"constantstream",
    [Iterators_ValueMatchesElementType,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Produce a statically defined stream of elements";
  let description = [{
    Produces a stream of LLVM structs given in the array of arrays attribute
    (each inner array being returned as a literal LLVM struct with the values
    and types of the elements of that array). The inner arrays all have to have
    matching types, i.e., the element at position i has to be the same for all
    inner arrays, and the element type of the return Stream has to be the
    corresponding literal LLVM struct. An empty array is allowed (in which case
    the return Stream does not need to match anything).

    Example:
    ```mlir
    %constantstream = "iterators.constantstream"() { value = [[42 : i32]] } :
            () -> (!iterators.stream<!llvm.struct<(i32)>>)
    ```
  }];
  let arguments = (ins Iterators_HomogeneouslyTypedLLVMNumericArrayArrayAttr:$value);
  let results = (outs Iterators_StreamOfLLVMStructOfNumerics:$result);
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "constantstream");
    }
  }];
}

/// Looks up the given symbol, which must refer to a FuncOp, in the scope of the
/// given op and returns the function type of that symbol.
class LookupFuncType<string opName, string symbolName>
  : StrFunc<"::mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>("
            "    &$" # opName # ","
            "    " # symbolName # ".dyn_cast<FlatSymbolRefAttr>())"
            "       .getFunctionType()">;

def Iterators_FilterOp : Iterators_Op<"filter",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Filter the elements of the input using a predicate";
  let description = [{
    Reads the elements of its operand stream and produces a stream consisting of
    those that match the provided predicate (i.e., those on which the provided
    predicate returns true).

    Example:
    ```mlir
    %filtered = "iterators.filter"(%input) {predicateRef = @is_positive} :
                   (!iterators.stream<i32>) -> (!iterators.stream<i32>)
    ```
  }];
  let arguments = (ins
      Iterators_Stream:$input,
      FlatSymbolRefAttr:$predicateRef
    );
  let results = (outs Iterators_Stream:$result);
  let extraClassDeclaration = [{
    /// Lookup the predicate in the nearest symbol table and return the
    /// corresponding FuncOp if it exists. It is not safe to call this function
    /// during verification.
    func::FuncOp getPredicate() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          *this, getPredicateRefAttr());
    }
  }];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "filtered");
    }

    /// Implement SymbolUserOpInterface.
    LogicalResult $cppClass::verifySymbolUses(SymbolTableCollection &symbolTable) {
      Type i1 = IntegerType::get(getContext(), 1);

      func::FuncOp funcOp = getPredicate();
      if (!funcOp)
        return emitOpError() << "uses the symbol '" << getPredicateRef()
                             << "', which does not reference a valid function";

      FunctionType funcType = funcOp.getFunctionType();
      if (funcType.getNumInputs() != 1 ||
          funcType.getNumResults() != 1 ||
          funcType.getResult(0) != i1)
        return emitOpError() << "uses the symbol '" << getPredicateRef()
                             << "', which does not refer to a function with a "
                             << "signature of the form (T) -> i1";

      if (getPredicate().getFunctionType().getInput(0) !=
          getInput().getType().cast<StreamType>().getElementType())
        return emitOpError() << "uses the symbol '" << getPredicateRef()
                             << "', whose return type does not match the "
                             << "element type";

      return success();
    }
  }];
}

def Iterators_MapOp : Iterators_Op<"map",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Maps (or transforms) each element of the input to another one.";
  let description = [{
    Reads the elements of its operand stream and maps each of them to a new
    element, i.e., transforms the input stream elementwise.

    Example:
    ```mlir
    %mapped = "iterators.map"(%input) {mapFuncRef = @abs} :
                   (!iterators.stream<i32>) -> (!iterators.stream<i32>)
    ```
  }];
  let arguments = (ins
      Iterators_Stream:$input,
      FlatSymbolRefAttr:$mapFuncRef
    );
  let results = (outs Iterators_Stream:$result);
  let extraClassDeclaration = [{
    /// Lookup the map function in the nearest symbol table and return the
    /// corresponding FuncOp if it exists. It is not safe to call this function
    /// during verification.
    func::FuncOp getMapFunc() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          *this, getMapFuncRefAttr());
    }
  }];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "mapped");
    }

    /// Implement SymbolUserOpInterface.
    LogicalResult $cppClass::verifySymbolUses(SymbolTableCollection &symbolTable) {
      func::FuncOp funcOp = getMapFunc();
      if (!funcOp)
        return emitOpError() << "uses the symbol '" << getMapFuncRef()
                             << "', which does not reference a valid function";

      FunctionType funcType = funcOp.getFunctionType();
      if (funcType.getNumInputs() != 1 ||
          funcType.getNumResults() != 1)
        return emitOpError() << "uses the symbol '" << getMapFuncRef()
                             << "', which does not refer to a function with a "
                             << "signature of the form (T1) -> T2";

      if (funcType.getInput(0) !=
          getInput().getType().cast<StreamType>().getElementType())
        return emitOpError() << "uses the symbol '" << getMapFuncRef()
                             << "', whose return type does not match the input "
                             << "element type";

      if (funcType.getResult(0) !=
          getResult().getType().cast<StreamType>().getElementType())
        return emitOpError() << "uses the symbol '" << getMapFuncRef()
                             << "', whose argument type does not match the "
                             << "result element type";

      return success();
    }
  }];
}

def Iterators_ReduceOp : Iterators_Op<"reduce",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Reduce the input to a single tuple";
  let description = [{
    Reads the elements of its operand stream and reduces them to a single
    element using the provided reduce function. The result stream is empty iff
    the operand stream is empty. Otherwise, the elements are reduced pairwise in
    an implementation-defined order until a single element is left, which
    constitutes the result stream. This is only deterministic if the reduce
    function is associative.

    Example:
    ```mlir
    %reduced = "iterators.reduce"(%input) {reduceFuncRef = @sum} :
                   (!iterators.stream<i32>) -> (!iterators.stream<i32>)
    ```
  }];
  let arguments = (ins
      Iterators_Stream:$input,
      FlatSymbolRefAttr:$reduceFuncRef
    );
  let results = (outs Iterators_Stream:$result);
  let extraClassDeclaration = [{
    /// Lookup the reduce function in the nearest symbol table and return the
    /// corresponding FuncOp if it exists. It is not safe to call this function
    /// during verification.
    func::FuncOp getReduceFunc() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          *this, getReduceFuncRefAttr());
    }
  }];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "reduced");
    }

    /// Implement SymbolUserOpInterface.
    LogicalResult $cppClass::verifySymbolUses(SymbolTableCollection &symbolTable) {
      func::FuncOp funcOp = getReduceFunc();
      if (!funcOp)
        return emitOpError() << "uses the symbol '" << getReduceFuncRef()
                             << "', which does not reference a valid function";

      FunctionType funcType = funcOp.getFunctionType();
      if (funcType.getNumInputs() != 2 ||
          funcType.getNumResults() != 1 ||
          !llvm::all_equal({funcType.getInput(0),
                            funcType.getInput(1),
                            funcType.getResult(0)}))
        return emitOpError() << "uses the symbol '" << getReduceFuncRef()
                             << "', which does not refer to a function with a "
                             << "signature of the form (T, T) -> T";

      if (funcType.getResult(0) !=
          getResult().getType().cast<StreamType>().getElementType())
        return emitOpError() << "uses the symbol '" << getReduceFuncRef()
                             << "', whose result type does not match the "
                             << "element type";

      return success();
    }
  }];
}

def Iterators_TabularViewToStreamOp : Iterators_Op<"tabular_view_to_stream",
    [TypesMatchWith<"element type of input stream must match result type",
                    "result", "input",
                    "TabularViewType::get("
                    "    $_self.getContext(),"
                    "    $_self.cast<StreamType>().getElementType()"
                    "          .cast<LLVM::LLVMStructType>().getBody())">,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  // TODO(ingomueller): Get rid of the LLVM dependency after designing a row
  //                    type or similar.
  let summary = "Extracts the tuples from a tabular view as LLVM structs";
  let description = [{
    Produces a stream of LLVM structs from a given `tabular_view` (i.e., "scans"
    the given `tabular_view`). Each LLVM struct represents one row, and the op
    produces all rows in ascending order.

    The current dependency on LLVM is not elegant and only a temporary solution
    before we get a more high-level `row` type or similar.

    Example:
    ```mlir
    %fromtabview = iterators.tabular_view_to_stream %view
                       to !iterators.stream<!llvm.struct<(!t1, ..., !tn)>>
    ```
  }];
  let arguments = (ins Tabular_TabularView:$input);
  let results = (outs Iterators_StreamOfLLVMStructOfNumerics:$result);
  let assemblyFormat = "$input attr-dict `to` type($result)";
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "fromtabview");
    }
  }];
}

/// The sink op is a special op that only consumes a stream of values and
/// produces nothing.
/// It is not marked with Iterators_IteratorOpInterface.
def Iterators_SinkOp : Iterators_Base_Op<"sink"> {
  let summary = "Consume tuples from an iterator";
  let arguments = (ins Iterators_StreamOfPrintableElements:$input);
}

def Iterators_StreamToValueOp : Iterators_Base_Op<"stream_to_value",
    [TypesMatchWith<"element type of input stream must match result type",
                    "input", "result",
                    "$_self.cast<StreamType>().getElementType()">]> {
  let summary = "Produce a value from the first element of the input stream";
  let description = [{
    Consumes the first element of the given input stream and returns it. The
    remaining elements from the input stream are not consumed.

    The purpose of this op is to pass from "stream land" to "value land," i.e.,
    to enable non-iterator ops to consume the results of a tree of iterator ops.

    Example:
    ```mlir
    %stream = ...
    %value = iterators.stream_to_value %stream : !iterators.stream<i32>
    ```
  }];
  let arguments = (ins Iterators_Stream:$input);
  let results = (outs AnyType:$result, I1:$hasResult);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input))";
}

def Iterators_ValueToStreamOp : Iterators_Op<"value_to_stream",
    [TypesMatchWith<"element type of result stream must match input type",
                    "result", "input",
                    "$_self.cast<StreamType>().getElementType()">,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Produce a stream with the given value as a single element";
  let description = [{
    Produces a stream consisting of a single element, namely the value given as
    input.

    The purpose of this op is to pass from "value land" to "stream land," i.e.,
    to enable iterator ops to consume arbitrary values as a (singleton) stream.

    Example:
    ```mlir
    %value = arith.constant 42 : i32
    %singletonstream = iterators.value_to_stream %value : !iterators.stream<i32>
    ```
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs Iterators_Stream:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($result))";
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "singletonstream");
    }
  }];
}

//===----------------------------------------------------------------------===//
// Ops related to Iterator bodies.
//===----------------------------------------------------------------------===//

def Iterators_CreateStateOp : Iterators_Base_Op<"createstate",
    [Pure,
     TypesMatchWith<"types of provided values much match the field types of "
                    "the resulting state", "result", "values",
                    "$_self.cast<StateType>().getFieldTypes()">,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create an iterator state out of the given values";
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs Iterators_State:$result);
  let assemblyFormat = "`(` $values `)` attr-dict `:` qualified(type($result))";
  let description = [{
    Create a new iterator state filled with the given values.

    This is similar to `complex.create`.

    Example:

    ```
    %i32 = ...
    %tensor = ...
    %state = iterators.createstate(%i32, %tensor) :
                 !iterators.state<i32, tensor<?xi32>>
    ```
  }];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "state");
    }
  }];
}

def Iterators_ExtractValueOp : Iterators_Base_Op<"extractvalue", [
    Pure,
    PredOpTrait<"index must exist in state",
      CPred<"static_cast<uint64_t>($index.getSExtValue())"
            "    < $state.getType().cast<StateType>().getFieldTypes().size()">>,
    AllMatch<["$state.getType().cast<StateType>().getFieldTypes()"
              "    [$index.getSExtValue()]",
              "$result.getType()"],
             "the return type must match the field type at the given index">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]> {
  let summary = "Extract the field value of the state";
  let arguments = (ins Iterators_State:$state, IndexAttr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $state `[` $index `]` attr-dict `:` qualified(type($state))
  }];
  let description = [{
    Extracts the value of the given iterator state at the given index.

    This is similar to `llvm.extractvalue` for `llvm.struct`.

    Example:

    ```
    %state = ...
    %value = iterators.extractvalue %state[0] :
                 !iterators.state<i32, tensor<?xi32>>
    ```
  }];
}

def Iterators_InsertValueOp : Iterators_Base_Op<"insertvalue", [
    Pure,
    PredOpTrait<
      "index must exist in state",
      CPred<"static_cast<uint64_t>($index.getSExtValue())"
            "    < $state.getType().cast<StateType>().getFieldTypes().size()">>,
    AllMatch<["$state.getType().cast<StateType>().getFieldTypes()"
              "    [$index.getSExtValue()]",
              "$value.getType()"],
             "the value type must match the field type at the given index">,
    AllTypesMatch<["state", "result"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = "Insert a field value into the state";
  let arguments = (ins Iterators_State:$state, IndexAttr:$index, AnyType:$value);
  let results = (outs Iterators_State:$result);
  let assemblyFormat = [{
    $value `into` $state `[` $index `]` attr-dict `:` qualified(type($state))
      custom<InsertValueType>(type($value), ref(type($state)), ref($index))
  }];
  let description = [{
    Inserts the given value into the given iterator state at the given index.

    This is similar to `llvm.insertvalue` for `llvm.struct`.

    Example:

    ```
    %initial_state = ...
    %value = ...
    %updated_state = iterators.insertvalue %initial_state[0] (%value : i32) :
                         !iterators.state<i32, tensor<?xi32>>
    ```
  }];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "state");
    }
  }];
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSOPS
