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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

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

def Iterators_ConstantTupleOp
    : Iterators_Base_Op<"constanttuple", [MatchingFieldCountsConstraint]> {
  let summary = "Creates a tuple from the given values";
  let arguments = (ins Iterators_NumericArrayAttr:$values);
  let results = (outs Iterators_TupleOfNumerics:$tuple);
}

def Iterators_PrintTupleOp : Iterators_Base_Op<"printtuple"> {
  let summary = "Prints the elements of a tuple";
  let arguments = (ins Iterators_TupleOfNumerics:$tuple);
}

def Iterators_PrintOp : Iterators_Base_Op<"print"> {
  let summary = "Prints the given element";
  let arguments = (ins Iterators_LLVMStructOfNumerics:$element);
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
                    [](Attribute attr) { return attr.getType(); }
                  )
                )
              )}]>;
def Iterators_ValueMatchesElementType
  : PredOpTrait<"value type matches return type",
                Iterators_ValueMatchesElementTypePred>;

def Iterators_ConstantStreamOp
  : Iterators_Op<"constantstream",
                 [Iterators_ValueMatchesElementType]> {
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
    %0 = "iterators.constantstream"() { value = [[42 : i32]] } :
            () -> (!iterators.stream<!llvm.struct<(i32)>>)
    ```
  }];
  let arguments = (ins Iterators_HomogeneouslyTypedNumericArrayArrayAttr:$value);
  let results = (outs Iterators_StreamOfLLVMStructOfNumerics:$result);
}

/// Looks up the given symbol, which must refer to a FuncOp, in the scope of the
/// given op and returns the function type of that symbol.
class LookupFuncType<string opName, string symbolName>
  : StrFunc<"::mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>("
            "    &$" # opName # ","
            "    " # symbolName # ".dyn_cast<FlatSymbolRefAttr>())"
            "       .getFunctionType()">;

/// A predicate, that is, a function with a signature of the form (T) -> i1.
class Iterators_PredicateFunctionType
  : Type<And<[FunctionType.predicate,
              CPred<"$_self.dyn_cast<FunctionType>().getInputs().size() == 1">,
              CPred<"$_self.dyn_cast<FunctionType>().getResults().size() == 1">,
              SubstLeaves<"$_self",
                      "$_self.dyn_cast<FunctionType>().getResult(0)",
                      I1.predicate>
             ]>,
         "function with signature (T) -> i1",
         "FunctionType">;

/// A FlatSymbolRef referring to a predicate.
def Iterators_PredicateFunctionSymbol
  : Confined<FlatSymbolRefAttr, [
        ReferToOp<"func::FuncOp">,
        AttrConstraint<
          SubstLeaves<"$_self",
                      LookupFuncType<"_op", "$_self.dyn_cast<FlatSymbolRefAttr>()">.result,
                      Iterators_PredicateFunctionType<>.predicate>,
          "referring to a " # Iterators_PredicateFunctionType<>.summary>
      ]>;

def Iterators_FilterOp : Iterators_Op<"filter",
    [AllMatch<["getPredicate().getArgumentTypes().front()",
               "$input.getType().dyn_cast<StreamType>().getElementType()"],
              "the signature of the predicate must match the element type">]> {
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
      Iterators_PredicateFunctionSymbol:$predicateRef
    );
  let results = (outs Iterators_Stream:$result);
  let extraClassDeclaration = [{
    func::FuncOp getPredicate() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          *this, predicateRefAttr());
    }
  }];
}

/// A predicate, that is, a function with a signature of the form (T) -> i1.
class Iterators_MapFunctionType
  : Type<And<[FunctionType.predicate,
              CPred<"$_self.dyn_cast<FunctionType>().getInputs().size() == 1">,
              CPred<"$_self.dyn_cast<FunctionType>().getResults().size() == 1">
             ]>,
         "function with signature (T1) -> T2",
         "FunctionType">;

/// A FlatSymbolRef referring to a predicate.
def Iterators_MapFunctionSymbol
  : Confined<FlatSymbolRefAttr, [
        ReferToOp<"func::FuncOp">,
        AttrConstraint<
          SubstLeaves<"$_self",
                      LookupFuncType<"_op", "$_self.dyn_cast<FlatSymbolRefAttr>()">.result,
                      Iterators_MapFunctionType<>.predicate>,
          "referring to a " # Iterators_MapFunctionType<>.summary>
      ]>;

def Iterators_MapOp : Iterators_Op<"map",
    [AllMatch<["getMapFunc().getArgumentTypes().front()",
               "$input.getType().dyn_cast<StreamType>().getElementType()"],
              "the argument of the map function must match the input element type">,
     AllMatch<["getMapFunc().getResultTypes().front()",
               "$result.getType().dyn_cast<StreamType>().getElementType()"],
              "the return type of the map function must match the result element type">]> {
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
      Iterators_MapFunctionSymbol:$mapFuncRef
    );
  let results = (outs Iterators_Stream:$result);
  let extraClassDeclaration = [{
    func::FuncOp getMapFunc() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          *this, mapFuncRefAttr());
    }
  }];
}

/// A reduce function, that is, a function with a signature of the form
/// (T, T) -> T.
class Iterators_ReduceFunctionType
  : Type<And<[FunctionType.predicate,
              CPred<"$_self.dyn_cast<FunctionType>().getInputs().size() == 2">,
              CPred<"$_self.dyn_cast<FunctionType>().getResults().size() == 1">,
              AllMatchPred<[
                "$_self.dyn_cast<FunctionType>().getInput(0)",
                "$_self.dyn_cast<FunctionType>().getInput(1)",
                "$_self.dyn_cast<FunctionType>().getResult(0)",
              ]>
             ]>,
         "function with signature (T, T) -> T",
         "FunctionType">;

/// A FlatSymbolRef referring to a reduce function.
def Iterators_ReduceFunctionSymbol
  : Confined<FlatSymbolRefAttr, [
        ReferToOp<"func::FuncOp">,
        AttrConstraint<
          SubstLeaves<"$_self",
                      LookupFuncType<"_op", "$_self.dyn_cast<FlatSymbolRefAttr>()">.result,
                      Iterators_ReduceFunctionType<>.predicate>,
          "referring to a " # Iterators_ReduceFunctionType<>.summary>
      ]>;

def Iterators_ReduceOp : Iterators_Op<"reduce",
    [AllMatch<["getReduceFunc().getResultTypes().front()",
               "$input.getType().dyn_cast<StreamType>().getElementType()"],
              "the signature of the reduce function must match the element type">]> {
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
      Iterators_StreamOfLLVMStructOfNumerics:$input,
      Iterators_ReduceFunctionSymbol:$reduceFuncRef
    );
  let results = (outs Iterators_StreamOfLLVMStructOfNumerics:$result);
  let extraClassDeclaration = [{
    func::FuncOp getReduceFunc() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          *this, reduceFuncRefAttr());
    }
  }];
}

/// The sink op is a special op that only consumes a stream of values and
/// produces nothing.
/// It is not marked with Iterators_IteratorOpInterface.
def Iterators_SinkOp : Iterators_Base_Op<"sink"> {
  let summary = "Consume tuples from an iterator";
  let arguments = (ins Iterators_StreamOfLLVMStructOfNumerics:$input);
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSOPS
