//===-- IteratorsTypes.td - Iterators dialect types --------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES

include "iterators/Dialect/Iterators/IR/IteratorsDialect.td"
include "iterators/Dialect/Iterators/IR/IteratorsInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

// Base class for Iterators dialect types.
class Iterators_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Iterators_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Streams (i.e., the data types passed between iterators)
//===----------------------------------------------------------------------===//

def Iterators_Stream : Iterators_Type<"Stream", "stream"> {
  let summary = "Stream of elements of the given type";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";
}

/// List of LLVM-compatible numeric types.
def Iterators_NumericTypes {
  list<Type> types = [I1, I8, I16, I32, I64,
                      F16, F32, F64];
}

/// Any LLVM-compatible numeric types..
def Iterators_AnyNumeric
  : AnyTypeOf<Iterators_NumericTypes.types>;

/// A tuple consisting only of numeric types.
def Iterators_TupleOfNumerics : TupleOf<Iterators_NumericTypes.types>;

/// Attribute of numeric type.
def Iterators_NumericAttr
  : Attr<SubstLeaves<"$_self", "$_self.getType()",
                     Iterators_AnyNumeric.predicate>,
        "any numeric type">;

/// Array of numeric attributes.
def Iterators_NumericArrayAttr
  : TypedArrayAttrBase<Iterators_NumericAttr, "array of numeric types">;

/// An LLVMStructType where the body only uses the provided types.
class Iterators_LLVMStructOf<list<Type> allowedTypes>
  : MixedContainerType<AnyTypeOf<allowedTypes>, LLVM_AnyStruct.predicate,
                       "$_self.cast<::mlir::LLVM::LLVMStructType>().getBody()",
                       "LLVM struct">;

/// An LLVMStructType where the body only uses numeric types.
def Iterators_LLVMStructOfNumerics
  : Iterators_LLVMStructOf<Iterators_NumericTypes.types>;

/// An LLVMStructType where the body consists of a single I32
def Iterators_LLVMStructOfSingleI32
  : Type<And<[Iterators_LLVMStructOf<[I32]>.predicate,
              CPred<[{$_self.dyn_cast<::mlir::LLVM::LLVMStructType>()
                            .getBody().size() == 1}]>]>,
         "LLVM struct with single I32 field">;

/// An Iterators stream of elements of the given type.
class Iterators_StreamOf<Type elementType>
  : Type<And<[Iterators_Stream.predicate,
              SubstLeaves<"$_self",
                          "$_self.dyn_cast<StreamType>().getElementType()",
                          elementType.predicate>]>,
         "stream with elements of type " # elementType.summary>;

/// An Iterators stream of LLVM structs consisting of a single I32.
def Iterators_StreamOfLLVMStructOfSingleI32
  : Iterators_StreamOf<Iterators_LLVMStructOfSingleI32>;

/// An Iterators stream of LLVM structs consisting of numerics.
def Iterators_StreamOfLLVMStructOfNumerics
  : Iterators_StreamOf<Iterators_LLVMStructOfNumerics>;

/// Predicate to verify that a named argument or result's stream type matches a
/// given type.
class Iterators_IsStreamOfPred<string name, Type type>
  : SubstLeaves<"$_self", "$" # name # ".getType()",
                Iterators_StreamOf<type>.predicate>;
class Iterators_IsStreamOf<string name, Type type>
  : PredOpTrait<"'" # name # "' has element type " # type.summary,
                Iterators_IsStreamOfPred<name, type>>;

/// Predicate to verify that a named argument or result's stream type is an
/// LLVM struct consisting of a single I32.
class Iterators_IsStreamOfLLVMStructOfSingleI32Pred<string name>
  : Iterators_IsStreamOf<name, Iterators_StreamOfLLVMStructOfSingleI32>;

/// Predicate to verify that a named argument or result's stream type is an
/// LLVM struct consisting of numerics.
class Iterators_IsStreamOfLLVMStructOfNumericsPred<string name>
  : Iterators_IsStreamOf<name, Iterators_StreamOfLLVMStructOfNumerics>;

/// ArrayAttr where the attribute elements are again ArrayAttrs.
def Iterators_ArrayArrayAttr
  : TypedArrayAttrBase<Builtin_ArrayAttr, "array attribute of array attributes"> {
  let storageType = [{ ::mlir::ArrayAttr }];
}

/// ArrayAttr of ArrayAttrs where, for all inner ArrayAttrs, the sequence of
/// types of the inner-most elements is the same. In other words, the type
/// of the element at position i is the same for all inner ArrayAttrs.
def Iterators_HomogeneouslyTypedArrayArrayAttrConstraint
  : AttrConstraint<
      CPred<[{
        $_self.dyn_cast<::mlir::ArrayAttr>().size() == 0 ||
        ::llvm::is_splat(
          ::llvm::map_range(
            $_self.dyn_cast<::mlir::ArrayAttr>().getValue(),
            [](::mlir::Attribute attr) {
              return ::llvm::SmallVector<Type>(
                ::llvm::map_range(
                  attr.dyn_cast<::mlir::ArrayAttr>(),
                  [](Attribute attr) { return attr.getType(); }));;
            }))
        }]>,
      "where all inner arrays have the same type sequence">;
def Iterators_HomogeneouslyTypedArrayArrayAttr
  : Confined<Iterators_ArrayArrayAttr,
            [Iterators_HomogeneouslyTypedArrayArrayAttrConstraint]>;

/// ArrayAttr of ArrayAttr where the inner ArrayAttrs have matching element
/// types and all those types are numeric.
def Iterators_HomogeneouslyTypedNumericArrayArrayAttr
  : Confined<Iterators_ArrayArrayAttr, [
        Iterators_HomogeneouslyTypedArrayArrayAttrConstraint,
        AttrConstraint<
          Or<[
            CPred<"$_self.dyn_cast<::mlir::ArrayAttr>().size() == 0">,
            SubstLeaves<"$_self", "(*($_self.dyn_cast<::mlir::ArrayAttr>().begin()))",
                        Iterators_NumericArrayAttr.predicate>]>,
          "and where the inner arrays consist of numeric values">
      ]>;

//===----------------------------------------------------------------------===//
// Element types
//===----------------------------------------------------------------------===//

def Iterators_ColumnarBatch : Iterators_Type<"ColumnarBatch", "columnar_batch"> {
  let summary = "Batch of tuples in columnar form";
  let description = [{
    A batch of tuples in columnar form, i.e., a horizontal slice of a table
    represented by a collection of columns, each holding elements of the same
    type and all with the same number of elements, where the elements at
    matching indices represent a tuple.
  }];
  let parameters = (ins Builtin_Tuple:$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";
  let extraClassDeclaration = [{
    ::llvm::ArrayRef<::mlir::Type> getTypes() {
      return this->getElementType().cast<TupleType>().getTypes();
    }
  }];
}

//===----------------------------------------------------------------------===//
// State of low-level iterators
//===----------------------------------------------------------------------===//

class Iterators_Iterator<string name, string typeMnemonic>
    : Iterators_Type<name, typeMnemonic,
                     [DeclareTypeInterfaceMethods<Iterators_IteratorInterface>]> {
  let summary = "State of a Volcano-style iterator";
}

def Iterators_SampleInputState
    : Iterators_Iterator<"SampleInputState", "sampleinputstate"> {
  let summary = "Sample iterator that produces some input";
  let description = [{
    Currently, this iterator behaves like `range(3)` in Python, i.e., produces
    the numbers 0, 1, 2. The state thus needs to remember the current index,
    which, if lower than 3, is returned and incremented by 1 in the call to
    next. In C++, the state could be represented by:

    ```C++
    struct SampleInputState { int currentIndex; };
    ```
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";
}

def Iterators_ReduceState : Iterators_Iterator<"ReduceState", "reducestate"> {
  let summary = "Sums up the input elements and returns the result once";
  let description = [{
    This iterator is a stub for more comples reductions. Currently, it only
    works on tuples with a single I32, which it consumes from its upstream
    iterator and of which it produces the sum. Since that is done in the first
    call to next (which is the only one that returns an element), this iterator
    does need a state beyond the state of its upstream iterator. In C++, this
    could be represented by:

    ```C++
    template <typename UpstreamIteratorState>
    struct ReduceState { UpstreamIteratorState upstreamState; };
    ```
  }];

  let parameters = (ins "IteratorInterface":$upstreamType);
  let assemblyFormat = "`<` qualified($upstreamType) `>`";
  let extraClassDeclaration = [{
    ::mlir::Type getElementType() {
      return this->getUpstreamType().getElementType();
    }
  }];
}

def IsIteratorTypePred
    : CPred<"$_self.isa<::mlir::iterators::IteratorInterface>()">;

def AnyIteratorState
    : Type<IsIteratorTypePred, "iteratorstate", "::mlir::Type">;

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
