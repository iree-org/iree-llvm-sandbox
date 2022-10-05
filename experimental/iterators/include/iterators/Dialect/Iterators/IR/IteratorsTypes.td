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
include "mlir/IR/OpBase.td"

// Base class for Iterators dialect types.
class Iterators_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Iterators_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Element types (for the element of streams).
//
// TODO(ingomueller): Decide on the definite place for these types.
//     The type of the elements of a stream are, in principle, orthogonal to
//     streams and iterators (see also the
//     [README](/experimental/iterators/README.md#basic-concepts)), so one
//     might argue that these definitions do not belong here. However, some
//     iterator ops have constraints on what element types they support (mainly
//     due to the restricted lowerings that exist for now). While these live in
//     this dialect, it is probably most convenient to leave the types here as
//     well.
//===----------------------------------------------------------------------===//

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
// Streams (i.e., the data types passed between iterators).
//===----------------------------------------------------------------------===//

def Iterators_Stream : Iterators_Type<"Stream", "stream"> {
  let summary = "Stream of elements of the given type";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";
  let description = [{
    A collection of elements of a particular type that (1) is ordered and (2)
    can only be iterated over in that order one element at the time.

    Stream is the main data type that iterator ops consume and produce.

    See also the [README](/experimental/iterators/README.md#basic-concepts).
  }];
}

/// An Iterators stream of elements of the given type.
class Iterators_StreamOf<Type elementType>
  : Type<And<[Iterators_Stream.predicate,
              SubstLeaves<"$_self",
                          "$_self.dyn_cast<StreamType>().getElementType()",
                          elementType.predicate>]>,
         "stream with elements of type " # elementType.summary>;

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
/// LLVM struct consisting of numerics.
class Iterators_IsStreamOfLLVMStructOfNumericsPred<string name>
  : Iterators_IsStreamOf<name, Iterators_StreamOfLLVMStructOfNumerics>;

//===----------------------------------------------------------------------===//
// Types related to Iterator bodies.
//===----------------------------------------------------------------------===//

def Iterators_State : Iterators_Type<"State", "state"> {
  let summary = "State of an iterator used by its body";
  let parameters =  (ins ArrayRefParameter<"Type", "list of types">:$fieldTypes);
  let assemblyFormat = "`<` qualified($fieldTypes) `>`";
  let description = [{
    An iterator state is a collection of values identified by ordinal numbers,
    i.e., an (unnamed but typed) tuple. The values are referred to as "fields";
    their types are referred to as "field types". An iterator state is used by
    iterator bodies, i.e., by the open, next, and close functions that implement
    the logic that iterator ops get lowered to, and holds the state that is
    required during the iteration (which gets passed around different calls to
    open, next, and close).

    This is similar to (anonymous) `llvm.struct` but allows for storing values
    of arbitrary types.

    Example:

    ```
    %undef_state = iterators.undefstate : !iterators.state<i32, tensor<?xi32>>
    ```
  }];
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
