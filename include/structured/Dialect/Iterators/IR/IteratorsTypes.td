//===-- IteratorsTypes.td - Iterators dialect types --------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES

include "structured/Dialect/Iterators/IR/IteratorsDialect.td"
include "structured/Dialect/Iterators/IR/IteratorsInterfaces.td"
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

/// List of printable float types.
def Iterators_PrintableFloatType :
  FloatOfWidths<[16, 32, 64]>;

/// Printable integer type.
def Iterators_PrintableIntegerType :
  ConfinedType<AnySignlessInteger,
    [CPred<"$_self.cast<IntegerType>().getWidth() <= 64">],
    "signless integer with at most 64 bits">;

/// Printable complex type.
def Iterators_PrintableComplexType : Type<Or<[
  Complex<Iterators_PrintableFloatType>.predicate,
  Complex<Iterators_PrintableIntegerType>.predicate,
  ]>>;

/// Any printable numeric type (i.e., any printable complex, float, or integer
/// type).
def Iterators_PrintableNumericType : AnyTypeOf<[
    Iterators_PrintableComplexType,
    Iterators_PrintableFloatType,
    Iterators_PrintableIntegerType,
    Index
  ]>;

/// Attribute of a printable numeric type.
def Iterators_PrintableNumericAttr
  : Attr<SubstLeaves<"$_self",
                     "$_self.isa<TypedAttr>() && $_self.cast<TypedAttr>().getType()",
                     Iterators_PrintableNumericType.predicate>,
        "any printable numeric type">;

/// Array of printable numeric attributes.
def Iterators_PrintableNumericArrayAttr
  : TypedArrayAttrBase<Iterators_PrintableNumericAttr,
                       "array of printable numeric types">;

/// List of LLVM-compatible numeric types.
def Iterators_NumericLLVMTypes {
  list<Type> types = [I1, I8, I16, I32, I64,
                      F16, F32, F64];
}

/// Any LLVM-compatible numeric type.
def Iterators_AnyLLVMNumeric
  : AnyTypeOf<Iterators_NumericLLVMTypes.types>;

/// Attribute of an LLVM-compatible numeric type.
def Iterators_LLVMNumericAttr
  : Attr<SubstLeaves<"$_self",
                     "$_self.isa<TypedAttr>() && $_self.cast<TypedAttr>().getType()",
                     Iterators_AnyLLVMNumeric.predicate>,
        "any LLVM-compatible numeric type">;

/// Array of LLVM-compatible numeric attributes.
def Iterators_LLVMNumericArrayAttr
  : TypedArrayAttrBase<Iterators_LLVMNumericAttr,
                       "array of LLVM-compatible numeric types">;

/// An LLVMStructType where the body only uses the provided types.
class Iterators_LLVMStructOf<list<Type> allowedTypes>
  : MixedContainerType<AnyTypeOf<allowedTypes>, LLVM_AnyStruct.predicate,
                       "$_self.cast<::mlir::LLVM::LLVMStructType>().getBody()",
                       "LLVM struct">;

/// An LLVMStructType where the body only uses numeric types.
def Iterators_LLVMStructOfNumerics
  : Iterators_LLVMStructOf<Iterators_NumericLLVMTypes.types>;

/// An LLVMStructType where the body only uses the given type or structs with
/// the same restriction (i.e., an LLVMStructType whose flattened element types
/// match the provided type).
class Iterators_NestedLLVMStructOf<Type type>
  : Type<
      And<[
        LLVM_AnyStruct.predicate,
        Concat<
          "[&](){"
          "  std::function<void(::mlir::Type, ::llvm::SmallVector<Type>&)>"
          "    getFlattenedTypes;"
          "  getFlattenedTypes ="
          "    [&getFlattenedTypes](::mlir::Type type,"
          "                         ::llvm::SmallVector<Type>& result) -> void {"
          "      if (auto structType = type.dyn_cast<::mlir::LLVM::LLVMStructType>()) {"
          "        for (Type t : structType.getBody())"
          "          getFlattenedTypes(t, result);"
          "      } else"
          "        result.push_back(type);"
          "    };"
          "  ::llvm::SmallVector<Type> nestedTypes;"
          "  getFlattenedTypes($_self, nestedTypes);"
          "  return ::llvm::all_of(nestedTypes, [](Type t) { "
          "    return t && (",
          SubstLeaves<"$_self", "t", type.predicate>,
          "        ); });"
          "}()"
        >
      ]>,
      "LLVM struct with any combination of " # type.summary # " values">;

/// An LLVMStructType whose nested element types are all numerics.
def Iterators_NestedLLVMStructOfNumerics
  : Iterators_NestedLLVMStructOf<Iterators_AnyLLVMNumeric>;

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
        ::llvm::all_equal(
          ::llvm::map_range(
            $_self.dyn_cast<::mlir::ArrayAttr>().getValue(),
            [](::mlir::Attribute attr) {
              return ::llvm::SmallVector<Type>(
                ::llvm::map_range(
                  attr.dyn_cast<::mlir::ArrayAttr>(),
                  [](Attribute attr) { return attr.cast<TypedAttr>().getType(); }));;
            }))
        }]>,
      "where all inner arrays have the same type sequence">;
def Iterators_HomogeneouslyTypedArrayArrayAttr
  : ConfinedAttr<Iterators_ArrayArrayAttr,
                 [Iterators_HomogeneouslyTypedArrayArrayAttrConstraint]>;

/// ArrayAttr of ArrayAttr where the inner ArrayAttrs have matching element
/// types and all those types are LLVM-compatible numerics.
def Iterators_HomogeneouslyTypedLLVMNumericArrayArrayAttr
  : ConfinedAttr<Iterators_ArrayArrayAttr, [
        Iterators_HomogeneouslyTypedArrayArrayAttrConstraint,
        AttrConstraint<
          Or<[
            CPred<"$_self.dyn_cast<::mlir::ArrayAttr>().size() == 0">,
            SubstLeaves<"$_self", "(*($_self.dyn_cast<::mlir::ArrayAttr>().begin()))",
                        Iterators_LLVMNumericArrayAttr.predicate>]>,
          "and where the inner arrays consist of numeric values">
      ]>;

/// Printable type that can occur as a (nested) element type of printable tuples.
def Iterators_PrintableElementType : AnyTypeOf<[
    Iterators_PrintableNumericType,
    Iterators_NestedLLVMStructOfNumerics
  ]>;

/// A tuple consisting only of printable types.
def Iterators_TupleOfPrintableTypes : TupleOf<[Iterators_PrintableElementType]>;

/// A potentially nested tuple consisting only of printable types.
def Iterators_NestedTupleOfPrintableTypes
  : NestedTupleOf<[Iterators_PrintableElementType]>;

/// Any printable type.
def Iterators_PrintableType : AnyTypeOf<[
    Iterators_PrintableElementType,
    Iterators_NestedTupleOfPrintableTypes
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

/// An Iterators stream of tuples of printable types.
def Iterators_StreamOfPrintableTuples
  : Iterators_StreamOf<Iterators_TupleOfPrintableTypes>;

/// An Iterators stream of printable elements.
def Iterators_StreamOfPrintableElements
  : Iterators_StreamOf<Iterators_PrintableType>;

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
    %i32 = ...
    %tensor = ...
    %state = iterators.createstate(%i32, %tensor) :
                 !iterators.state<i32, tensor<?xi32>>
    ```
  }];
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
