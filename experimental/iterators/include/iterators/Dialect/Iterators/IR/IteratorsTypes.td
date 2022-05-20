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

/// An LLVMStructType where the body only uses the provided types.
class Iterators_LLVMStructOf<list<Type> allowedTypes>
  : MixedContainerType<AnyTypeOf<allowedTypes>, LLVM_AnyStruct.predicate,
                       "$_self.cast<::mlir::LLVM::LLVMStructType>().getBody()",
                       "LLVM struct">;

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

/// Prediate to verify that a named argument or result's stream type matches a
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
