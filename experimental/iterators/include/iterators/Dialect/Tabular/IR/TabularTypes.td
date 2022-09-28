//===-- TabularTypes.td - Tabular dialect types ------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TABULAR_DIALECT_TABULAR_IR_TABULARTYPES
#define TABULAR_DIALECT_TABULAR_IR_TABULARTYPES

include "iterators/Dialect/Tabular/IR/TabularDialect.td"
include "iterators/Dialect/Tabular/IR/TabularInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

// Base class for Tabular dialect types.
class Tabular_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tabular_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Tabular_TabularView : Tabular_Type<"TabularView", "tabular_view"> {
  let summary = "Tabular view into externally-managed buffers";
  let description = [{
    A tabular view exposes the data of existing buffers in tabular form, i.e.,
    as a two-dimensional grid where the values in one row may have mixed types
    but the values in each column all have the same type. This is structurally
    similar to `memref<?x tuple<T1, ..., Tn>>`. The types of the values in each
    row, i.e., `T1`, ..., `Tn` are called "column types"; the type of one row,
    i.e., tuple<T1, ..., Tn>, is called "row type".

    The vertical dimension, i.e., the number of rows is (currently) always
    dynamic; the horizontal dimension, i.e., the number of columns, is always
    known statically and equal to the number of column types. An empty list of
    column types is currently not supported.

    The view does not prescribe the physical data layout of the data. One
    strawman format for LLVM-compatible column types would be an "array of
    structs" (or `llvm.ptr<llvm.struct<(T1, ..., Tn)>>`), i.e., a single buffer
    where each row is represented as a struct; another one would be a "struct of
    arrays" (or `llvm.struct<(llvm.ptr<T1>, ..., llvm.ptr<Tn>)>`), i.e., one
    buffer for each column.
  }];
  let parameters = (ins ArrayRefParameter<"Type", "list of types">:$columnTypes);
  let assemblyFormat = "`<` $columnTypes `>`";
  let extraClassDeclaration = [{
    /// Return the number of column types.
    size_t getNumColumnTypes() const {
      return getColumnTypes().size();
    }

    /// Return the column type at index 'index'.
    Type getColumnType(size_t index) const {
      assert(index < getNumColumnTypes() && "invalid index for column type");
      return getColumnTypes()[index];
    }

    /// Return the `TupleType` that represents one row.
    TupleType getRowType() const {
      return TupleType::get(getContext(), getColumnTypes());
    }
  }];
}

#endif // TABULAR_DIALECT_TABULAR_IR_TABULARTYPES
