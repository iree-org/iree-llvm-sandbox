//===-- Dialects.cpp - CAPI for dialects ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured-c/Dialects.h"

#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Types.h"
#include "structured/Dialect/Iterators/IR/Iterators.h"
#include "structured/Dialect/Substrait/IR/Substrait.h"
#include "structured/Dialect/Tabular/IR/Tabular.h"
#include "structured/Dialect/Tuple/IR/Tuple.h"
#include "structured/Target/SubstraitPB/Export.h"
#include "structured/Target/SubstraitPB/Import.h"
#include "structured/Target/SubstraitPB/Options.h"

using namespace mlir;
using namespace mlir::iterators;
using namespace mlir::substrait;
using namespace mlir::tabular;
using namespace mlir::tuple;

//===----------------------------------------------------------------------===//
// Iterators dialect and types
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Iterators, iterators, IteratorsDialect)

bool mlirTypeIsAIteratorsStreamType(MlirType type) {
  return unwrap(type).isa<StreamType>();
}

MlirType mlirIteratorsStreamTypeGet(MlirContext context, MlirType elementType) {
  return wrap(StreamType::get(unwrap(context), unwrap(elementType)));
}

//===----------------------------------------------------------------------===//
// Substrait dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Substrait, substrait, SubstraitDialect)

/// Converts the provided enum value into the equivalent value from
/// `::mlir::substrait::SerdeFormat`.
SerdeFormat convertSerdeFormat(MlirSubstraitSerdeFormat format) {
  switch (format) {
  case MlirSubstraitBinarySerdeFormat:
    return SerdeFormat::kBinary;
  case MlirSubstraitTextSerdeFormat:
    return SerdeFormat::kText;
  case MlirSubstraitJsonSerdeFormat:
    return SerdeFormat::kJson;
  case MlirSubstraitPrettyJsonSerdeFormat:
    return SerdeFormat::kPrettyJson;
  }
}

MlirModule mlirSubstraitImportPlan(MlirContext context, MlirStringRef input,
                                   MlirSubstraitSerdeFormat format) {
  ImportExportOptions options;
  options.serdeFormat = convertSerdeFormat(format);
  OwningOpRef<ModuleOp> owning =
      translateProtobufToSubstrait(unwrap(input), unwrap(context), options);
  if (!owning)
    return MlirModule{nullptr};
  return MlirModule{owning.release().getOperation()};
}

MlirAttribute mlirSubstraitExportPlan(MlirOperation op,
                                      MlirSubstraitSerdeFormat format) {
  std::string str;
  llvm::raw_string_ostream stream(str);
  ImportExportOptions options;
  options.serdeFormat = convertSerdeFormat(format);
  if (failed(translateSubstraitToProtobuf(unwrap(op), stream, options)))
    return wrap(Attribute());
  MLIRContext *context = unwrap(op)->getContext();
  Attribute attr = StringAttr::get(context, str);
  return wrap(attr);
}

//===----------------------------------------------------------------------===//
// Tabular dialect and types
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tabular, tabular, TabularDialect)

/// Checks whether the given type is a tabular view type.
bool mlirTypeIsATabularView(MlirType type) {
  return unwrap(type).isa<TabularViewType>();
}

/// Creates a tabular view type that consists of the given list of column types.
/// The type is owned by the context.
MlirType mlirTabularViewTypeGet(MlirContext ctx, intptr_t numColumns,
                                MlirType const *columnTypes) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typesRef = unwrapList(numColumns, columnTypes, types);
  return wrap(TabularViewType::get(unwrap(ctx), typesRef));
}

/// Returns the number of types contained in a tabular view.
intptr_t mlirTabularViewTypeGetNumColumnTypes(MlirType type) {
  return unwrap(type).cast<TabularViewType>().getColumnTypes().size();
}

/// Returns the pos-th type in the tabular view type.
MlirType mlirTabularViewTypeGetColumnType(MlirType type, intptr_t pos) {
  return wrap(unwrap(type)
                  .cast<TabularViewType>()
                  .getColumnTypes()[static_cast<size_t>(pos)]);
}

MlirType mlirTabularViewTypeGetRowType(MlirType type) {
  return wrap(unwrap(type).cast<TabularViewType>().getRowType());
}

//===----------------------------------------------------------------------===//
// Tuple dialect and attributes
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tuple, tuple, TupleDialect)
