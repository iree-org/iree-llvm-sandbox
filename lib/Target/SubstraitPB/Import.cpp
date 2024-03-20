//===-- Import.cpp - Import protobuf to Substrait dialect -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Target/SubstraitPB/Import.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "structured/Dialect/Substrait/IR/Substrait.h"
#include "structured/Target/SubstraitPB/Options.h"

#include <google/protobuf/descriptor.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <substrait/proto/algebra.pb.h>
#include <substrait/proto/plan.pb.h>
#include <substrait/proto/type.pb.h>

using namespace mlir;
using namespace mlir::substrait;
using namespace ::substrait;
using namespace ::substrait::proto;

namespace pb = google::protobuf;

namespace {

// Forward declaration for the import function of the given message type.
//
// We need one such function for most message types that we want to import. The
// forward declarations are necessary such all import functions are available
// for the definitions indepedently of the order of these definitions. The
// message type passed to the function (specified by `MESSAGE_TYPE`) may be
// different than the one it is responsible for: often the target op type
// (specified by `OP_TYPE`) depends on a nested field value (such as `oneof`)
// but the import logic needs the whole context; the message that is passed in
// is the most deeply nested message that provides the whole context.
#define DECLARE_IMPORT_FUNC(MESSAGE_TYPE, ARG_TYPE, OP_TYPE)                   \
  static FailureOr<OP_TYPE> import##MESSAGE_TYPE(ImplicitLocOpBuilder builder, \
                                                 const ARG_TYPE &message);

DECLARE_IMPORT_FUNC(NamedTable, Rel, NamedTableOp)
DECLARE_IMPORT_FUNC(Plan, Plan, PlanOp)
DECLARE_IMPORT_FUNC(PlanRel, PlanRel, PlanRelOp)
DECLARE_IMPORT_FUNC(ReadRel, Rel, RelOpInterface)
DECLARE_IMPORT_FUNC(Rel, Rel, RelOpInterface)

static mlir::FailureOr<mlir::Type> importType(MLIRContext *context,
                                              const proto::Type &type) {

  proto::Type::KindCase kind_case = type.kind_case();
  switch (kind_case) {
  case proto::Type::kBool: {
    return IntegerType::get(context, 1, IntegerType::Signed);
  }
  case proto::Type::kI32: {
    return IntegerType::get(context, 32, IntegerType::Signed);
  }
  case proto::Type::kStruct: {
    const proto::Type::Struct &structType = type.struct_();
    llvm::SmallVector<mlir::Type> fieldTypes;
    fieldTypes.reserve(structType.types_size());
    for (const proto::Type &fieldType : structType.types()) {
      FailureOr<mlir::Type> mlirFieldType = importType(context, fieldType);
      if (failed(mlirFieldType))
        return failure();
      fieldTypes.push_back(mlirFieldType.value());
    }
    return TupleType::get(context, fieldTypes);
  }
    // TODO(ingomueller): Support more types.
  default: {
    auto loc = UnknownLoc::get(context);
    const pb::FieldDescriptor *desc =
        proto::Type::GetDescriptor()->FindFieldByNumber(kind_case);
    return emitError(loc) << "could not import unsupported type "
                          << desc->name();
  }
  }
}

static mlir::FailureOr<NamedTableOp>
importNamedTable(ImplicitLocOpBuilder builder, const Rel &message) {
  const ReadRel &readRel = message.read();
  const ReadRel::NamedTable &namedTable = readRel.named_table();
  MLIRContext *context = builder.getContext();

  // Assemble table name.
  llvm::SmallVector<FlatSymbolRefAttr> tableNameRefs;
  tableNameRefs.reserve(namedTable.names_size());
  for (const std::string &name : namedTable.names()) {
    auto attr = FlatSymbolRefAttr::get(context, name);
    tableNameRefs.push_back(attr);
  }
  llvm::ArrayRef<FlatSymbolRefAttr> tableNameNestedRefs =
      llvm::ArrayRef<FlatSymbolRefAttr>(tableNameRefs).drop_front();
  llvm::StringRef tableNameRootRef = tableNameRefs.front().getValue();
  auto tableName =
      SymbolRefAttr::get(context, tableNameRootRef, tableNameNestedRefs);

  // Assemble field names from schema.
  const NamedStruct &baseSchema = readRel.base_schema();
  llvm::SmallVector<Attribute> fieldNames;
  fieldNames.reserve(baseSchema.names_size());
  for (const std::string &name : baseSchema.names()) {
    auto attr = StringAttr::get(context, name);
    fieldNames.push_back(attr);
  }
  auto fieldNamesAttr = ArrayAttr::get(context, fieldNames);

  // Assemble field names from schema.
  const proto::Type::Struct &struct_ = baseSchema.struct_();
  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.reserve(struct_.types_size());
  for (const proto::Type &type : struct_.types()) {
    FailureOr<mlir::Type> mlirType = importType(context, type);
    if (failed(mlirType))
      return failure();
    resultTypes.push_back(mlirType.value());
  }
  auto resultType = TupleType::get(context, resultTypes);

  // Assemble final op.
  auto namedTableOp =
      builder.create<NamedTableOp>(resultType, tableName, fieldNamesAttr);

  return namedTableOp;
}

static FailureOr<PlanOp> importPlan(ImplicitLocOpBuilder builder,
                                    const Plan &message) {
  const Version &version = message.version();
  auto planOp = builder.create<PlanOp>(
      version.major_number(), version.minor_number(), version.patch_number(),
      version.git_hash(), version.producer());
  planOp.getBody().push_back(new Block());

  for (const auto &relation : message.relations()) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToEnd(&planOp.getBody().front());
    if (failed(importPlanRel(builder, relation)))
      return failure();
  }

  return planOp;
}

static FailureOr<PlanRelOp> importPlanRel(ImplicitLocOpBuilder builder,
                                          const PlanRel &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  PlanRel::RelTypeCase relType = message.rel_type_case();
  switch (relType) {
  case PlanRel::RelTypeCase::kRel: {
    auto planRelOp = builder.create<PlanRelOp>();
    planRelOp.getBody().push_back(new Block());
    Block *block = &planRelOp.getBody().front();

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToEnd(block);
    const Rel &rel = message.rel();
    mlir::FailureOr<Operation *> rootRel = importRel(builder, rel);
    if (failed(rootRel))
      return failure();

    builder.setInsertionPointToEnd(block);
    builder.create<YieldOp>(rootRel.value()->getResult(0));

    return planRelOp;
  }
  default: {
    const pb::FieldDescriptor *desc =
        PlanRel::GetDescriptor()->FindFieldByNumber(relType);
    return emitError(loc) << Twine("unsupported PlanRel type: ") + desc->name();
  }
  }
}

static mlir::FailureOr<RelOpInterface>
importReadRel(ImplicitLocOpBuilder builder, const Rel &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  const ReadRel &readRel = message.read();
  ReadRel::ReadTypeCase readType = readRel.read_type_case();
  switch (readType) {
  case ReadRel::ReadTypeCase::kNamedTable: {
    return importNamedTable(builder, message);
  }
  default:
    const pb::FieldDescriptor *desc =
        ReadRel::GetDescriptor()->FindFieldByNumber(readType);
    return emitError(loc) << Twine("unsupported ReadRel type: ") + desc->name();
  }
}

static mlir::FailureOr<RelOpInterface> importRel(ImplicitLocOpBuilder builder,
                                                 const Rel &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  Rel::RelTypeCase relType = message.rel_type_case();
  switch (relType) {
  case Rel::RelTypeCase::kRead: {
    return importReadRel(builder, message);
  }
  default:
    const pb::FieldDescriptor *desc =
        Rel::GetDescriptor()->FindFieldByNumber(relType);
    return emitError(loc) << Twine("unsupported Rel type: ") + desc->name();
  }
}

} // namespace

namespace mlir {
namespace substrait {

OwningOpRef<ModuleOp>
translateProtobufToSubstrait(llvm::StringRef input, MLIRContext *context,
                             ImportExportOptions options) {
  Location loc = UnknownLoc::get(context);
  auto plan = std::make_unique<Plan>();
  switch (options.serdeFormat) {
  case substrait::SerdeFormat::kText:
    if (!pb::TextFormat::ParseFromString(input.str(), plan.get())) {
      emitError(loc) << "could not parse string as 'Plan' message.";
      return {};
    }
    break;
  case substrait::SerdeFormat::kBinary:
    if (!plan->ParseFromString(input.str())) {
      emitError(loc) << "could not deserialize input as 'Plan' message.";
      return {};
    }
    break;
  case substrait::SerdeFormat::kJson:
  case substrait::SerdeFormat::kPrettyJson: {
    pb::util::Status status =
        pb::util::JsonStringToMessage(input.str(), plan.get());
    if (!status.ok()) {
      emitError(loc) << "could not deserialize JSON as 'Plan' message:\n"
                     << status.message().as_string();
      return {};
    }
  }
  }

  context->loadDialect<SubstraitDialect>();

  ImplicitLocOpBuilder builder(loc, context);
  auto module = builder.create<ModuleOp>(loc);
  auto moduleRef = OwningOpRef<ModuleOp>(module);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  if (failed(importPlan(builder, *plan)))
    return {};

  return moduleRef;
}

} // namespace substrait
} // namespace mlir
