//===-- Import.cpp - Import protobuf to Substrait dialect -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Target/SubstraitPB/Import.h"

#include "ProtobufUtils.h"
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

DECLARE_IMPORT_FUNC(CrossRel, Rel, CrossOp)
DECLARE_IMPORT_FUNC(FilterRel, Rel, FilterOp)
DECLARE_IMPORT_FUNC(Expression, Expression, ExpressionOpInterface)
DECLARE_IMPORT_FUNC(FieldReference, Expression::FieldReference,
                    FieldReferenceOp)
DECLARE_IMPORT_FUNC(Literal, Expression::Literal, LiteralOp)
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

static mlir::FailureOr<CrossOp> importCrossRel(ImplicitLocOpBuilder builder,
                                               const Rel &message) {
  const CrossRel &crossRel = message.cross();

  // Import left and right inputs.
  const Rel &leftRel = crossRel.left();
  const Rel &rightRel = crossRel.right();

  mlir::FailureOr<RelOpInterface> leftOp = importRel(builder, leftRel);
  mlir::FailureOr<RelOpInterface> rightOp = importRel(builder, rightRel);

  if (failed(leftOp) || failed(rightOp))
    return failure();

  // Build `CrossOp`.
  Value leftVal = leftOp.value()->getResult(0);
  Value rightVal = rightOp.value()->getResult(0);

  return builder.create<CrossOp>(leftVal, rightVal);
}

static mlir::FailureOr<ExpressionOpInterface>
importExpression(ImplicitLocOpBuilder builder, const Expression &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  Expression::RexTypeCase rex_type = message.rex_type_case();
  switch (rex_type) {
  case Expression::RexTypeCase::kLiteral: {
    return importLiteral(builder, message.literal());
  }
  case Expression::RexTypeCase::kSelection: {
    return importFieldReference(builder, message.selection());
  }
  default: {
    const pb::FieldDescriptor *desc =
        Expression::GetDescriptor()->FindFieldByNumber(rex_type);
    return emitError(loc) << Twine("unsupported Expression type: ") +
                                 desc->name();
  }
  }
}

static mlir::FailureOr<FieldReferenceOp>
importFieldReference(ImplicitLocOpBuilder builder,
                     const Expression::FieldReference &message) {
  using ReferenceSegment = Expression::ReferenceSegment;

  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  // Emit error on unsupported cases.
  // TODO(ingomueller): support more cases.
  if (!message.has_direct_reference())
    return emitError(loc) << "only direct reference supported";

  // Traverse list to extract indices.
  llvm::SmallVector<int64_t> indices;
  const ReferenceSegment *currentSegment = &message.direct_reference();
  while (true) {
    if (!currentSegment->has_struct_field())
      return emitError(loc) << "only struct fields supported";

    const ReferenceSegment::StructField &structField =
        currentSegment->struct_field();
    indices.push_back(structField.field());

    // Continue in linked list or end traversal.
    if (!structField.has_child())
      break;
    currentSegment = &structField.child();
  }

  // Build `position` attribute of indices.
  ArrayAttr position = builder.getI64ArrayAttr(indices);

  // Get input value.
  Value container;
  if (message.has_root_reference()) {
    // For the `root_reference` case, that's the current block argument.
    mlir::Block::BlockArgListType blockArgs =
        builder.getInsertionBlock()->getArguments();
    assert(blockArgs.size() == 1 && "expected a single block argument");
    container = blockArgs.front();
  } else if (message.has_expression()) {
    // For the `expression`case, recursively import the expression.
    FailureOr<ExpressionOpInterface> maybeContainer =
        importExpression(builder, message.expression());
    if (failed(maybeContainer))
      return failure();
    container = maybeContainer.value()->getResult(0);
  } else {
    // For the `outer_reference` case, we need to refer to an argument of some
    // outer-level block.
    // TODO(ingomueller): support outer references.
    assert(message.has_outer_reference() && "unexpected 'root_type` case");
    return emitError(loc) << "outer references not supported";
  }

  // Build and return the op.
  return builder.create<FieldReferenceOp>(container, position);
}

static mlir::FailureOr<LiteralOp>
importLiteral(ImplicitLocOpBuilder builder,
              const Expression::Literal &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  Expression::Literal::LiteralTypeCase literalType =
      message.literal_type_case();
  switch (literalType) {
  case Expression::Literal::LiteralTypeCase::kBoolean: {
    auto attr = IntegerAttr::get(
        IntegerType::get(context, 1, IntegerType::Signed), message.boolean());
    return builder.create<LiteralOp>(attr);
  }
  default: {
    const pb::FieldDescriptor *desc =
        Expression::Literal::GetDescriptor()->FindFieldByNumber(literalType);
    return emitError(loc) << Twine("unsupported Literal type: ") + desc->name();
  }
  }
}

static mlir::FailureOr<FilterOp> importFilterRel(ImplicitLocOpBuilder builder,
                                                 const Rel &message) {
  const FilterRel &filterRel = message.filter();

  // Import input op.
  const Rel &inputRel = filterRel.input();
  mlir::FailureOr<RelOpInterface> inputOp = importRel(builder, inputRel);
  if (failed(inputOp))
    return failure();

  // Create filter op.
  auto filterOp = builder.create<FilterOp>(inputOp.value()->getResult(0));
  filterOp.getCondition().push_back(new Block);
  Block &conditionBlock = filterOp.getCondition().front();
  conditionBlock.addArgument(filterOp.getResult().getType(),
                             filterOp->getLoc());

  // Create condition region.
  const Expression &expression = filterRel.condition();
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&conditionBlock);

    FailureOr<ExpressionOpInterface> conditionOp =
        importExpression(builder, expression);
    if (failed(conditionOp))
      return failure();

    builder.create<YieldOp>(conditionOp.value()->getResult(0));
  }

  return filterOp;
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

  if (!message.has_rel() && !message.has_root()) {
    PlanRel::RelTypeCase relType = message.rel_type_case();
    const pb::FieldDescriptor *desc =
        PlanRel::GetDescriptor()->FindFieldByNumber(relType);
    return emitError(loc) << Twine("unsupported PlanRel type: ") + desc->name();
  }

  // Create new `PlanRelOp`.
  auto planRelOp = builder.create<PlanRelOp>();
  planRelOp.getBody().push_back(new Block());
  Block *block = &planRelOp.getBody().front();

  // Handle `Rel` and `RelRoot` separately.
  const Rel *rel;
  if (message.has_rel())
    rel = &message.rel();
  else {
    const RelRoot &root = message.root();
    rel = &root.input();

    // Extract names.
    SmallVector<std::string> names(root.names().begin(), root.names().end());
    SmallVector<llvm::StringRef> nameAttrs(names.begin(), names.end());
    ArrayAttr namesAttr = builder.getStrArrayAttr(nameAttrs);
    planRelOp.setFieldNamesAttr(namesAttr);
  }

  // Import body of `PlanRelOp`.
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToEnd(block);
  mlir::FailureOr<Operation *> rootRel = importRel(builder, *rel);
  if (failed(rootRel))
    return failure();

  builder.setInsertionPointToEnd(block);
  builder.create<YieldOp>(rootRel.value()->getResult(0));

  return planRelOp;
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

  // Import rel depending on its type.
  Rel::RelTypeCase relType = message.rel_type_case();
  FailureOr<RelOpInterface> maybeOp;
  switch (relType) {
  case Rel::RelTypeCase::kCross:
    maybeOp = importCrossRel(builder, message);
    break;
  case Rel::RelTypeCase::kFilter:
    maybeOp = importFilterRel(builder, message);
    break;
  case Rel::RelTypeCase::kRead:
    maybeOp = importReadRel(builder, message);
    break;
  default:
    const pb::FieldDescriptor *desc =
        Rel::GetDescriptor()->FindFieldByNumber(relType);
    return emitError(loc) << Twine("unsupported Rel type: ") + desc->name();
  }
  if (failed(maybeOp))
    return failure();
  RelOpInterface op = maybeOp.value();

  // Remainder: Import `emit` op if needed.

  // Extract `RelCommon` message.
  FailureOr<const RelCommon *> maybeRelCommon =
      protobuf_utils::getCommon(message, loc);
  if (failed(maybeRelCommon))
    return failure();
  const RelCommon *relCommon = maybeRelCommon.value();

  // For the `direct` case, no further op needs to be created.
  if (relCommon->has_direct())
    return op;
  assert(relCommon->has_emit() && "expected either 'direct' or 'emit'");

  // For the `emit` case, we need to insert an `EmitOp`.
  const proto::RelCommon::Emit &emit = relCommon->emit();
  SmallVector<int64_t> mapping;
  append_range(mapping, emit.output_mapping());
  ArrayAttr mappingAttr = builder.getI64ArrayAttr(mapping);
  auto emitOp = builder.create<EmitOp>(op->getResult(0), mappingAttr);

  return {emitOp};
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
