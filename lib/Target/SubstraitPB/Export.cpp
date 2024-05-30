//===-- Export.cpp - Export Substrait dialect to protobuf -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Target/SubstraitPB/Export.h"
#include "ProtobufUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "structured/Dialect/Substrait/IR/Substrait.h"
#include "structured/Target/SubstraitPB/Options.h"
#include "llvm/ADT/TypeSwitch.h"

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

// Forward declaration for the export function of the given operation type.
//
// We need one such function for most op type that we want to export. The
// forward declarations are necessary such all export functions are available
// for the definitions indepedently of the order of these definitions. The
// `MESSAGE_TYPE` argument corresponds to the protobuf message type returned
// by the function.
#define DECLARE_EXPORT_FUNC(OP_TYPE, MESSAGE_TYPE)                             \
  static FailureOr<std::unique_ptr<MESSAGE_TYPE>> exportOperation(OP_TYPE op);

DECLARE_EXPORT_FUNC(CrossOp, Rel)
DECLARE_EXPORT_FUNC(EmitOp, Rel)
DECLARE_EXPORT_FUNC(ExpressionOpInterface, Expression)
DECLARE_EXPORT_FUNC(FieldReferenceOp, Expression)
DECLARE_EXPORT_FUNC(FilterOp, Rel)
DECLARE_EXPORT_FUNC(LiteralOp, Expression)
DECLARE_EXPORT_FUNC(ModuleOp, Plan)
DECLARE_EXPORT_FUNC(NamedTableOp, Rel)
DECLARE_EXPORT_FUNC(PlanOp, Plan)
DECLARE_EXPORT_FUNC(RelOpInterface, Rel)

FailureOr<std::unique_ptr<pb::Message>> exportOperation(Operation *op);
FailureOr<std::unique_ptr<Rel>> exportOperation(RelOpInterface op);

FailureOr<std::unique_ptr<proto::Type>> exportType(Location loc,
                                                   mlir::Type mlirType) {
  MLIRContext *context = mlirType.getContext();

  // Handle SI1.
  auto si1 = IntegerType::get(context, 1, IntegerType::Signed);
  if (mlirType == si1) {
    // TODO(ingomueller): support other nullability modes.
    auto i1Type = std::make_unique<proto::Type::Boolean>();
    i1Type->set_nullability(
        Type_Nullability::Type_Nullability_NULLABILITY_REQUIRED);

    auto type = std::make_unique<proto::Type>();
    type->set_allocated_bool_(i1Type.release());
    return std::move(type);
  }

  // Handle SI32.
  auto si32 = IntegerType::get(context, 32, IntegerType::Signed);
  if (mlirType == si32) {
    // TODO(ingomueller): support other nullability modes.
    auto i32Type = std::make_unique<proto::Type::I32>();
    i32Type->set_nullability(
        Type_Nullability::Type_Nullability_NULLABILITY_REQUIRED);

    auto type = std::make_unique<proto::Type>();
    type->set_allocated_i32(i32Type.release());
    return std::move(type);
  }

  if (auto tupleType = llvm::dyn_cast<TupleType>(mlirType)) {
    auto structType = std::make_unique<proto::Type::Struct>();
    for (mlir::Type fieldType : tupleType.getTypes()) {
      // Convert field type recursively.
      FailureOr<std::unique_ptr<proto::Type>> type = exportType(loc, fieldType);
      if (failed(type))
        return failure();
      *structType->add_types() = *type.value();
    }

    auto type = std::make_unique<proto::Type>();
    type->set_allocated_struct_(structType.release());
    return std::move(type);
  }

  // TODO(ingomueller): Support other types.
  return emitError(loc) << "could not export unsupported type " << mlirType;
}

FailureOr<std::unique_ptr<Rel>> exportOperation(CrossOp op) {
  // Build `RelCommon` message.
  auto relCommon = std::make_unique<RelCommon>();
  auto direct = std::make_unique<RelCommon::Direct>();
  relCommon->set_allocated_direct(direct.release());

  // Build `left` input message.
  auto leftOp =
      llvm::dyn_cast_if_present<RelOpInterface>(op.getLeft().getDefiningOp());
  if (!leftOp)
    return op->emitOpError(
        "left input was not produced by Substrait relation op");

  FailureOr<std::unique_ptr<Rel>> leftRel = exportOperation(leftOp);
  if (failed(leftRel))
    return failure();

  // Build `right` input message.
  auto rightOp =
      llvm::dyn_cast_if_present<RelOpInterface>(op.getRight().getDefiningOp());
  if (!rightOp)
    return op->emitOpError(
        "right input was not produced by Substrait relation op");

  FailureOr<std::unique_ptr<Rel>> rightRel = exportOperation(rightOp);
  if (failed(rightRel))
    return failure();

  // Build `CrossRel` message.
  auto crossRel = std::make_unique<CrossRel>();
  crossRel->set_allocated_common(relCommon.release());
  crossRel->set_allocated_left(leftRel->release());
  crossRel->set_allocated_right(rightRel->release());

  // Build `Rel` message.
  auto rel = std::make_unique<Rel>();
  rel->set_allocated_cross(crossRel.release());

  return rel;
}

FailureOr<std::unique_ptr<Rel>> exportOperation(EmitOp op) {
  auto inputOp =
      dyn_cast_if_present<RelOpInterface>(op.getInput().getDefiningOp());
  if (!inputOp)
    return op->emitOpError(
        "has input that was not produced by Substrait relation op");

  // Export input op.
  FailureOr<std::unique_ptr<Rel>> inputRel = exportOperation(inputOp);
  if (failed(inputRel))
    return failure();

  // Build the `emit` message.
  auto emit = std::make_unique<RelCommon::Emit>();
  for (auto intAttr : op.getMapping().getAsRange<IntegerAttr>())
    emit->add_output_mapping(intAttr.getInt());

  // Attach the `emit` message to the `RelCommon` message.
  FailureOr<RelCommon *> relCommon =
      protobuf_utils::getMutableCommon(inputRel->get(), op.getLoc());
  if (failed(relCommon))
    return failure();

  if (relCommon.value()->has_emit()) {
    InFlightDiagnostic diag =
        op->emitOpError("has 'input' that already has 'emit' message "
                        "(try running canonicalization?)");
    diag.attachNote(inputOp.getLoc()) << "op exported to 'input' message";
    return diag;
  }

  relCommon.value()->set_allocated_emit(emit.release());

  return inputRel;
}

FailureOr<std::unique_ptr<Expression>>
exportOperation(ExpressionOpInterface op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::unique_ptr<Expression>>>(
             op)
      .Case<FieldReferenceOp, LiteralOp>(
          [&](auto op) { return exportOperation(op); })
      .Default(
          [](auto op) { return op->emitOpError("not supported for export"); });
}

FailureOr<std::unique_ptr<Expression>> exportOperation(FieldReferenceOp op) {
  using FieldReference = Expression::FieldReference;
  using ReferenceSegment = Expression::ReferenceSegment;

  // Build linked list of `ReferenceSegment` messages.
  // TODO: support masked references.
  std::unique_ptr<Expression::ReferenceSegment> referenceRoot;
  for (int64_t pos : llvm::reverse(op.getPosition())) {
    // Remember child segment and create new `ReferenceSegment` message.
    auto childReference = std::move(referenceRoot);
    referenceRoot = std::make_unique<ReferenceSegment>();

    // Create `StructField` message.
    // TODO(ingomueller): support other segment types.
    auto structField = std::make_unique<ReferenceSegment::StructField>();
    structField->set_field(pos);
    structField->set_allocated_child(childReference.release());

    referenceRoot->set_allocated_struct_field(structField.release());
  }

  // Build `FieldReference` message.
  auto fieldReference = std::make_unique<FieldReference>();
  fieldReference->set_allocated_direct_reference(referenceRoot.release());

  // Handle different `root_type`s.
  Value inputVal = op.getContainer();
  if (Operation *definingOp = inputVal.getDefiningOp()) {
    // If there is a defining op, the `root_type` is an `Expression`.
    ExpressionOpInterface exprOp =
        llvm::dyn_cast<ExpressionOpInterface>(definingOp);
    if (!exprOp)
      return op->emitOpError("has 'container' operand that was not produced by "
                             "Substrait expression");

    FailureOr<std::unique_ptr<Expression>> expression = exportOperation(exprOp);
    fieldReference->set_allocated_expression(expression->release());
  } else {
    // Input must be a `BlockArgument`. Only support root references for now.
    auto blockArg = llvm::cast<BlockArgument>(inputVal);
    if (blockArg.getOwner() != op->getBlock())
      // TODO(ingomueller): support outer reference type.
      return op.emitOpError("has unsupported outer reference");

    auto rootReference = std::make_unique<FieldReference::RootReference>();
    fieldReference->set_allocated_root_reference(rootReference.release());
  }

  // Build `Expression` message.
  auto expression = std::make_unique<Expression>();
  expression->set_allocated_selection(fieldReference.release());

  return expression;
}

FailureOr<std::unique_ptr<Rel>> exportOperation(FilterOp op) {
  // Build `RelCommon` message.
  auto relCommon = std::make_unique<RelCommon>();
  auto direct = std::make_unique<RelCommon::Direct>();
  relCommon->set_allocated_direct(direct.release());

  // Build input `Rel` message.
  auto inputOp =
      llvm::dyn_cast_if_present<RelOpInterface>(op.getInput().getDefiningOp());
  if (!inputOp)
    return op->emitOpError("input was not produced by Substrait relation op");

  FailureOr<std::unique_ptr<Rel>> inputRel = exportOperation(inputOp);
  if (failed(inputRel))
    return failure();

  // Build condition `Expression` message.
  auto yieldOp = llvm::cast<YieldOp>(op.getCondition().front().getTerminator());
  // TODO(ingomueller): There can be cases where there isn't a defining op but
  //                    the region argument is returned directly. Support that.
  auto conditionOp = llvm::dyn_cast_if_present<ExpressionOpInterface>(
      yieldOp.getValue().getDefiningOp());
  if (!conditionOp)
    return op->emitOpError("condition not supported for export: yielded op was "
                           "not produced by Substrait expression op");
  FailureOr<std::unique_ptr<Expression>> condition =
      exportOperation(conditionOp);
  if (failed(condition))
    return failure();

  // Build `FilterRel` message.
  auto filterRel = std::make_unique<FilterRel>();
  filterRel->set_allocated_common(relCommon.release());
  filterRel->set_allocated_input(inputRel->release());
  filterRel->set_allocated_condition(condition->release());

  // Build `Rel` message.
  auto rel = std::make_unique<Rel>();
  rel->set_allocated_filter(filterRel.release());

  return rel;
}

FailureOr<std::unique_ptr<Expression>> exportOperation(LiteralOp op) {
  // Build `Literal` message depending on type.
  auto value = llvm::cast<TypedAttr>(op.getValue());
  mlir::Type literalType = value.getType();
  auto literal = std::make_unique<Expression::Literal>();

  // `IntegerType`s.
  if (auto intType = dyn_cast<IntegerType>(literalType)) {
    if (!intType.isSigned())
      op->emitOpError("has integer value with unsupported signedness");
    switch (intType.getWidth()) {
    case 1:
      literal->set_boolean(value.cast<IntegerAttr>().getSInt());
      break;
    case 32:
      // TODO(ingomueller): Add tests when we can express plans that use i32.
      literal->set_i32(value.cast<IntegerAttr>().getSInt());
      break;
    default:
      op->emitOpError("has integer value with unsupported width");
    }
  } else
    op->emitOpError("has unsupported value");

  // Build `Expression` message.
  auto expression = std::make_unique<Expression>();
  expression->set_allocated_literal(literal.release());

  return expression;
}

FailureOr<std::unique_ptr<Plan>> exportOperation(ModuleOp op) {
  if (!op->getAttrs().empty()) {
    op->emitOpError("has attributes");
    return failure();
  }

  Region &body = op.getBodyRegion();
  if (llvm::range_size(body.getOps()) != 1) {
    op->emitOpError("has more than one op in its body");
    return failure();
  }

  if (auto plan = llvm::dyn_cast<PlanOp>(&*body.op_begin()))
    return exportOperation(plan);

  op->emitOpError("contains an op that is not a 'substrait.plan'");
  return failure();
}

FailureOr<std::unique_ptr<Rel>> exportOperation(NamedTableOp op) {
  Location loc = op.getLoc();

  // Build `NamedTable` message.
  auto namedTable = std::make_unique<ReadRel::NamedTable>();
  namedTable->add_names(op.getTableName().getRootReference().str());
  for (SymbolRefAttr attr : op.getTableName().getNestedReferences()) {
    namedTable->add_names(attr.getLeafReference().str());
  }

  // Build `RelCommon` message.
  auto relCommon = std::make_unique<RelCommon>();
  auto direct = std::make_unique<RelCommon::Direct>();
  relCommon->set_allocated_direct(direct.release());

  // Build `Struct` message.
  auto struct_ = std::make_unique<proto::Type::Struct>();
  struct_->set_nullability(
      Type_Nullability::Type_Nullability_NULLABILITY_REQUIRED);
  auto tupleType = llvm::cast<TupleType>(op.getResult().getType());
  for (mlir::Type fieldType : tupleType.getTypes()) {
    FailureOr<std::unique_ptr<proto::Type>> type = exportType(loc, fieldType);
    if (failed(type))
      return (failure());
    *struct_->add_types() = *std::move(type.value());
  }

  // Build `NamedStruct` message.
  auto namedStruct = std::make_unique<NamedStruct>();
  namedStruct->set_allocated_struct_(struct_.release());
  for (Attribute attr : op.getFieldNames()) {
    namedStruct->add_names(attr.cast<StringAttr>().getValue().str());
  }

  // Build `ReadRel` message.
  auto readRel = std::make_unique<ReadRel>();
  readRel->set_allocated_common(relCommon.release());
  readRel->set_allocated_base_schema(namedStruct.release());
  readRel->set_allocated_named_table(namedTable.release());

  // Build `Rel` message.
  auto rel = std::make_unique<Rel>();
  rel->set_allocated_read(readRel.release());

  return rel;
}

FailureOr<std::unique_ptr<Plan>> exportOperation(PlanOp op) {
  // Build `Version` message.
  auto version = std::make_unique<Version>();
  version->set_major_number(op.getMajorNumber());
  version->set_minor_number(op.getMinorNumber());
  version->set_patch_number(op.getPatchNumber());
  version->set_producer(op.getProducer().str());
  version->set_git_hash(op.getGitHash().str());

  // Build `Plan` message.
  auto plan = std::make_unique<Plan>();
  plan->set_allocated_version(version.release());

  // Add `relation`s to plan.
  for (auto relOp : op.getOps<PlanRelOp>()) {
    Operation *terminator = relOp.getBody().front().getTerminator();
    auto rootOp =
        llvm::cast<RelOpInterface>(terminator->getOperand(0).getDefiningOp());

    FailureOr<std::unique_ptr<Rel>> rel = exportOperation(rootOp);
    if (failed(rel))
      return failure();

    // Handle `Rel`/`RelRoot` cases depending on whether `names` is set.
    PlanRel *planRel = plan->add_relations();
    if (std::optional<Attribute> names = relOp.getFieldNames()) {
      auto root = std::make_unique<RelRoot>();
      root->set_allocated_input(rel->release());

      auto namesArray = cast<ArrayAttr>(names.value()).getAsRange<StringAttr>();
      for (StringAttr name : namesArray) {
        root->add_names(name.getValue().str());
      }

      planRel->set_allocated_root(root.release());
    } else {
      planRel->set_allocated_rel(rel->release());
    }
  }

  return std::move(plan);
}

FailureOr<std::unique_ptr<Rel>> exportOperation(RelOpInterface op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::unique_ptr<Rel>>>(op)
      .Case<CrossOp, EmitOp, FieldReferenceOp, FilterOp, NamedTableOp>(
          [&](auto op) { return exportOperation(op); })
      .Default([](auto op) {
        op->emitOpError("not supported for export");
        return failure();
      });
}

FailureOr<std::unique_ptr<pb::Message>> exportOperation(Operation *op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::unique_ptr<pb::Message>>>(
             op)
      .Case<ModuleOp, PlanOp>(
          [&](auto op) -> FailureOr<std::unique_ptr<pb::Message>> {
            auto typedMessage = exportOperation(op);
            if (failed(typedMessage))
              return failure();
            return std::unique_ptr<pb::Message>(typedMessage.value().release());
          })
      .Default([](auto op) {
        op->emitOpError("not supported for export");
        return failure();
      });
}

} // namespace

namespace mlir {
namespace substrait {

LogicalResult
translateSubstraitToProtobuf(Operation *op, llvm::raw_ostream &output,
                             substrait::ImportExportOptions options) {
  FailureOr<std::unique_ptr<pb::Message>> result = exportOperation(op);
  if (failed(result))
    return failure();

  std::string out;
  switch (options.serdeFormat) {
  case substrait::SerdeFormat::kText:
    if (!pb::TextFormat::PrintToString(*result.value(), &out)) {
      op->emitOpError("could not be serialized to text format");
      return failure();
    }
    break;
  case substrait::SerdeFormat::kBinary:
    if (!result->get()->SerializeToString(&out)) {
      op->emitOpError("could not be serialized to binary format");
      return failure();
    }
    break;
  case substrait::SerdeFormat::kJson:
  case substrait::SerdeFormat::kPrettyJson: {
    pb::util::JsonOptions jsonOptions;
    if (options.serdeFormat == SerdeFormat::kPrettyJson)
      jsonOptions.add_whitespace = true;
    pb::util::Status status =
        pb::util::MessageToJsonString(*result.value(), &out, jsonOptions);
    if (!status.ok()) {
      InFlightDiagnostic diag =
          op->emitOpError("could not be serialized to JSON format");
      diag.attachNote() << status.message();
      return diag;
    }
  }
  }

  output << out;
  return success();
}

} // namespace substrait
} // namespace mlir
