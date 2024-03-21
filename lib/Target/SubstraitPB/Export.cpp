//===-- Export.cpp - Export Substrait dialect to protobuf -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Target/SubstraitPB/Export.h"
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

DECLARE_EXPORT_FUNC(ModuleOp, Plan)
DECLARE_EXPORT_FUNC(NamedTableOp, Rel)
DECLARE_EXPORT_FUNC(PlanOp, Plan)
DECLARE_EXPORT_FUNC(RelOpInterface, Rel)

FailureOr<std::unique_ptr<proto::Type>> exportType(Location loc,
                                                   mlir::Type mlirType) {
  // TODO(ingomueller): Support other types.
  auto si32 = IntegerType::get(mlirType.getContext(), 32, IntegerType::Signed);
  if (mlirType != si32)
    return emitError(loc) << "could not export unsupported type " << mlirType;

  // TODO(ingomueller): support other nullability modes.
  auto i32Type = std::make_unique<proto::Type::I32>();
  i32Type->set_nullability(
      Type_Nullability::Type_Nullability_NULLABILITY_REQUIRED);

  auto type = std::make_unique<proto::Type>();
  type->set_allocated_i32(i32Type.release());

  return std::move(type);
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

    PlanRel *planRel = plan->add_relations();
    planRel->set_allocated_rel(rel.value().release());
  }

  return std::move(plan);
}

FailureOr<std::unique_ptr<Rel>> exportOperation(RelOpInterface op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::unique_ptr<Rel>>>(op)
      .Case<NamedTableOp>([&](auto op) { return exportOperation(op); })
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
          op->emitOpError("could not be serialized to binary format");
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
