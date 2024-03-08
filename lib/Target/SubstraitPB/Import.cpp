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

#include <google/protobuf/descriptor.h>
#include <google/protobuf/text_format.h>
#include <substrait/plan.pb.h>

using namespace mlir;
using namespace mlir::substrait;
using namespace ::substrait;

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

DECLARE_IMPORT_FUNC(Plan, Plan, PlanOp)
DECLARE_IMPORT_FUNC(PlanRel, PlanRel, PlanRelOp)

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

  PlanRel::RelTypeCase rel_type = message.rel_type_case();
  switch (rel_type) {
  case PlanRel::RelTypeCase::kRel: {
    auto planRelOp = builder.create<PlanRelOp>();
    // TODO(ingomueller): import content once defined.
    return planRelOp;
  }
  default: {
    const pb::EnumDescriptor *desc =
        PlanRel::GetDescriptor()->enum_type(rel_type);
    emitError(loc) << Twine("unsupported PlanRel type: ") + desc->name() + ".";
    return {};
  }
  }
}

} // namespace

namespace mlir {
namespace substrait {

OwningOpRef<ModuleOp> translateProtobufToSubstrait(llvm::StringRef input,
                                                   MLIRContext *context) {
  Location loc = UnknownLoc::get(context);

  auto plan = std::make_unique<Plan>();
  if (!pb::TextFormat::ParseFromString(input.str(), plan.get())) {
    emitError(loc) << "could not parse string as 'Plan' message.";
    return {};
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
