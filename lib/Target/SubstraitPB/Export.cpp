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
#include "llvm/ADT/TypeSwitch.h"

#include <google/protobuf/text_format.h>
#include <substrait/plan.pb.h>

using namespace mlir;
using namespace mlir::substrait;

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
  static FailureOr<std::unique_ptr<::substrait::MESSAGE_TYPE>>                 \
  exportOperation(OP_TYPE op);

DECLARE_EXPORT_FUNC(ModuleOp, Plan)
DECLARE_EXPORT_FUNC(PlanOp, Plan)

FailureOr<std::unique_ptr<::substrait::Plan>> exportOperation(ModuleOp op) {
  if (!op->getAttrs().empty())
    return failure();

  Region &body = op.getBodyRegion();
  if (llvm::range_size(body.getOps()) != 1)
    return failure();

  if (auto plan = llvm::dyn_cast<PlanOp>(&*body.op_begin()))
    return exportOperation(plan);

  return failure();
}

FailureOr<std::unique_ptr<::substrait::Plan>> exportOperation(PlanOp op) {
  // Build `Version` message.
  auto version = std::make_unique<::substrait::Version>();
  version->set_major_number(op.getMajorNumber());
  version->set_minor_number(op.getMinorNumber());
  version->set_patch_number(op.getPatchNumber());
  version->set_producer(op.getProducer().str());
  version->set_git_hash(op.getGitHash().str());

  // Build `Plan` message.
  auto plan = std::make_unique<::substrait::Plan>();
  plan->set_allocated_version(version.release());

  // TODO(ingomueller): build plan content once defined.

  return std::move(plan);
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
      .Default([](auto) { return failure(); });
}

} // namespace

namespace mlir {
namespace substrait {

LogicalResult translateSubstraitToProtobuf(Operation *op,
                                           llvm::raw_ostream &output) {
  FailureOr<std::unique_ptr<pb::Message>> result = exportOperation(op);
  if (failed(result))
    return failure();

  std::string out;
  if (!pb::TextFormat::PrintToString(*result.value(), &out))
    return failure();

  output << out;
  return success();
}

} // namespace substrait
} // namespace mlir
