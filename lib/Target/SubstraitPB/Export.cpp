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
#include <substrait/proto/extensions/extensions.pb.h>
#include <substrait/proto/plan.pb.h>
#include <substrait/proto/type.pb.h>

using namespace mlir;
using namespace mlir::substrait;
using namespace ::substrait;
using namespace ::substrait::proto;

namespace pb = google::protobuf;

namespace {

/// Main structure to drive export from the dialect to protobuf. This class
/// holds the visitor functions for the various ops etc. from the dialect as
/// well as state and utilities around the state that is built up during export.
class SubstraitExporter {
public:
// Declaration for the export function of the given operation type.
//
// We need one such function for most op type that we want to export. The
// `MESSAGE_TYPE` argument corresponds to the protobuf message type returned
// by the function.
#define DECLARE_EXPORT_FUNC(OP_TYPE, MESSAGE_TYPE)                             \
  FailureOr<std::unique_ptr<MESSAGE_TYPE>> exportOperation(OP_TYPE op);

  DECLARE_EXPORT_FUNC(CrossOp, Rel)
  DECLARE_EXPORT_FUNC(EmitOp, Rel)
  DECLARE_EXPORT_FUNC(ExpressionOpInterface, Expression)
  DECLARE_EXPORT_FUNC(FieldReferenceOp, Expression)
  DECLARE_EXPORT_FUNC(FilterOp, Rel)
  DECLARE_EXPORT_FUNC(LiteralOp, Expression)
  DECLARE_EXPORT_FUNC(ModuleOp, Plan)
  DECLARE_EXPORT_FUNC(NamedTableOp, Rel)
  DECLARE_EXPORT_FUNC(PlanOp, Plan)
  DECLARE_EXPORT_FUNC(ProjectOp, Rel)
  DECLARE_EXPORT_FUNC(RelOpInterface, Rel)

  FailureOr<std::unique_ptr<pb::Message>> exportOperation(Operation *op);
  FailureOr<std::unique_ptr<proto::Type>> exportType(Location loc,
                                                     mlir::Type mlirType);

private:
  /// Returns the nearest symbol table to op. The symbol table is cached in
  /// `this` such that repeated calls that request the same symbol do not
  /// rebuild that table.
  SymbolTable &getSymbolTableFor(Operation *op) {
    Operation *nearestSymbolTableOp = SymbolTable::getNearestSymbolTable(op);
    if (!symbolTable || symbolTable->getOp() != nearestSymbolTableOp) {
      symbolTable = std::make_unique<SymbolTable>(nearestSymbolTableOp);
    }
    return *symbolTable;
  }

  /// Looks up the anchor value corresponding to the given symbol name in the
  /// context of the given op. The op is used to determine which symbol table
  /// was used to assign anchors.
  template <typename SymNameType>
  int32_t lookupAnchor(Operation *contextOp, const SymNameType &symName) {
    SymbolTable &symbolTable = getSymbolTableFor(contextOp);
    Operation *calleeOp = symbolTable.lookup(symName);
    return anchorsByOp.at(calleeOp);
  }

  DenseMap<Operation *, int32_t> anchorsByOp{}; // Maps anchors to ops.
  std::unique_ptr<SymbolTable> symbolTable;     // Symbol table cache.
};

FailureOr<std::unique_ptr<proto::Type>>
SubstraitExporter::exportType(Location loc, mlir::Type mlirType) {
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

FailureOr<std::unique_ptr<Rel>> SubstraitExporter::exportOperation(CrossOp op) {
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

FailureOr<std::unique_ptr<Rel>> SubstraitExporter::exportOperation(EmitOp op) {
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
SubstraitExporter::exportOperation(ExpressionOpInterface op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::unique_ptr<Expression>>>(
             op)
      .Case<FieldReferenceOp, LiteralOp>(
          [&](auto op) { return exportOperation(op); })
      .Default(
          [](auto op) { return op->emitOpError("not supported for export"); });
}

FailureOr<std::unique_ptr<Expression>>
SubstraitExporter::exportOperation(FieldReferenceOp op) {
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
    if (failed(expression))
      return failure();

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

FailureOr<std::unique_ptr<Rel>>
SubstraitExporter::exportOperation(FilterOp op) {
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
  assert(yieldOp.getValue().size() == 1 &&
         "fitler op must yield exactly one value");
  auto conditionOp = llvm::dyn_cast_if_present<ExpressionOpInterface>(
      yieldOp.getValue().front().getDefiningOp());
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

FailureOr<std::unique_ptr<Expression>>
SubstraitExporter::exportOperation(LiteralOp op) {
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

FailureOr<std::unique_ptr<Plan>>
SubstraitExporter::exportOperation(ModuleOp op) {
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

FailureOr<std::unique_ptr<Rel>>
SubstraitExporter::exportOperation(NamedTableOp op) {
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

/// Helper for creating unique anchors from symbol names. While in MLIR, symbol
/// names and their references are strings, in Substrait they are integer
/// numbers. In order to preserve the anchor values through an import/export
/// process (without modifications), the symbol names generated during import
/// have the form `<prefix>.<anchor>` such that the `anchor` value can be
/// recovered. During assigning of anchors, the uniquer fills a map mapping the
/// symbol ops to the assigned anchor values such that uses of the symbol can
/// look them up.
class AnchorUniquer {
public:
  AnchorUniquer(StringRef prefix, DenseMap<Operation *, int32_t> &anchorsByOp)
      : prefix(prefix), anchorsByOp(anchorsByOp) {}

  /// Assign a unique anchor to the given op and register the result in the
  /// mapping.
  template <typename OpTy>
  int32_t assignAnchor(OpTy op) {
    StringRef symName = op.getSymName();
    int32_t anchor;
    {
      // Attempt to recover the anchor from the symbol name.
      if (!symName.starts_with(prefix) ||
          symName.drop_front(prefix.size()).getAsInteger(10, anchor)) {
        // If that fails, find one that isn't used yet.
        anchor = nextAnchor;
      }
      // Ensure uniqueness either way.
      while (anchors.contains(anchor))
        anchor = nextAnchor++;
    }
    anchors.insert(anchor);
    auto [_, hasInserted] = anchorsByOp.try_emplace(op, anchor);
    assert(hasInserted && "op had already been assigned an anchor");
    return anchor;
  }

private:
  StringRef prefix;
  DenseMap<Operation *, int32_t> &anchorsByOp; // Maps ops to anchor values.
  DenseSet<int32_t> anchors;                   // Already assigned anchors.
  int32_t nextAnchor{0};                       // Next anchor candidate.
};

/// Traits for common handling of `ExtensionFunctionOp`, `ExtensionTypeOp`, and
/// `ExtensionTypeVariationOp`. While their corresponding protobuf message types
/// are structurally the same, they are (1) different classes and (2) have
/// different field names. The Trait thus provides the message type class as
/// well as accessors for that class for each of the op types.
template <typename OpTy>
struct ExtensionOpTraits;

template <>
struct ExtensionOpTraits<ExtensionFunctionOp> {
  using ExtensionMessageType =
      extensions::SimpleExtensionDeclaration::ExtensionFunction;
  static void setAnchor(ExtensionMessageType &ext, int32_t anchor) {
    ext.set_function_anchor(anchor);
  }
  static ExtensionMessageType *
  getMutableExtension(extensions::SimpleExtensionDeclaration &decl) {
    return decl.mutable_extension_function();
  }
};

template <>
struct ExtensionOpTraits<ExtensionTypeOp> {
  using ExtensionMessageType =
      extensions::SimpleExtensionDeclaration::ExtensionType;
  static void setAnchor(ExtensionMessageType &ext, int32_t anchor) {
    ext.set_type_anchor(anchor);
  }
  static ExtensionMessageType *
  getMutableExtension(extensions::SimpleExtensionDeclaration &decl) {
    return decl.mutable_extension_type();
  }
};

template <>
struct ExtensionOpTraits<ExtensionTypeVariationOp> {
  using ExtensionMessageType =
      extensions::SimpleExtensionDeclaration::ExtensionTypeVariation;
  static void setAnchor(ExtensionMessageType &ext, int32_t anchor) {
    ext.set_type_variation_anchor(anchor);
  }
  static ExtensionMessageType *
  getMutableExtension(extensions::SimpleExtensionDeclaration &decl) {
    return decl.mutable_extension_type_variation();
  }
};

FailureOr<std::unique_ptr<Plan>> SubstraitExporter::exportOperation(PlanOp op) {
  using extensions::SimpleExtensionDeclaration;
  using extensions::SimpleExtensionURI;

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

  // Add `extension_uris` to plan.
  {
    AnchorUniquer anchorUniquer("extension_uri.", anchorsByOp);
    for (auto uriOp : op.getOps<ExtensionUriOp>()) {
      int32_t anchor = anchorUniquer.assignAnchor(uriOp);

      // Create `SimpleExtensionURI` message.
      SimpleExtensionURI *uri = plan->add_extension_uris();
      uri->set_uri(uriOp.getUri().str());
      uri->set_extension_uri_anchor(anchor);
    }
  }

  // Add `extensions` to plan. This requires the URIs to exist.
  {
    // Each extension type has its own anchor uniquer.
    AnchorUniquer funcUniquer("extension_function.", anchorsByOp);
    AnchorUniquer typeUniquer("extension_type.", anchorsByOp);
    AnchorUniquer typeVarUniquer("extension_type_variation.", anchorsByOp);

    // Export an op of a given type using the corresponding uniquer.
    auto exportExtensionOperation = [&](AnchorUniquer *uniquer, auto extOp) {
      using OpTy = decltype(extOp);
      using OpTraits = ExtensionOpTraits<OpTy>;

      // Compute URI reference and anchor value.
      int32_t uriReference = lookupAnchor(op, extOp.getUri());
      int32_t anchor = uniquer->assignAnchor(extOp);

      // Create `SimpleExtensionDeclaration` and extension-specific messages.
      typename OpTraits::ExtensionMessageType ext;
      OpTraits::setAnchor(ext, anchor);
      ext.set_extension_uri_reference(uriReference);
      ext.set_name(extOp.getName().str());
      SimpleExtensionDeclaration *decl = plan->add_extensions();
      *OpTraits::getMutableExtension(*decl) = ext;
    };

    // Iterate over the different types of extension ops. This must be a single
    // loop in order to preserve the order, which allows for interleaving of
    // different types in both the protobuf and the MLIR form.
    for (Operation &extOp : op.getOps()) {
      TypeSwitch<Operation &>(extOp)
          .Case<ExtensionFunctionOp>([&](auto extOp) {
            exportExtensionOperation(&funcUniquer, extOp);
          })
          .Case<ExtensionTypeOp>([&](auto extOp) {
            exportExtensionOperation(&typeUniquer, extOp);
          })
          .Case<ExtensionTypeVariationOp>([&](auto extOp) {
            exportExtensionOperation(&typeVarUniquer, extOp);
          });
    }
  }

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

FailureOr<std::unique_ptr<Rel>>
SubstraitExporter::exportOperation(ProjectOp op) {
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

  // Build `ProjectRel` message.
  auto projectRel = std::make_unique<ProjectRel>();
  projectRel->set_allocated_common(relCommon.release());
  projectRel->set_allocated_input(inputRel->release());

  // Build `Expression` messages.
  auto yieldOp =
      llvm::cast<YieldOp>(op.getExpressions().front().getTerminator());
  for (Value val : yieldOp.getValue()) {
    // Make sure the yielded value was produced by an expression op.
    auto exprRootOp =
        llvm::dyn_cast_if_present<ExpressionOpInterface>(val.getDefiningOp());
    if (!exprRootOp)
      return op->emitOpError(
          "expression not supported for export: yielded op was "
          "not produced by Substrait expression op");

    // Export the expression recursively.
    FailureOr<std::unique_ptr<Expression>> expression =
        exportOperation(exprRootOp);
    if (failed(expression))
      return failure();

    // Add the expression to the `ProjectRel` message.
    *projectRel->add_expressions() = *expression.value();
  }

  // Build `Rel` message.
  auto rel = std::make_unique<Rel>();
  rel->set_allocated_project(projectRel.release());

  return rel;
}

FailureOr<std::unique_ptr<Rel>>
SubstraitExporter::exportOperation(RelOpInterface op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::unique_ptr<Rel>>>(op)
      .Case<
          // clang-format off
          CrossOp,
          EmitOp,
          FieldReferenceOp,
          FilterOp,
          NamedTableOp,
          ProjectOp
          // clang-format on
          >([&](auto op) { return exportOperation(op); })
      .Default([](auto op) {
        op->emitOpError("not supported for export");
        return failure();
      });
}

FailureOr<std::unique_ptr<pb::Message>>
SubstraitExporter::exportOperation(Operation *op) {
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
  SubstraitExporter exporter;
  FailureOr<std::unique_ptr<pb::Message>> result = exporter.exportOperation(op);
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
