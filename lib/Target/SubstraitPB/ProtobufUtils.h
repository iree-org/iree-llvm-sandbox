//===-- ProtobufUtils.h - Utils for Substrait protobufs ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TARGET_SUBSTRAITPB_PROTOBUFUTILS_H
#define LIB_TARGET_SUBSTRAITPB_PROTOBUFUTILS_H

#include "mlir/IR/Location.h"

namespace substrait::proto {
class RelCommon;
class Rel;
} // namespace substrait::proto

namespace mlir::substrait::protobuf_utils {

/// Extract the `RelCommon` message from any possible `rel_type` message of the
/// given `rel`. Reports errors using the given `loc`.
FailureOr<const ::substrait::proto::RelCommon *>
getCommon(const ::substrait::proto::Rel &rel, Location loc);

/// Extract the `RelCommon` message from any possible `rel_type` message of the
/// given `rel`. Reports errors using the given `loc`.
FailureOr<::substrait::proto::RelCommon *>
getMutableCommon(::substrait::proto::Rel *rel, Location loc);

} // namespace mlir::substrait::protobuf_utils

#endif // LIB_TARGET_SUBSTRAITPB_PROTOBUFUTILS_H
