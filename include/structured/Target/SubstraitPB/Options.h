//===-- Options.h - Options for import/and export of Substrait --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_TARGET_SUBSTRAITPB_OPTIONS_H
#define STRUCTURED_TARGET_SUBSTRAITPB_OPTIONS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace substrait {

/// Serialization formats for serialization and deserialization to and from
/// protobuf messages.
enum class SerdeFormat { kText, kBinary, kJson, kPrettyJson };

struct ImportExportOptions {
  /// Specifies which serialization formats is used for serialization and
  /// deserialization to and from protobuf messages.
  SerdeFormat serdeFormat;
};

} // namespace substrait
} // namespace mlir

#endif // STRUCTURED_TARGET_SUBSTRAITPB_OPTIONS_H
