//===- Registration.h - Handle Registration --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_REGISTRATION_H_
#define IREE_LLVM_SANDBOX_REGISTRATION_H_

namespace mlir {
class DialectRegistry;

void registerOutsideOfDialectRegistry();
void registerIntoDialectRegistry(DialectRegistry &registry);
} // namespace mlir

#endif // IREE_LLVM_SANDBOX_REGISTRATION_H_
