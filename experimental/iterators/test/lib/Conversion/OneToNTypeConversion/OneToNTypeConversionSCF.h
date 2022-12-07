//===-- OneToNTypeConversionSCF.h - 1:N type conversion for scf -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIB_CONVERSION_ONETONTYPECONVERSION_ONETONTYPECONVERSIONSCF_H
#define TEST_LIB_CONVERSION_ONETONTYPECONVERSION_ONETONTYPECONVERSIONSCF_H

namespace mlir {
class TypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace mlir {
namespace iterators {

// Populates the provided pattern set with patterns that do 1:N type conversions
// on (some) SCF ops. This is intended to be used with applyOneToNConversion.
void populateSCFTypeConversionPatterns(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns);

} // namespace iterators
} // namespace mlir

#endif // TEST_LIB_CONVERSION_ONETONTYPECONVERSION_ONETONTYPECONVERSIONSCF_H
