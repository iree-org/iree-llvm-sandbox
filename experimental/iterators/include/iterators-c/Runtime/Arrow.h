//===-- Arrow.h - Runtime for handling Apache Arrow data --------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a shallow runtime around the C interfaces of Apache
// Arrow, namely the Arrow C data interface and the Arrow C stream interface
// (see https://arrow.apache.org/docs/format/CDataInterface.html and
// https://arrow.apache.org/docs/format/CStreamInterface.html). While these
// interfaces are already very simple and low-level, the goal of this runtime is
// to simplify its usage even further by doing all error handling and
// verification of current limitations.
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_C_RUNTIME_ARROW_H
#define ITERATORS_C_RUNTIME_ARROW_H

#include "mlir-c/Support.h"

struct ArrowArray;
struct ArrowArrayStream;
struct ArrowSchema;

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Arrow Array (aka Arrow RecordBatch).
//===----------------------------------------------------------------------===//

/// Returns the number of rows in the given Arrow array.
MLIR_CAPI_EXPORTED
int64_t mlirIteratorsArrowArrayGetSize(ArrowArray *array);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is an int8 column.
MLIR_CAPI_EXPORTED const int8_t *
mlirIteratorsArrowArrayGetInt8Column(ArrowArray *array, ArrowSchema *schema,
                                     int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is a uint8 column.
MLIR_CAPI_EXPORTED const uint8_t *
mlirIteratorsArrowArrayGetUInt8Column(ArrowArray *array, ArrowSchema *schema,
                                      int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is an int16 column.
MLIR_CAPI_EXPORTED const int16_t *
mlirIteratorsArrowArrayGetInt16Column(ArrowArray *array, ArrowSchema *schema,
                                      int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is a uint16 column.
MLIR_CAPI_EXPORTED const uint16_t *
mlirIteratorsArrowArrayGetUInt16Column(ArrowArray *array, ArrowSchema *schema,
                                       int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is an int32 column.
MLIR_CAPI_EXPORTED const int32_t *
mlirIteratorsArrowArrayGetInt32Column(ArrowArray *array, ArrowSchema *schema,
                                      int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is a uint32 column.
MLIR_CAPI_EXPORTED const uint32_t *
mlirIteratorsArrowArrayGetUInt32Column(ArrowArray *array, ArrowSchema *schema,
                                       int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is an int64 column.
MLIR_CAPI_EXPORTED const int64_t *
mlirIteratorsArrowArrayGetInt64Column(ArrowArray *array, ArrowSchema *schema,
                                      int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is a uint64 column.
MLIR_CAPI_EXPORTED const uint64_t *
mlirIteratorsArrowArrayGetUInt64Column(ArrowArray *array, ArrowSchema *schema,
                                       int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is a float16 column.
MLIR_CAPI_EXPORTED const uint16_t *
mlirIteratorsArrowArrayGetFloat16Column(ArrowArray *array, ArrowSchema *schema,
                                        int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is a float32 column.
MLIR_CAPI_EXPORTED const float *
mlirIteratorsArrowArrayGetFloat32Column(ArrowArray *array, ArrowSchema *schema,
                                        int64_t i);

/// Returns the raw data pointer to the buffer of the i-th column of the given
/// Arrow array, ensuring that that column is a float64 column.
MLIR_CAPI_EXPORTED const double *
mlirIteratorsArrowArrayGetFloat64Column(ArrowArray *array, ArrowSchema *schema,
                                        int64_t i);

/// Releases the memory owned by the given Arrow array (by calling its release
/// function). Unlike the lower-level release function from the Arrow C
/// interface, this function may be called on already released structs, in which
/// case the release function is not called.
MLIR_CAPI_EXPORTED
void mlirIteratorsArrowArrayRelease(ArrowArray *array);

//===----------------------------------------------------------------------===//
// ArrowSchema.
//===----------------------------------------------------------------------===//

/// Releases the memory owned by the given schema (by calling its release
/// function). Unlike the lower-level release function from the Arrow C
/// interface, this function may be called on already released structs, in which
/// case the release function is not called.
MLIR_CAPI_EXPORTED
void mlirIteratorsArrowSchemaRelease(ArrowSchema *schema);

//===----------------------------------------------------------------------===//
// ArrowArrayStream (aka RecordBatchReader).
//===----------------------------------------------------------------------===//

/// Attempts to extract the next record batch from the given stream. Stores the
/// returned batch in the given result pointer and returns true iff the stream
/// did return a batch. If an error occurs, prints a message and exits.
MLIR_CAPI_EXPORTED
bool mlirIteratorsArrowArrayStreamGetNext(ArrowArrayStream *stream,
                                          ArrowArray *result);

/// Gets the schema for the stream and stores it in the result pointer. If an
/// error occurs, prints a message and exits.
MLIR_CAPI_EXPORTED
void mlirIteratorsArrowArrayStreamGetSchema(ArrowArrayStream *stream,
                                            ArrowSchema *result);

/// Releases the memory owned by the given schema (by calling its release
/// function). Unlike the lower-level release function from the Arrow C
/// interface, this function may be called on already released structs, in which
/// case the release function is not called.
MLIR_CAPI_EXPORTED
void mlirIteratorsArrowArrayStreamRelease(ArrowArrayStream *stream);

#ifdef __cplusplus
}
#endif

#endif // ITERATORS_C_RUNTIME_ARROW_H
