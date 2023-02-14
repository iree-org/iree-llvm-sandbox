//===-- Arrow.cpp - Runtime for handling Apache Arrow data ------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators-c/Runtime/Arrow.h"
#include "iterators-c/Runtime/ArrowInterfaces.h"

#include <iostream>
#include <string>
#include <vector>

#include <assert.h>
#include <stdint.h>
#include <string.h>

using namespace std::string_literals;

template <typename T>
static void callReleaseCallbackIfUnreleased(T *ptr) {
  assert(ptr && "need to provide non-null pointer");
  if (ptr->release == nullptr)
    return;
  ptr->release(ptr);
  assert(ptr->release == nullptr &&
         "struct not marked as release after calling release");
}

int64_t mlirIteratorsArrowArrayGetSize(ArrowArray *array) {
  assert(array && "need to provide non-null pointer");
  assert(array->release != nullptr &&
         "provided record batch has been released already");
  return array->length;
}

template <
    typename DataType, const char *kTypeFormat,
    typename BufferPointerType = std::add_pointer_t<std::add_const_t<DataType>>>
static BufferPointerType getColumnImpl(ArrowArray *array, ArrowSchema *schema,
                                       int64_t i) {
  assert(array && "need to provide non-null pointer");
  assert(array->release != nullptr &&
         "provided record batch has been released already");
  assert(array->n_buffers <= 1 &&
         "unexpected number of buffers for struct type");
  assert(schema && "need to provide non-null pointer");
  assert(schema->n_children == array->n_children &&
         "mismatch between provided array and schema");
  assert(schema->format == "+s"s && "only struct arrays are supported");
  assert((schema->flags & ARROW_FLAG_NULLABLE) == 0 &&
         "nullable fields are unsupported");
  assert(i < schema->n_children && "attempt to access non-existing column");

  ArrowArray *childArray = array->children[i];
  ArrowSchema *childSchema = schema->children[i];

  assert(childSchema->n_children == childArray->n_children &&
         "mismatch between provided array and schema");
  assert(childSchema->n_children == 0 && "nested structs not supported");
  assert(childSchema->format == std::string_view(kTypeFormat) &&
         "attempt to access column with wrong type");
  assert((childSchema->flags & ARROW_FLAG_NULLABLE) == 0 &&
         "nullable fields are unsupported");
  assert(childArray->n_buffers == 2 && "unsupported number of buffers");
  assert(childArray->buffers[0] == nullptr && "nullable types not supported");
  assert(childArray->offset == 0 && "offset unsupported");

  return reinterpret_cast<BufferPointerType>(childArray->buffers[1]);
}

const int8_t *mlirIteratorsArrowArrayGetInt8Column(ArrowArray *array,
                                                   ArrowSchema *schema,
                                                   int64_t i) {
  static constexpr const char kFormatString[] = "c";
  return getColumnImpl<int8_t, kFormatString>(array, schema, i);
}

const uint8_t *mlirIteratorsArrowArrayGetUInt8Column(ArrowArray *array,
                                                     ArrowSchema *schema,
                                                     int64_t i) {
  static constexpr const char kFormatString[] = "C";
  return getColumnImpl<uint8_t, kFormatString>(array, schema, i);
}

const int16_t *mlirIteratorsArrowArrayGetInt16Column(ArrowArray *array,
                                                     ArrowSchema *schema,
                                                     int64_t i) {
  static constexpr const char kFormatString[] = "s";
  return getColumnImpl<int16_t, kFormatString>(array, schema, i);
}

const uint16_t *mlirIteratorsArrowArrayGetUInt16Column(ArrowArray *array,
                                                       ArrowSchema *schema,
                                                       int64_t i) {
  static constexpr const char kFormatString[] = "S";
  return getColumnImpl<uint16_t, kFormatString>(array, schema, i);
}

const int32_t *mlirIteratorsArrowArrayGetInt32Column(ArrowArray *array,
                                                     ArrowSchema *schema,
                                                     int64_t i) {
  static constexpr const char kFormatString[] = "i";
  return getColumnImpl<int32_t, kFormatString>(array, schema, i);
}

const uint32_t *mlirIteratorsArrowArrayGetUInt32Column(ArrowArray *array,
                                                       ArrowSchema *schema,
                                                       int64_t i) {
  static constexpr const char kFormatString[] = "I";
  return getColumnImpl<uint32_t, kFormatString>(array, schema, i);
}

const int64_t *mlirIteratorsArrowArrayGetInt64Column(ArrowArray *array,
                                                     ArrowSchema *schema,
                                                     int64_t i) {
  static constexpr const char kFormatString[] = "l";
  return getColumnImpl<int64_t, kFormatString>(array, schema, i);
}

const uint64_t *mlirIteratorsArrowArrayGetUInt64Column(ArrowArray *array,
                                                       ArrowSchema *schema,
                                                       int64_t i) {
  static constexpr const char kFormatString[] = "L";
  return getColumnImpl<uint64_t, kFormatString>(array, schema, i);
}

const uint16_t *mlirIteratorsArrowArrayGetFloat16Column(ArrowArray *array,
                                                        ArrowSchema *schema,
                                                        int64_t i) {
  static constexpr const char kFormatString[] = "e";
  return getColumnImpl<double, kFormatString, const uint16_t *>(array, schema,
                                                                i);
}

const float *mlirIteratorsArrowArrayGetFloat32Column(ArrowArray *array,
                                                     ArrowSchema *schema,
                                                     int64_t i) {
  static constexpr const char kFormatString[] = "f";
  return getColumnImpl<float, kFormatString>(array, schema, i);
}

const double *mlirIteratorsArrowArrayGetFloat64Column(ArrowArray *array,
                                                      ArrowSchema *schema,
                                                      int64_t i) {
  static constexpr const char kFormatString[] = "g";
  return getColumnImpl<double, kFormatString>(array, schema, i);
}

void mlirIteratorsArrowArrayRelease(ArrowArray *schema) {
  callReleaseCallbackIfUnreleased(schema);
}

void mlirIteratorsArrowSchemaRelease(ArrowSchema *schema) {
  callReleaseCallbackIfUnreleased(schema);
}

static void handleError(ArrowArrayStream *stream, int errorCode) {
  const char *errorMessage = stream->get_last_error(stream);
  if (!errorMessage)
    errorMessage = strerror(errorCode);
  std::cerr << "Error while getting next record batch: " << errorMessage
            << std::endl;
  std::exit(1);
}

bool mlirIteratorsArrowArrayStreamGetNext(ArrowArrayStream *stream,
                                          ArrowArray *result) {
  assert(stream && "need to provide non-null pointer");
  assert(result && "need to provide non-null pointer");
  assert(result->release == nullptr &&
         "provided result pointer still owned memory");

  if (int errorCode = stream->get_next(stream, result))
    handleError(stream, errorCode);

  return result->release;
}

void mlirIteratorsArrowArrayStreamGetSchema(ArrowArrayStream *stream,
                                            ArrowSchema *result) {
  assert(stream && "need to provide non-null pointer");
  assert(result && "need to provide non-null pointer");
  assert(result->release == nullptr &&
         "provided result pointer still owned memory");

  if (int errorCode = stream->get_schema(stream, result))
    handleError(stream, errorCode);
}

void mlirIteratorsArrowArrayStreamRelease(ArrowArrayStream *stream) {
  callReleaseCallbackIfUnreleased(stream);
}
