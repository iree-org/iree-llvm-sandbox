// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   -substrait-protobuf-format=text \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN:   -substrait-protobuf-format=text \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   -substrait-protobuf-format=text \
// RUN: | FileCheck --check-prefix=CHECK-TEXT %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   -substrait-protobuf-format=binary \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN:   -substrait-protobuf-format=binary \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   -substrait-protobuf-format=binary \
// RUN: | FileCheck --check-prefix=CHECK-BINARY %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   -substrait-protobuf-format=json \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN:   -substrait-protobuf-format=json \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   -substrait-protobuf-format=json \
// RUN: | FileCheck --check-prefix=CHECK-JSON %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   -substrait-protobuf-format=pretty-json \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN:   -substrait-protobuf-format=pretty-json \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   -substrait-protobuf-format=pretty-json \
// RUN: | FileCheck --check-prefix=CHECK-PRETTYJSON %s

substrait.plan version 0 : 42 : 1 {}
// CHECK: substrait.plan version 0 : 42 : 1 {

// CHECK-TEXT:      version {
// CHECK-TEXT-NEXT: minor_number: 42

// CHECK-BINARY:  2

// CHECK-JSON:  {"version":{"minorNumber":42,"patchNumber":1}}

// CHECK-PRETTYJSON:      "version": {
// CHECK-PRETTYJSON-NEXT:   "minorNumber": 42,
