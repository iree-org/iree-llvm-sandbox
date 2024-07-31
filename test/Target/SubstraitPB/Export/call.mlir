// RUN: structured-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | structured-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK:      extension_uris {
// CHECK-NEXT:   uri: "http://some.url/with/extensions.yml"
// CHECK-NEXT: }
// CHECK-NEXT: extensions {
// CHECK-NEXT:   extension_function {
// CHECK-NEXT:     name: "somefunc"
// CHECK-NEXT:   }
// CHECK:      extensions {
// CHECK-NEXT:   extension_function {
// CHECK-NEXT:     function_anchor: 1
// CHECK-NEXT:     name: "somefunc"
// CHECK:      relations {
// CHECK-NEXT:   rel {
// CHECK-NEXT:     filter {
// CHECK-NOT:        condition
// CHECK:            condition {
// CHECK-NEXT:         scalar_function {
// CHECK-NEXT:           function_reference: 1
// CHECK-NEXT:           output_type {
// CHECK-NEXT:             bool {
// CHECK-NEXT:               nullability: NULLABILITY_REQUIRED
// CHECK:                arguments {
// CHECK-NEXT:             value {
// CHECK-NEXT:               selection {

substrait.plan version 0 : 42 : 1 {
  extension_uri @extension at "http://some.url/with/extensions.yml"
  extension_function @f1 at @extension["somefunc"]
  extension_function @f2 at @extension["somefunc"]
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = filter %0 : tuple<si32> {
    ^bb0(%arg : tuple<si32>):
      %2 = field_reference %arg[0] : tuple<si32>
      %3 = call @f2(%2) : (si32) -> si1
      yield %3 : si1
    }
    yield %1 : tuple<si32>
  }
}
